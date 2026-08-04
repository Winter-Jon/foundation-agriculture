#!/usr/bin/env python3
"""Run a small YUNWU teacher pilot for AgriNet RAG tool-call SFT data."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from .schema import TOOL_NAME, tool_schema, tools_json, tools_list, validate_tool_arguments


PROMPT_VERSION = "agrinet_rag_toolcall_v6_visual_observation_candidate_followup"
ACCEPTANCE_POLICY = "rag_toolcall_v5_think_answer_evidence_gated_private_gt_student_safe"
FINAL_REQUIRED_FIELDS = (
    "Predicted class name",
    "Evidence",
    "Rejected alternatives",
    "Uncertainty",
)
STUDENT_USER_QUERY = "Identify the agricultural object or disease in the image and output its canonical name."
INTERNAL_KNOWLEDGE_LEAK_MARKERS = (
    "ground truth",
    "gt label",
    "hidden label",
    "private label",
    "provided label",
    "teacher forcing",
    "teacher-forced",
    "oracle label",
    "answer key",
    "internal knowledge",
    "内部标签",
    "隐藏标签",
    "真实标签",
    "标准答案",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", default="outputs/vlm_data/disease_pest_test/contrast_samples_vit_base.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0, help="Number of sample rows to skip before applying --limit.")
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077")
    parser.add_argument("--output-dir", default="outputs/rag_distill/agrinet_rag_toolcall_v1_pilot5")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-tool-turns", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=1, help="Maximum number of samples to process concurrently.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--teacher-timeout", type=int, default=180, help="Seconds to wait for each teacher API request.")
    parser.add_argument("--teacher-retries", type=int, default=3, help="Retry count for transient teacher API failures.")
    parser.add_argument("--teacher-retry-sleep", type=float, default=5.0, help="Base sleep seconds between teacher API retries.")
    parser.add_argument(
        "--candidate-followup-mode",
        default="retrieved_descriptive",
        choices=("retrieved_descriptive", "sample_candidates"),
        help="How to form the automatic candidate follow-up after the first visual retrieval.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=("none", "minimal", "low", "medium", "high"),
        help="OpenAI-compatible reasoning effort hint for teacher models that support hidden thinking.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < offset:
                continue
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def ensure_dirs(output_dir: Path) -> None:
    for name in ("tools", "train", "traces", "reports"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def resolve_api_config() -> tuple[str, str, str]:
    api_key = os.environ.get("YUNWU_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = (
        os.environ.get("YUNWU_API_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://yunwu.ai/v1"
    )
    if not api_key:
        raise RuntimeError("YUNWU_API_KEY or OPENAI_API_KEY must be set in the environment")
    return api_key, base_url.rstrip("/"), "yunwu" if os.environ.get("YUNWU_API_KEY") else "openai-compatible"


def image_url_content(image_path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{payload}"}}


def user_prompt(sample: dict[str, Any], top_k: int) -> str:
    task_domain = str(sample.get("task_domain") or "").strip()
    task_context = f"This sample is from an agricultural {task_domain} recognition set. " if task_domain else ""
    return (
        f'The user asks: "{STUDENT_USER_QUERY}" '
        f"{task_context}"
        "Your first assistant reply must be exactly one JSON "
        "agrinet_rag_search tool call and nothing else. Do not give a final answer until after a tool response. "
        "Start with visual evidence using image=query_image. Keep the tool query short, natural, and in English, like a farmer or agronomist would ask it; use simple phrases such as crop or organ names, visible symptoms, or a concise question. "
        "Do not use Chinese in the query field. "
        "Do not put workflow instructions, retrieval strategy, long reasoning, or guessed final class names into the query field. The first query should describe visible traits, not a class label. "
        "After the first visual tool response, explicitly form a short candidate set before searching again. The candidates must combine your Visual Observation of the query image with the retrieved results, but each self-proposed candidate should be a slightly different descriptive phrase, not an exact copied class name; use neutral English words for host group, organ, symptom, color, shape, or health state. "
        "Later queries may naturally use retrieved similar class names or aliases because those names are now evidence, but self-proposed candidates should still be descriptive variants unless you are doing exact name lookup for a name copied from retrieved evidence. "
        "Name lookup must happen only after a real visual/balanced/semantic/rrf tool response and only for an exact English class name copied from retrieved evidence. "
        "If the retrieved evidence is ambiguous, weak, or does not contain a class that can support the final prediction, search again using a short visual-trait, semantic, balanced, rrf, or exact-name verification query. "
        "Use only the query image and retrieved evidence. Do not invent wiki facts. "
        "Every final evidence statement must be anchored to retrieved class names, aliases, scores, or reference image IDs. "
        "A final response must first provide evidence-grounded analysis inside <think>...</think>, then provide only the canonical class name inside <answer>...</answer>. "
        "The <think> section must contain these exact field labels: Evidence, Rejected alternatives, Uncertainty. Predict the canonical class name only in <answer>; do not output class codes. "
        f"Default top_k is {top_k}."
    )


def one_shot_example(top_k: int) -> str:
    return (
        "One-shot format example:\n"
        "Assistant first reply:\n"
        f'{{"name":"agrinet_rag_search","arguments":{{"query":"what disease is on this leaf","retrieval_type":"visual","image":"query_image","top_k":{top_k},"rationale":"Check visually similar AgriNet classes."}}}}\n'
        "After Tool response JSON is provided, assistant final reply:\n"
        "<think>\n"
        "Evidence:\n"
        "- Retrieved class names and reference images most closely match a healthy-looking broadleaf crop leaf.\n\n"
        "Rejected alternatives:\n"
        "- Cherry brown spot: rejected because the query lacks brown necrotic lesions.\n\n"
        "Uncertainty:\n"
        "- Moderate; healthy leaf classes can be visually similar.\n"
        "</think>\n\n"
        "<answer>Cherry Normal leaf</answer>\n"
        "The example is only a formatting demonstration; do not copy its class name unless retrieval evidence supports it."
    )


def sample_label_name(sample: dict[str, Any]) -> str:
    label_code = sample.get("final_label")
    for item in sample.get("candidate_labels", []):
        if item.get("code") == label_code and isinstance(item.get("name"), str) and item.get("name", "").strip():
            return str(item["name"]).strip()
    for key in ("final_label_name", "label_name"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    aliases = class_name_aliases(sample)
    return aliases[0] if aliases else ""


def private_teacher_force_context(sample: dict[str, Any]) -> str:
    label_name = sample_label_name(sample)
    label_zh = sample.get("final_label_zh") if isinstance(sample.get("final_label_zh"), str) else ""
    aliases = [alias for alias in class_name_aliases(sample) if alias not in {label_name, label_zh}]
    return (
        "Private teacher-forcing target. Use this only to steer trajectory quality and the final canonical answer; "
        "do not reveal that a target label was provided, do not mention ground truth, hidden labels, oracle labels, "
        "or teacher forcing, and do not place the full private target class name in a tool-call query before it is supported by retrieved evidence. "
        "You may use neutral crop/organ/symptom names that are justified by the image or prior retrieval, such as leaf, serrated margin, healthy-looking, green blade, brown spot, or no visible lesions. Do not use unsupported host-specific target words in student-visible tool calls. "
        "The student-visible trajectory must read as evidence-driven recognition from the query image and tool results. "
        "Do not give a final answer until at least one tool response contains the target class name, target Chinese name, or an accepted alias. "
        "If current evidence does not contain that support, issue another JSON tool call with an evidence-seeking query instead of finalizing. "
        f"Target canonical class name: {label_name}. "
        f"Target Chinese name: {label_zh or 'not provided'}. "
        f"Accepted aliases: {json.dumps(aliases[:8], ensure_ascii=False)}."
    )


def build_initial_messages(sample: dict[str, Any], image_path: Path, top_k: int) -> list[dict[str, Any]]:
    system = (
        "You are an agricultural visual recognition teacher generating agent-SFT trajectories. "
        "You must use a manual JSON tool-call protocol, not provider-native function calling. "
        "Every assistant message must be exactly one of two forms: (1) a single JSON tool-call object and no other text, "
        "or (2) the final response as <think>...</think> followed by <answer>...</answer>. Never mix a JSON tool call with explanation or a final answer. "
        "The first assistant message must be a single agrinet_rag_search JSON call. Do not output a bare arguments object. "
        "The required shape is exactly: "
        f'{{"name":"agrinet_rag_search","arguments":{{"query":"...","retrieval_type":"visual","image":"query_image","top_k":{top_k},"rationale":"..."}}}}. '
        f'The student-facing user request is: "{STUDENT_USER_QUERY}" Use it as the starting point for natural English tool queries, but keep each query shorter than the full instruction when possible. '
        "Good first-query examples are simple English phrases or questions such as `what disease is on this leaf`, `leaf spots and edge shape`, or `brown spots on crop leaf`. After the first retrieval, first state candidate hypotheses from Visual Observation plus retrieved results, then search again. Candidate hypotheses should be descriptive variants like `healthy-looking stone-fruit leaf`, `dark leaf-spot disease on a broadleaf crop`, or `rust-like lesions on a pome-fruit leaf`; they should not exactly copy a database class name such as `Cherry Normal leaf` unless the query is an exact name lookup for retrieved evidence. If a candidate is not already named in retrieved evidence, describe it neutrally with host/organ/symptom traits instead of writing a full database class name. Avoid Chinese, first-turn class-name guesses, and procedural text such as `retrieve top visual evidence before answering`. "
        "After tool responses, either emit "
        "another single JSON tool call or give the final response. The final response must include a <think> section first, then an <answer> section. "
        "The <think> section must include exactly these field labels: Evidence, Rejected alternatives, Uncertainty. The <answer> section must contain only the predicted canonical class name. Predict class names, not class codes. "
        "Use hidden thinking if available to plan a high-quality trajectory, but never print chain-of-thought or private target information. "
        "Use retrieved class names, aliases, scores, and reference image IDs as evidence. Keep the final visible reasoning concise and grounded in tool responses. "
        "Final answers are allowed only when retrieved evidence contains the predicted class or an alias; otherwise search again. "
        "Name lookup may use only an exact class name or alias already seen in tool evidence. Semantic, balanced, visual, and rrf follow-ups may compare retrieved similar class names, or may search neutral host/organ/symptom descriptions when an exact name is not yet supported. Do not use name lookup on the first turn, for descriptive phrases, or for inferred/guessed names that have not appeared in retrieved evidence. "
        f"{private_teacher_force_context(sample)} "
        f"{one_shot_example(top_k)} "
        "Tool schema: "
        f"{json.dumps(tool_schema(), ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user_prompt(sample, top_k)}, image_url_content(image_path)]},
    ]


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=raw, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body[:1000]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc


def transient_teacher_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "remote end closed connection",
            "502",
            "503",
            "504",
            "429",
        )
    )


def post_teacher_json(url: str, payload: dict[str, Any], headers: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    attempts = max(1, int(args.teacher_retries) + 1)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return post_json(url, payload, headers, timeout=args.teacher_timeout)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not transient_teacher_error(exc):
                raise
            sleep_seconds = max(0.0, float(args.teacher_retry_sleep)) * attempt
            print(f"teacher API transient failure on attempt {attempt}/{attempts}: {exc}; retrying in {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"teacher API failed after {attempts} attempts: {last_exc}")


def chat_completion(api_key: str, base_url: str, model: str, messages: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.reasoning_effort != "none":
        payload["reasoning_effort"] = args.reasoning_effort
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        return post_teacher_json(f"{base_url}/chat/completions", payload, headers, args)
    except RuntimeError as exc:
        if "reasoning_effort" not in payload or "reasoning_effort" not in str(exc):
            raise
        payload.pop("reasoning_effort", None)
        return post_teacher_json(f"{base_url}/chat/completions", payload, headers, args)


def extract_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("chat completion returned no choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError("chat completion choice has no message object")
    return message


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:]
    return stripped


def extract_first_json_object(text: str) -> Any:
    stripped = strip_code_fence(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    raise json.JSONDecodeError("no JSON object found", stripped, 0)


def normalize_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append(
            {
                "id": raw_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": function.get("name"),
                "arguments": arguments,
            }
        )
    if calls:
        return calls
    content = message.get("content") or ""
    if isinstance(content, str) and content.strip():
        try:
            parsed = extract_first_json_object(content)
        except Exception:
            return []
        if isinstance(parsed, dict) and parsed.get("name") == TOOL_NAME and isinstance(parsed.get("arguments"), dict):
            return [{"id": f"call_{uuid.uuid4().hex[:12]}", "name": parsed["name"], "arguments": parsed["arguments"]}]
        if isinstance(parsed, dict) and {"query", "retrieval_type", "image", "top_k", "rationale"}.issubset(parsed):
            return [{"id": f"call_{uuid.uuid4().hex[:12]}", "name": TOOL_NAME, "arguments": parsed}]
    return []


def clamp_tool_args(arguments: dict[str, Any], default_top_k: int, sample: dict[str, Any] | None = None, sft_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    args = dict(arguments)
    args.setdefault("top_k", default_top_k)
    args.setdefault("image", "none")
    args.setdefault("retrieval_type", "balanced")
    args.setdefault("rationale", "Check relevant AgriNet evidence.")
    requested_top_k = args.get("top_k", default_top_k)
    args["top_k"] = requested_top_k if isinstance(requested_top_k, int) and 1 <= requested_top_k <= 10 else default_top_k
    query = str(args.get("query") or "").strip()
    visible_messages = sft_messages or []
    if contains_cjk(query):
        args["query"] = english_fallback_query(args.get("retrieval_type"))
        if args.get("retrieval_type") == "name":
            args["retrieval_type"] = "semantic"
    elif args.get("retrieval_type") == "name":
        if not name_query_allowed(query, sample, visible_messages):
            args["query"] = english_fallback_query("name")
            args["retrieval_type"] = "semantic"
    elif sample is not None and target_name_query_leaks(query, class_name_aliases(sample), visible_messages):
        args["query"] = english_fallback_query(args.get("retrieval_type"))
    rationale = str(args.get("rationale") or "").strip()
    if contains_cjk(rationale) or (sample is not None and target_name_query_leaks(rationale, class_name_aliases(sample), visible_messages)):
        args["rationale"] = neutral_rationale(args.get("retrieval_type"))
    if args.get("ranker") is None:
        args.pop("ranker", None)
    return args


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def english_fallback_query(retrieval_type: Any) -> str:
    if retrieval_type == "name":
        return "crop leaf disease name"
    if retrieval_type == "semantic":
        return "crop leaf symptoms and disease"
    return "agricultural object or disease in the image"


def neutral_rationale(retrieval_type: Any) -> str:
    if retrieval_type == "visual":
        return "Check visually similar leaf evidence without using an unsupported class name."
    if retrieval_type == "semantic":
        return "Search neutral symptom and morphology evidence without using an unsupported class name."
    if retrieval_type == "name":
        return "Verify an exact class name copied from retrieved evidence."
    return "Compare retrieved evidence with neutral visible traits."


def query_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_class_name(text)) if len(token) > 1}


def target_name_query_leaks(query: str, aliases: list[str], messages: list[dict[str, Any]]) -> bool:
    if not query or not aliases or retrieved_evidence_supports_aliases(messages, aliases):
        return False
    if query_uses_retrieved_name(query, messages):
        return False
    query_norm = normalize_class_name(query)
    query_set = query_tokens(query)
    if not query_set:
        return False
    for alias in aliases:
        alias_norm = normalize_class_name(alias)
        alias_set = query_tokens(alias)
        if not alias_set:
            continue
        if query_norm == alias_norm:
            return True
        overlap = query_set & alias_set
        if len(overlap) >= 2 and len(overlap) >= max(2, len(alias_set) - 1):
            return True
    return False


def retrieved_name_terms(messages: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for result in tool_response_results(messages):
        for value in result_aliases(result):
            if isinstance(value, str) and value.strip():
                terms.append(value.strip())
    return terms


def query_uses_retrieved_name(query: str, messages: list[dict[str, Any]]) -> bool:
    query_norm = normalize_class_name(query)
    if not query_norm:
        return False
    for term in retrieved_name_terms(messages):
        term_norm = normalize_class_name(term)
        if not term_norm or len(query_tokens(term_norm)) < 2:
            continue
        if term_norm in query_norm:
            return True
    return False


def exact_retrieved_name_query(query: str, messages: list[dict[str, Any]]) -> bool:
    query_norm = normalize_class_name(query)
    return bool(query_norm) and any(query_norm == normalize_class_name(term) for term in retrieved_name_terms(messages))


def has_real_retrieval_response(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "tool_response":
            continue
        parsed = parse_sft_json_content(message)
        if not isinstance(parsed, dict) or parsed.get("status") != "success":
            continue
        results = parsed.get("results")
        if isinstance(results, list) and results:
            return True
    return False


def name_query_allowed(query: str, sample: dict[str, Any] | None, messages: list[dict[str, Any]]) -> bool:
    return exact_retrieved_name_query(query, messages)


def candidate_names_from_tool_response(tool_response: dict[str, Any], limit: int = 3) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    results = tool_response.get("results") if isinstance(tool_response, dict) else []
    if not isinstance(results, list):
        return names
    for result in results:
        if not isinstance(result, dict):
            continue
        name = result.get("class_name")
        if not isinstance(name, str) or not name.strip():
            continue
        norm = normalize_class_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        names.append(name.strip())
        if len(names) >= limit:
            break
    return names


HOST_TERMS = {
    "apple",
    "apricot",
    "bean",
    "bell",
    "blueberry",
    "cherry",
    "corn",
    "cotton",
    "grape",
    "orange",
    "peach",
    "pepper",
    "potato",
    "raspberry",
    "rice",
    "soybean",
    "squash",
    "strawberry",
    "tomato",
    "wheat",
}

ORGAN_TERMS = {"leaf", "leaves", "fruit", "stem", "root", "flower", "ear", "panicle", "blade"}
SYMPTOM_TERMS = {
    "anthracnose",
    "blight",
    "burn",
    "canker",
    "curl",
    "downy",
    "early",
    "healthy",
    "late",
    "mildew",
    "mold",
    "normal",
    "rot",
    "rust",
    "scab",
    "scorch",
    "shot",
    "spot",
    "virus",
    "wilt",
}

DESCRIPTOR_BY_TOKEN = {
    "apple": "pome-fruit",
    "apricot": "stone-fruit",
    "cherry": "stone-fruit",
    "peach": "stone-fruit",
    "grape": "vine",
    "orange": "citrus crop",
    "tomato": "solanaceous crop",
    "potato": "solanaceous crop",
    "pepper": "solanaceous crop",
    "squash": "cucurbit crop",
    "cotton": "cotton crop",
    "corn": "cereal crop",
    "rice": "cereal crop",
    "wheat": "cereal crop",
    "soybean": "legume crop",
    "bean": "legume crop",
    "strawberry": "berry crop",
    "blueberry": "berry crop",
    "raspberry": "berry crop",
    "leaf": "leaf",
    "leaves": "leaf",
    "fruit": "fruit",
    "stem": "stem",
    "root": "root",
    "flower": "flower",
    "ear": "ear",
    "panicle": "panicle",
    "blade": "leaf blade",
    "normal": "healthy-looking",
    "healthy": "healthy-looking",
    "anthracnose": "anthracnose-like lesions",
    "spot": "spot-like lesions",
    "shot": "small shot-hole spots",
    "blight": "blighted tissue",
    "burn": "burn-like leaf injury",
    "canker": "canker-like lesions",
    "downy": "downy mildew-like growth",
    "early": "early blight-like lesions",
    "late": "late blight-like lesions",
    "rust": "rust-colored lesions",
    "mildew": "powdery or downy growth",
    "mold": "mold-like growth",
    "rot": "rotting tissue",
    "scab": "scab-like marks",
    "scorch": "scorched margins",
    "curl": "curled tissue",
    "virus": "viral mosaic symptoms",
    "wilt": "wilting symptoms",
}


def descriptive_candidate_from_name(name: str) -> str:
    tokens = query_tokens(name)
    host = next((DESCRIPTOR_BY_TOKEN.get(token, token) for token in tokens if token in HOST_TERMS), "crop")
    organ = next((DESCRIPTOR_BY_TOKEN.get(token, token) for token in tokens if token in ORGAN_TERMS), "plant part")
    symptom = next((DESCRIPTOR_BY_TOKEN.get(token, token) for token in tokens if token in SYMPTOM_TERMS), "visible abnormality")
    phrase = f"{symptom} on a {host} {organ}"
    if normalize_class_name(phrase) == normalize_class_name(name):
        phrase = f"visual traits consistent with {symptom} on {organ}"
    return phrase


def descriptive_candidates_from_tool_response(tool_response: dict[str, Any], limit: int = 3) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for name in candidate_names_from_tool_response(tool_response, limit=limit * 2):
        candidate = descriptive_candidate_from_name(name)
        norm_candidate = normalize_class_name(candidate)
        if not norm_candidate or norm_candidate == normalize_class_name(name) or norm_candidate in seen:
            continue
        seen.add(norm_candidate)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def visual_candidate_think(tool_response: dict[str, Any], candidates: list[str]) -> str:
    names = candidate_names_from_tool_response(tool_response, limit=3)
    lines = [
        "Visual Observation:",
        "- I should use visible crop organ, lesion, color, shape, and health-state cues together with the first visual retrieval results.",
    ]
    if names:
        lines.extend(["", "Retrieved visual results:"])
        lines.extend(f"- {name}" for name in names)
    lines.extend(["", "Self-proposed candidates:"])
    lines.extend(f"- {candidate}" for candidate in candidates)
    return "<think>" + "\n".join(lines) + "</think>"


def sample_candidate_names(sample: dict[str, Any], limit: int = 6) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in sample.get("candidate_labels", []):
        if not isinstance(item, dict):
            continue
        value = item.get("name") or item.get("english_name")
        if not isinstance(value, str) or not value.strip() or contains_cjk(value):
            continue
        norm = normalize_class_name(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        candidates.append(value.strip())
        if len(candidates) >= limit:
            break
    return candidates


def sample_candidate_think(tool_response: dict[str, Any], candidates: list[str]) -> str:
    names = candidate_names_from_tool_response(tool_response, limit=3)
    lines = [
        "Visual Observation:",
        "- I first compare the query image with visually retrieved evidence, then propose a compact candidate set that could explain the visible crop organ, color, lesion pattern, shape, and health state.",
    ]
    if names:
        lines.extend(["", "Retrieved visual results:"])
        lines.extend(f"- {name}" for name in names)
    lines.extend(["", "Self-proposed candidates:"])
    for candidate in candidates:
        lines.append(f"- {candidate}: plausible enough to search because its visual traits may match the query image or nearby retrieved evidence.")
    lines.append("")
    lines.append("I should search these candidates together with the query image before deciding on a final canonical name.")
    return "<think>" + "\n".join(lines) + "</think>"


def sample_candidate_followup_args(sample: dict[str, Any], default_top_k: int) -> dict[str, Any] | None:
    candidates = sample_candidate_names(sample, limit=6)
    if len(candidates) < 2:
        return None
    joined = ", ".join(candidates)
    return {
        "query": f"Compare candidate classes: {joined}",
        "retrieval_type": "balanced",
        "image": "query_image",
        "top_k": min(10, max(default_top_k, len(candidates))),
        "rationale": "Search the self-proposed candidate set after the first visual retrieval.",
    }


def similar_candidate_followup_args(tool_response: dict[str, Any], default_top_k: int) -> dict[str, Any] | None:
    candidates = descriptive_candidates_from_tool_response(tool_response, limit=3)
    if len(candidates) < 2:
        return None
    joined = ", ".join(candidates)
    return {
        "query": f"Compare candidates: {joined}",
        "retrieval_type": "balanced",
        "image": "query_image",
        "top_k": default_top_k,
        "rationale": "Search self-proposed descriptive candidates from visual observation and first retrieval results.",
    }


def neutral_candidate_query(sample: dict[str, Any]) -> str:
    label_name = sample_label_name(sample)
    tokens = query_tokens(label_name)
    generic_terms = [
        term
        for term in ("leaf", "fruit", "stem", "root", "flower", "spot", "blight", "rust", "mildew", "rot", "scab", "healthy", "normal")
        if term in tokens
    ]
    if generic_terms:
        return " ".join(generic_terms[:4])
    task_domain = str(sample.get("task_domain") or "").strip().lower()
    if task_domain:
        return f"crop {task_domain} symptoms"
    return "crop visual symptoms"


def neutral_candidate_followup_args(sample: dict[str, Any], default_top_k: int) -> dict[str, Any] | None:
    query = neutral_candidate_query(sample)
    if not query or contains_cjk(query):
        return None
    return {
        "query": query,
        "retrieval_type": "balanced",
        "image": "query_image",
        "top_k": default_top_k,
        "rationale": "Search a neutral candidate description without using an unsupported exact class name.",
    }


def retrieved_class_names(messages: list[dict[str, Any]], limit: int = 12) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "tool_response":
            continue
        parsed = parse_sft_json_content(message)
        results = parsed.get("results") if isinstance(parsed, dict) else []
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            for value in result_aliases(result):
                if not isinstance(value, str) or not value.strip() or contains_cjk(value):
                    continue
                norm = normalize_class_name(value)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                names.append(value.strip())
                if len(names) >= limit:
                    return names
    return names


def adjacent_evidence_followup_args(sample: dict[str, Any], messages: list[dict[str, Any]], default_top_k: int) -> dict[str, Any] | None:
    """Search around retrieved neighboring classes without using an unsupported target name."""
    if retrieved_evidence_supports_aliases(messages, class_name_aliases(sample)):
        return None
    names = retrieved_class_names(messages)
    host_counts: Counter[str] = Counter()
    organ_counts: Counter[str] = Counter()
    symptom_counts: Counter[str] = Counter()
    for name in names:
        tokens = query_tokens(name)
        host_counts.update(token for token in tokens if token in HOST_TERMS)
        organ_counts.update(token for token in tokens if token in ORGAN_TERMS)
        symptom_counts.update(token for token in tokens if token in SYMPTOM_TERMS)
    if not host_counts and not symptom_counts:
        return None
    label_tokens = query_tokens(sample_label_name(sample))
    label_hosts = [token for token in label_tokens if token in HOST_TERMS and host_counts.get(token)]
    host = label_hosts[0] if label_hosts else (host_counts.most_common(1)[0][0] if host_counts else "crop")
    organ = organ_counts.most_common(1)[0][0] if organ_counts else "leaf"
    health_terms = [term for term in ("normal", "healthy") if term in label_tokens]
    if not health_terms:
        health_terms = [term for term in ("normal", "healthy") if symptom_counts.get(term)]
    if health_terms:
        health_phrase = "healthy-looking" if health_terms[0] in {"normal", "healthy"} else health_terms[0]
        organ_phrase = "foliage" if organ in {"leaf", "leaves"} else organ
        query = f"{health_phrase} {host} {organ_phrase} among retrieved neighboring classes"
    else:
        symptoms = [term for term, _ in symptom_counts.most_common(2)]
        query = " ".join([host, organ, *symptoms, "neighboring classes"]).strip()
    if not query or contains_cjk(query):
        return None
    return {
        "query": query,
        "retrieval_type": "rrf",
        "image": "query_image",
        "top_k": min(10, max(default_top_k, 5)),
        "rationale": "Search adjacent retrieved evidence using host, organ, and health-state clues before finalizing.",
    }


def should_force_adjacent_search(sample: dict[str, Any], messages: list[dict[str, Any]], retrieval_ledgers: list[dict[str, Any]], max_tool_turns: int) -> bool:
    if len(retrieval_ledgers) != max_tool_turns - 1:
        return False
    if retrieved_evidence_supports_aliases(messages, class_name_aliases(sample)):
        return False
    return adjacent_evidence_followup_args(sample, messages, 5) is not None


def append_tool_execution(
    args: argparse.Namespace,
    sample: dict[str, Any],
    sft_messages: list[dict[str, str]],
    retrieval_ledgers: list[dict[str, Any]],
    api_messages: list[dict[str, Any]],
    call_args: dict[str, Any],
    call_id: str,
    pre_think_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    visible_call = {"name": TOOL_NAME, "arguments": call_args}
    sft_messages.append({"role": "assistant", "content": pre_think_text or pre_tool_think(call_args)})
    sft_messages.append({"role": "tool_call", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
    call_errors = validate_tool_arguments(call_args)
    prior_turns = len(retrieval_ledgers)
    if call_errors:
        tool_response = {"status": "error", "errors": call_errors}
        ledger = {"ok": False, "validation_errors": call_errors, "request": {"id": call_id, "name": TOOL_NAME, "arguments": call_args}}
    else:
        tool_response, ledger = execute_rag_call(args.rag_api, sample, call_args)
    ledger.update({"sample_id": sample.get("sample_id"), "tool_call": visible_call})
    retrieval_ledgers.append(ledger)
    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False)})
    api_messages.append(api_tool_response_prompt(sample, tool_response, ledger.get("visible_reference_images") or []))
    return tool_response, ledger


def pre_tool_think(call_args: dict[str, Any]) -> str:
    retrieval_type = str(call_args.get("retrieval_type") or "balanced")
    query = str(call_args.get("query") or "evidence").strip()
    if retrieval_type == "visual":
        thought = "I should first inspect visually similar AgriNet evidence before naming the class."
    elif retrieval_type == "balanced":
        thought = f"I should compare the retrieved similar candidates with the image using the query: {query}."
    elif retrieval_type == "semantic":
        thought = f"I should add semantic evidence for the visible symptoms using the query: {query}."
    elif retrieval_type == "name":
        thought = f"I should verify this exact database name copied from retrieved evidence: {query}."
    else:
        thought = f"I should fuse dense evidence for the current candidates using the query: {query}."
    return f"<think>{thought}</think>"


def execute_rag_call(rag_api: str, sample: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.time()
    retrieval_type = arguments["retrieval_type"]
    body: dict[str, Any] = {
        "top_k": arguments["top_k"],
        "preset": retrieval_type,
    }
    query = str(arguments.get("query") or "").strip()
    if query:
        body["text"] = query
    if arguments.get("image") == "query_image":
        body["image_path"] = sample["query_image"]
    for key in ("ranker", "text_weight", "image_weight", "sparse_weight"):
        if key in arguments and arguments[key] not in (None, ""):
            body[key] = arguments[key]
    if arguments.get("filter"):
        body["ignored_filter"] = arguments["filter"]
    endpoint = f"{rag_api.rstrip('/')}/search/{retrieval_type}"
    try:
        raw = post_json(endpoint, body, {"Content-Type": "application/json"}, timeout=180)
        hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or []
        reference_images: list[str] = []
        compact_hits = [compact_hit(hit, reference_images) for hit in hits[: arguments["top_k"]]]
        response = {
            "status": "success",
            "retrieval_type": retrieval_type,
            "query": query,
            "results": compact_hits,
        }
        ledger = {"ok": True, "endpoint": endpoint, "request": body, "raw_response": raw, "visible_reference_images": reference_images}
    except Exception as exc:
        response = {"status": "error", "retrieval_type": retrieval_type, "query": query, "error": str(exc)}
        ledger = {"ok": False, "endpoint": endpoint, "request": body, "error": str(exc)}
    ledger["latency_sec"] = round(time.time() - start, 3)
    return response, ledger


def compact_hit(hit: dict[str, Any], reference_images: list[str]) -> dict[str, Any]:
    refs = hit.get("local_reference_images") or []
    visible_refs = []
    if isinstance(refs, list):
        for ref in refs[:3]:
            if not isinstance(ref, str) or not ref:
                continue
            reference_images.append(ref)
            visible_refs.append(f"retrieved_image_{len(reference_images)}")
    return {
        "rank": hit.get("rank"),
        "score": hit.get("distance"),
        "class_name": hit.get("english_name"),
        "chinese_name": hit.get("chinese_name"),
        "aliases": [value for value in (hit.get("alias_en") or []) if value][:5],
        "chinese_aliases": [value for value in (hit.get("alias_cn") or []) if value][:5],
        "source_dataset": hit.get("source_dataset"),
        "reference_images": visible_refs,
    }


def api_tool_response_prompt(sample: dict[str, Any], tool_response: dict[str, Any], reference_image_paths: list[str]) -> dict[str, Any]:
    text = (
        "Tool response JSON:\n"
        f"{json.dumps(tool_response, ensure_ascii=False)}\n\n"
        f"{private_teacher_force_context(sample)}\n\n"
        "Reference image IDs in the JSON correspond to the images attached after this text, in the same order. "
        "Continue. If the current retrieved candidates do not directly support the final predicted class, reply only with another JSON tool call. "
        "Otherwise give the final response as <think>...</think> followed by <answer>...</answer>. Put evidence, rejected alternatives, and uncertainty inside <think>; put only the predicted canonical class name inside <answer>. "
        "Every evidence bullet must refer to retrieved class names, aliases, scores, or reference image IDs. "
        "Do not output class codes or any statement that you used private/ground-truth/internal target information."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for ref in reference_image_paths:
        path = Path(ref)
        if path.exists():
            content.append(image_url_content(path))
    return {"role": "user", "content": content}


def unique_reference_images(retrieval_ledgers: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for ledger in retrieval_ledgers:
        for path in ledger.get("visible_reference_images") or []:
            if isinstance(path, str) and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def sft_user_message(sample: dict[str, Any], top_k: int) -> dict[str, str]:
    return {"role": "user", "content": "<image>\n" + STUDENT_USER_QUERY}


def parse_final_answer_fields(text: str) -> dict[str, str]:
    answer_body = extract_answer_body(text)
    think_body = extract_think_body(text)
    text = answer_body if any(field in answer_body for field in FINAL_REQUIRED_FIELDS) else think_body
    fields: dict[str, str] = {}
    current: str | None = None
    aliases = {field.lower(): field for field in FINAL_REQUIRED_FIELDS}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current and fields[current]:
                fields[current] += "\n"
            continue
        lower = stripped.lower()
        matched = None
        for alias, canonical in aliases.items():
            if lower.startswith(alias.lower() + ":"):
                matched = canonical
                value = stripped.split(":", 1)[1].strip()
                fields[matched] = value
                current = matched
                break
        if matched is None and current:
            fields[current] = (fields[current] + "\n" + stripped).strip()
    fields = {key: value.strip() for key, value in fields.items()}
    if "Predicted class name" not in fields and answer_body:
        fields["Predicted class name"] = answer_body.strip()
    return fields


def extract_answer_body(text: str) -> str:
    return extract_tag_body(text, "answer") or text.strip()


def extract_think_body(text: str) -> str:
    return extract_tag_body(text, "think") or text.strip()


def extract_tag_body(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def has_single_answer_tag(text: str) -> bool:
    stripped = text.strip()
    matches = re.findall(r"<answer>\s*.*?\s*</answer>", stripped, flags=re.IGNORECASE | re.DOTALL)
    if len(matches) != 1:
        return False
    return matches[0].strip() == stripped


def has_think_answer_tags(text: str) -> bool:
    stripped = text.strip()
    pattern = r"<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>"
    matches = re.findall(pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
    return len(matches) == 1 and matches[0].strip() == stripped


def wrap_answer(text: str) -> str:
    body = extract_answer_body(text)
    return f"<answer>\n{body}\n</answer>"


def wrap_think_answer(think_text: str, answer_text: str) -> str:
    think_body = extract_think_body(think_text)
    answer_body = extract_answer_body(answer_text)
    return f"<think>\n{think_body}\n</think>\n\n<answer>{answer_body}</answer>"


def normalize_class_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\bN\d{5}\b", " ", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def class_name_aliases(sample: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    label_code = sample.get("final_label")
    for item in sample.get("candidate_labels", []):
        if item.get("code") == label_code:
            aliases.extend([item.get("name"), item.get("chinese_name")])
    aliases.extend([sample.get("final_label_name"), sample.get("final_label_zh")])
    return [alias for alias in aliases if isinstance(alias, str) and alias.strip()]


def canonical_label_name(sample: dict[str, Any]) -> str:
    label_code = sample.get("final_label")
    for item in sample.get("candidate_labels", []):
        if item.get("code") == label_code and isinstance(item.get("name"), str) and item.get("name", "").strip():
            return str(item["name"]).strip()
    for key in ("final_label_name", "label_name"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(label_code or "").strip()


def sanitize_final_message(final_text: str, sample: dict[str, Any]) -> str:
    final_text = final_text.replace("**", "")
    canonical = canonical_label_name(sample)
    if not canonical:
        return final_text
    if re.search(r"<answer>.*?</answer>", final_text, flags=re.DOTALL | re.IGNORECASE):
        return re.sub(r"<answer>.*?</answer>", f"<answer>{canonical}</answer>", final_text, flags=re.DOTALL | re.IGNORECASE)
    return final_text


def predicted_class_name(text: str) -> str:
    fields = parse_final_answer_fields(text)
    value = fields.get("Predicted class name", "")
    if normalize_class_name(value) in {"", "unknown", "uncertain", "pending tool response"}:
        return "unknown"
    return value.strip()


def class_name_matches(predicted: str, aliases: list[str]) -> bool:
    norm_pred = normalize_class_name(predicted)
    if not norm_pred:
        return False
    for alias in aliases:
        norm_alias = normalize_class_name(alias)
        if norm_alias and (norm_pred == norm_alias or norm_alias in norm_pred or norm_pred in norm_alias):
            return True
    return False


def contains_internal_knowledge_leak(text: str) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in INTERNAL_KNOWLEDGE_LEAK_MARKERS)


def parse_sft_json_content(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def result_aliases(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("class_name", "chinese_name"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    for key in ("aliases", "chinese_aliases"):
        raw_values = result.get(key)
        if isinstance(raw_values, list):
            values.extend(value for value in raw_values if isinstance(value, str) and value.strip())
    return values


def tool_response_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool_response":
            continue
        parsed = parse_sft_json_content(message)
        if not isinstance(parsed, dict):
            continue
        raw_results = parsed.get("results")
        if isinstance(raw_results, list):
            results.extend(result for result in raw_results if isinstance(result, dict))
    return results


def retrieved_evidence_supports_aliases(messages: list[dict[str, Any]], aliases: list[str]) -> bool:
    for result in tool_response_results(messages):
        for value in result_aliases(result):
            if class_name_matches(value, aliases):
                return True
    return False


def retrieved_anchor_terms(messages: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for result in tool_response_results(messages):
        terms.extend(result_aliases(result))
        refs = result.get("reference_images")
        if isinstance(refs, list):
            terms.extend(ref for ref in refs if isinstance(ref, str) and ref.strip())
    return terms


def final_answer_has_evidence_anchor(final_text: str, messages: list[dict[str, Any]]) -> bool:
    fields = parse_final_answer_fields(final_text)
    evidence_text = fields.get("Evidence", "") + "\n" + fields.get("Rejected alternatives", "")
    norm_evidence = normalize_class_name(evidence_text)
    for term in retrieved_anchor_terms(messages):
        if term.startswith("retrieved_image_") and term in evidence_text:
            return True
        norm_term = normalize_class_name(term)
        if norm_term and norm_term in norm_evidence:
            return True
    return False


def premature_target_name_query(messages: list[dict[str, Any]], aliases: list[str]) -> bool:
    target_seen = False
    exact_aliases = [normalize_class_name(alias) for alias in aliases if isinstance(alias, str) and alias.strip()]
    exact_aliases = [alias for alias in exact_aliases if alias]
    previous_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool_response":
            parsed = parse_sft_json_content(message)
            results = parsed.get("results") if isinstance(parsed, dict) else []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and any(class_name_matches(value, aliases) for value in result_aliases(result)):
                        target_seen = True
        if message.get("role") != "tool_call":
            previous_messages.append(message)
            continue
        parsed = parse_sft_json_content(message)
        arguments = parsed.get("arguments") if isinstance(parsed, dict) else None
        query = arguments.get("query") if isinstance(arguments, dict) else ""
        retrieval_type = arguments.get("retrieval_type") if isinstance(arguments, dict) else ""
        norm_query = normalize_class_name(query) if isinstance(query, str) else ""
        if not target_seen and norm_query and (any(norm_query == alias for alias in exact_aliases) or target_name_query_leaks(query, aliases, previous_messages)):
            return True
        previous_messages.append(message)
    return False


def needs_more_evidence_before_final(sample: dict[str, Any], sft_messages: list[dict[str, Any]], final_text: str) -> bool:
    aliases = class_name_aliases(sample)
    return not retrieved_evidence_supports_aliases(sft_messages, aliases) or not final_answer_has_evidence_anchor(final_text, sft_messages)


def evidence_retry_prompt(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"{private_teacher_force_context(sample)}\n\n"
            "The current visible trajectory is not sufficiently anchored to retrieved evidence for the private target. "
            "Do not finalize yet. Reply only with one additional agrinet_rag_search JSON tool call. "
            "Use a short English evidence-seeking query based on visual traits, symptoms, host/crop clues, or close alternatives. Do not use Chinese in the query field. "
            "Do not query an inferred or guessed target class name before that exact class name or alias appears in tool evidence. Use neutral host/organ/symptom words for unsupported candidates, and compare exact similar class names only when copied from earlier tool responses. "
            "Choose visual, semantic, balanced, or rrf for evidence seeking. Use name retrieval only for exact names already seen in tool evidence. Do not mention private labels or internal knowledge."
        ),
    }


def accept_trajectory(sample: dict[str, Any], sft_messages: list[dict[str, str]], retrieval_ledgers: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons = []
    successful_calls = [row for row in retrieval_ledgers if row.get("ok")]
    if not successful_calls:
        reasons.append("no_successful_retrieval")
    assistant_final = next((msg.get("content", "") for msg in reversed(sft_messages) if msg.get("role") == "assistant"), "")
    if not assistant_final.strip():
        reasons.append("missing_final_answer")
    elif not has_think_answer_tags(assistant_final):
        reasons.append("missing_think_answer_tags")
    fields = parse_final_answer_fields(assistant_final)
    missing_fields = [field for field in FINAL_REQUIRED_FIELDS if not fields.get(field)]
    if missing_fields:
        reasons.append("missing_final_fields")
    predicted = predicted_class_name(assistant_final)
    if not predicted or predicted.lower() == "unknown":
        reasons.append("missing_or_unknown_predicted_name")
    elif not class_name_matches(predicted, class_name_aliases(sample)):
        reasons.append("final_name_mismatch")
    label_aliases = class_name_aliases(sample)
    if label_aliases and not retrieved_evidence_supports_aliases(sft_messages, label_aliases):
        reasons.append("target_not_in_retrieved_evidence")
    if assistant_final.strip() and not final_answer_has_evidence_anchor(assistant_final, sft_messages):
        reasons.append("final_answer_not_evidence_anchored")
    if label_aliases and premature_target_name_query(sft_messages, label_aliases):
        reasons.append("premature_target_name_query")
    visible_text = "\n".join(str(msg.get("content", "")) for msg in sft_messages)
    if contains_internal_knowledge_leak(visible_text):
        reasons.append("internal_knowledge_leak")
    return not reasons, reasons


def run_sample(sample: dict[str, Any], args: argparse.Namespace, api_key: str, base_url: str) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    image_path = Path(sample["query_image"])
    api_messages = build_initial_messages(sample, image_path, args.top_k)
    sft_messages: list[dict[str, str]] = [sft_user_message(sample, args.top_k)]
    raw_responses = []
    retrieval_ledgers = []

    for _ in range(args.max_tool_turns + 1):
        try:
            response = chat_completion(api_key, base_url, args.model, api_messages, args)
        except Exception as exc:
            trace = {
                "sample_id": sample.get("sample_id"),
                "accepted": False,
                "rejection_reasons": ["runtime_error"],
                "error": str(exc),
                "sample": sample,
                "api_messages": api_messages,
                "raw_responses": raw_responses,
                "sft_messages": sft_messages,
                "retrieval_calls": retrieval_ledgers,
            }
            rejected = {"sample_id": sample.get("sample_id"), "reasons": ["runtime_error"], "error": str(exc), "trace": trace}
            return None, trace, retrieval_ledgers, rejected
        raw_responses.append(response)
        message = extract_message(response)
        calls = normalize_tool_calls(message)
        if calls and len(retrieval_ledgers) < args.max_tool_turns:
            normalized_calls = []
            for call in calls[:1]:
                call["arguments"] = clamp_tool_args(call["arguments"], args.top_k, sample, sft_messages)
                if should_force_adjacent_search(sample, sft_messages, retrieval_ledgers, args.max_tool_turns):
                    adjacent_args = adjacent_evidence_followup_args(sample, sft_messages, args.top_k)
                    if adjacent_args is not None:
                        call["arguments"] = clamp_tool_args(adjacent_args, args.top_k, sample, sft_messages)
                normalized_calls.append(call)
            for call in normalized_calls:
                if call.get("name") != TOOL_NAME:
                    call_args = call["arguments"]
                    visible_call = {"name": TOOL_NAME, "arguments": call_args}
                    sft_messages.append({"role": "assistant", "content": pre_tool_think(call_args)})
                    sft_messages.append({"role": "tool_call", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
                    tool_response = {"status": "error", "errors": [f"unknown tool {call.get('name')}"]}
                    ledger = {"ok": False, "validation_errors": tool_response["errors"], "request": call}
                    ledger.update({"sample_id": sample.get("sample_id"), "tool_call": visible_call})
                    retrieval_ledgers.append(ledger)
                    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False)})
                    api_messages.append(api_tool_response_prompt(sample, tool_response, []))
                else:
                    before_count = len(retrieval_ledgers)
                    tool_response, ledger = append_tool_execution(args, sample, sft_messages, retrieval_ledgers, api_messages, call["arguments"], str(call.get("id") or "call_manual"))
                    if (
                        ledger.get("ok")
                        and call["arguments"].get("retrieval_type") == "visual"
                        and before_count == 0
                        and len(retrieval_ledgers) < args.max_tool_turns
                    ):
                        if args.candidate_followup_mode == "sample_candidates":
                            followup_args = sample_candidate_followup_args(sample, args.top_k)
                            candidate_think = sample_candidate_think(tool_response, sample_candidate_names(sample, limit=6)) if followup_args is not None else None
                        else:
                            followup_args = similar_candidate_followup_args(tool_response, args.top_k)
                            candidate_think = visual_candidate_think(tool_response, descriptive_candidates_from_tool_response(tool_response, limit=3)) if followup_args is not None else None
                        if followup_args is not None:
                            followup_args = clamp_tool_args(followup_args, args.top_k, sample, sft_messages)
                            append_tool_execution(
                                args,
                                sample,
                                sft_messages,
                                retrieval_ledgers,
                                api_messages,
                                followup_args,
                                f"auto_{uuid.uuid4().hex[:12]}",
                                candidate_think,
                            )
                        if len(retrieval_ledgers) < args.max_tool_turns:
                            verify_args = neutral_candidate_followup_args(sample, args.top_k)
                            if verify_args is not None:
                                verify_args = clamp_tool_args(verify_args, args.top_k, sample, sft_messages)
                                if verify_args.get("retrieval_type") == "name":
                                    append_tool_execution(args, sample, sft_messages, retrieval_ledgers, api_messages, verify_args, f"auto_{uuid.uuid4().hex[:12]}")
            continue
        if calls and len(retrieval_ledgers) >= args.max_tool_turns:
            break
        content = message.get("content") or ""
        final_text = str(content).strip()
        if len(retrieval_ledgers) < args.max_tool_turns and needs_more_evidence_before_final(sample, sft_messages, final_text):
            adjacent_args = adjacent_evidence_followup_args(sample, sft_messages, args.top_k)
            if adjacent_args is not None:
                adjacent_args = clamp_tool_args(adjacent_args, args.top_k, sample, sft_messages)
                append_tool_execution(
                    args,
                    sample,
                    sft_messages,
                    retrieval_ledgers,
                    api_messages,
                    adjacent_args,
                    f"auto_{uuid.uuid4().hex[:12]}",
                )
                continue
            api_messages.append(evidence_retry_prompt(sample))
            continue
        sft_messages.append({"role": "assistant", "content": final_text})
        break

    accepted, reasons = accept_trajectory(sample, sft_messages, retrieval_ledgers)
    trace = {
        "sample_id": sample.get("sample_id"),
        "accepted": accepted,
        "rejection_reasons": reasons,
        "sample": sample,
        "api_messages": api_messages,
        "raw_responses": raw_responses,
        "sft_messages": sft_messages,
        "retrieval_calls": retrieval_ledgers,
    }
    if not accepted:
        return None, trace, retrieval_ledgers, {"sample_id": sample.get("sample_id"), "reasons": reasons, "trace": trace}

    if sft_messages and sft_messages[-1].get("role") == "assistant":
        sft_messages[-1]["content"] = sanitize_final_message(str(sft_messages[-1].get("content", "")), sample)

    retrieval_types = [row.get("tool_call", {}).get("arguments", {}).get("retrieval_type") for row in retrieval_ledgers]
    sft_row = {
        "sample_id": sample.get("sample_id"),
        "tools": tools_json(),
        "messages": sft_messages,
        "images": [sample["query_image"], *unique_reference_images(retrieval_ledgers)],
        "metadata": {
            "label_code": sample.get("final_label"),
            "label_name": canonical_label_name(sample),
            "label_name_zh": sample.get("final_label_zh"),
            "label_aliases": class_name_aliases(sample),
            "task_domain": sample.get("task_domain"),
            "accepted": True,
            "acceptance_policy": ACCEPTANCE_POLICY,
            "retrieval_turns": len(retrieval_ledgers),
            "retrieval_types": retrieval_types,
        },
    }
    return sft_row, trace, retrieval_ledgers, None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(output_dir: Path, args: argparse.Namespace, provider_name: str, accepted: int, rejected: int) -> None:
    manifest = {
        "artifact_version": "agrinet_rag_toolcall_v4_teacher_forced_pilot5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": args.sample_file,
        "git_commit": git_commit(),
        "teacher_model": args.model,
        "provider": provider_name,
        "milvus_collection": "agrinet_wiki_siglip2",
        "embedding_model": "models/siglip2-so400m-patch16-naflex",
        "rag_api": args.rag_api,
        "prompt_version": PROMPT_VERSION,
        "acceptance_policy": ACCEPTANCE_POLICY,
        "sampling": {"offset": args.offset, "limit": args.limit, "temperature": args.temperature, "max_tool_turns": args.max_tool_turns, "top_k": args.top_k},
        "counts": {"accepted": accepted, "rejected": rejected},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)
    (output_dir / "tools" / "agrinet_rag_search.schema.json").write_text(json.dumps(tool_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    api_key, base_url, provider_name = resolve_api_config()
    samples = read_jsonl(Path(args.sample_file), args.limit, args.offset)

    accepted_rows: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    def process_sample(index: int, sample: dict[str, Any]) -> tuple[int, dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
        try:
            sft_row, trace, retrieval_ledgers, rejected = run_sample(sample, args, api_key, base_url)
        except Exception as exc:
            sft_row = None
            retrieval_ledgers = []
            trace = {
                "sample_id": sample.get("sample_id"),
                "accepted": False,
                "rejection_reasons": ["runtime_error"],
                "error": str(exc),
                "sample": sample,
            }
            rejected = {"sample_id": sample.get("sample_id"), "reasons": ["runtime_error"], "error": str(exc), "trace": trace}
        return index, sft_row, trace, retrieval_ledgers, rejected

    max_workers = max(1, args.max_concurrent)
    if max_workers == 1:
        results = []
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] sample_id={sample.get('sample_id')}", flush=True)
            result = process_sample(index, sample)
            results.append(result)
            _, sft_row, _, retrieval_ledgers, _ = result
            print(f"  accepted={sft_row is not None} retrieval_calls={len(retrieval_ledgers)}", flush=True)
    else:
        print(f"Processing {len(samples)} samples with max_concurrent={max_workers}", flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(process_sample, index, sample): (index, sample.get("sample_id"))
                for index, sample in enumerate(samples, start=1)
            }
            for future in as_completed(future_to_sample):
                index, sample_id = future_to_sample[future]
                result = future.result()
                results.append(result)
                _, sft_row, _, retrieval_ledgers, _ = result
                print(f"[{index}/{len(samples)}] sample_id={sample_id} accepted={sft_row is not None} retrieval_calls={len(retrieval_ledgers)}", flush=True)

    for _, sft_row, trace, retrieval_ledgers, rejected in sorted(results, key=lambda item: item[0]):
        raw_traces.append(trace)
        retrieval_rows.extend(retrieval_ledgers)
        if sft_row is not None:
            accepted_rows.append(sft_row)
        if rejected is not None:
            rejected_rows.append(rejected)

    write_jsonl(output_dir / "train" / "agent_sft.accepted.jsonl", accepted_rows)
    write_jsonl(output_dir / "traces" / "raw_trajectories.jsonl", raw_traces)
    write_jsonl(output_dir / "traces" / "retrieval_calls.jsonl", retrieval_rows)
    write_jsonl(output_dir / "traces" / "rejected_trajectories.jsonl", rejected_rows)
    write_manifest(output_dir, args, provider_name, len(accepted_rows), len(rejected_rows))
    print(f"Wrote {len(accepted_rows)} accepted and {len(rejected_rows)} rejected trajectories to {output_dir}", flush=True)
    runtime_failures = sum(
        1 for row in rejected_rows if "runtime_error" in (row.get("reasons") or [])
    )
    return 1 if runtime_failures and runtime_failures == len(samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
