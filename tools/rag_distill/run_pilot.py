#!/usr/bin/env python3
"""Run a small YUNWU teacher pilot for AgriNet RAG tool-call SFT data."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import signal
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from agrinet.data.retrieval_strategies import strategy_spec
from urllib import error, request, parse

from .schema import TOOL_NAME, tool_schema, tools_json, tools_list, validate_tool_arguments


PROMPT_VERSION = "agrinet_rag_toolcall_v6_visual_observation_candidate_followup"
ACCEPTANCE_POLICY = "rag_toolcall_v5_think_answer_evidence_gated_private_gt_student_safe"
DEFAULT_TEACHER_MODEL = "gpt-5.6-luna"


class UnknownTeacherDelivery(RuntimeError):
    """The POST may have reached the provider, so replaying it is unsafe."""


FINAL_REQUIRED_FIELDS = (
    "Predicted class name",
    "Evidence",
    "Rejected alternatives",
    "Uncertainty",
)
FINAL_FIELD_ALIASES = {
    "预测类别名称": "Predicted class name", "证据": "Evidence",
    "排除的候选": "Rejected alternatives", "不确定性": "Uncertainty",
}
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
    parser.add_argument("--plan-file", help="Optional recovery Pilot candidate-attempt JSONL.")
    parser.add_argument("--candidate-source", help="Source JSONL used to hydrate plan rows.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0, help="Number of sample rows to skip before applying --limit.")
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077")
    parser.add_argument("--output-dir", default="outputs/rag_distill/agrinet_rag_toolcall_v1_pilot5")
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
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
    parser.add_argument("--preflight-only", action="store_true", help="Check teacher endpoint reachability and exit without sampling.")
    parser.add_argument("--preflight-image", action="store_true", help="Additionally send one real image request and require a valid manual JSON tool call.")
    parser.add_argument("--preflight-report", help="Write bounded preflight evidence without prompts, credentials, or image payloads.")
    parser.add_argument("--image-max-side", type=int, default=0, help="Optional in-memory resize for multimodal teacher payloads; 0 preserves source resolution.")
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


def read_plan_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.plan_file:
        return read_jsonl(Path(args.sample_file), args.limit, args.offset)
    if not args.candidate_source:
        raise RuntimeError("--plan-file requires --candidate-source")
    source_rows = read_jsonl(Path(args.candidate_source), 10**9, 0)
    by_id = {
        str(row.get("sample_id") or row.get("source_sample_id")): row
        for row in source_rows
        if row.get("sample_id") or row.get("source_sample_id")
    }
    plan_rows = read_jsonl(Path(args.plan_file), args.limit, args.offset)
    samples = []
    for plan in plan_rows:
        source_id = str(plan.get("source_sample_id") or "")
        if source_id not in by_id:
            raise RuntimeError(f"plan source sample is absent: {source_id}")
        sample = {**by_id[source_id], **plan}
        candidate_index = plan.get("candidate_index", 0)
        sample["sample_id"] = str(plan.get("sample_id") or f"{plan['target_id']}-candidate-{candidate_index}")
        samples.append(sample)
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_progress_checkpoint(
    output_dir: Path,
    results: list[tuple[int, dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]],
) -> None:
    """Persist completed sample evidence without making it trainable."""
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: item[0])
    accepted = [row for _, row, _, _, _ in ordered if row is not None]
    traces = [trace for _, _, trace, _, _ in ordered]
    retrievals = [ledger for _, _, _, ledgers, _ in ordered for ledger in ledgers]
    rejected = [row for _, _, _, _, row in ordered if row is not None]
    write_jsonl(progress_dir / "accepted_candidates.jsonl", accepted)
    write_jsonl(progress_dir / "raw_trajectories.jsonl", traces)
    write_jsonl(progress_dir / "retrieval_calls.jsonl", retrievals)
    write_jsonl(progress_dir / "rejected_trajectories.jsonl", rejected)
    write_run_status(
        output_dir,
        "running",
        delivery_status="in_progress",
        completed_samples=len(ordered),
        accepted_observed=len(accepted),
        rejected_observed=len(rejected),
        progress_checkpoint=str(progress_dir),
        training_eligible=False,
    )


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
    base_url = base_url.rstrip("/")
    # OpenAI-compatible relay sites expose the API below /v1 while their root
    # serves an HTML landing page. Preserve explicit paths and normalize bare
    # host URLs so preflight and sampling hit the JSON API route.
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return api_key, base_url, "yunwu" if os.environ.get("YUNWU_API_KEY") else "openai-compatible"


def image_url_content(image_path: Path, max_side: int = 0) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    raw = image_path.read_bytes()
    if max_side > 0:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB")
                image.thumbnail((max_side, max_side))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=88, optimize=True)
                raw = encoded.getvalue()
                mime = "image/jpeg"
        except Exception as exc:
            raise RuntimeError(f"failed to resize teacher image {image_path}: {exc}") from exc
    payload = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{payload}"}}


def sample_strategy(sample: dict[str, Any], fallback_top_k: int) -> tuple[str, tuple[str, ...], int]:
    strategy_id = str(sample.get("strategy_id") or "legacy_visual")
    spec = strategy_spec(sample.get("strategy_id"), fallback_top_k)
    sequence = tuple(str(value) for value in sample.get("preferred_sequence", spec["sequence"]))
    top_k = int(sample.get("top_k") or spec["top_k"] or fallback_top_k)
    return strategy_id, sequence, top_k


def public_option_question(sample: dict[str, Any]) -> str:
    language = str(sample.get("language") or "en")
    choices = [str(item.get("name") or "").strip() for item in sample.get("candidate_labels", [])]
    if len(choices) != 4 or any(not choice for choice in choices):
        raise RuntimeError("option sample requires four named public candidates")
    answer = canonical_label_name(sample)
    correct_option = str(sample.get("correct_option") or "")
    if answer not in choices or correct_option not in {"A", "B", "C", "D"}:
        raise RuntimeError("option sample requires a target candidate and one A-D correct option")
    choices.remove(answer)
    choices.insert(ord(correct_option) - ord("A"), answer)
    prompt = (
        "请选择图中病虫害的规范名称，只在答案标签中输出选项字母。"
        if language == "zh"
        else "Select the canonical name shown in the image; output only the option letter in the answer tag."
    )
    return prompt + "\n" + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))


def option_contract_retry_prompt(sample: dict[str, Any], previous_final: str) -> str:
    previous_answer = extract_answer_body(previous_final).strip()
    public_question = public_option_question(sample)
    if sample.get("language") == "zh":
        return (
            "选项答案格式重试：下面再次给出与用户题面完全相同的公开选项。保持你上一轮基于图像和检索证据选中的类别语义，"
            "仅把它映射为对应的公开选项字母；不要重新猜测其他类别。不要再调用工具，不要在 <answer> 中写类别名称，"
            "也不要使用任何私有标签。现在重新输出完整的 <think>...</think><answer>...</answer>。\n"
            f"你上一轮的公开答案内容：{previous_answer or '未提供'}\n"
            f"公开题面与选项：\n{public_question}"
        )
    return (
        "Option answer-format retry: the same public choices from the user question are repeated below. Preserve the class semantics you selected from the image and retrieved evidence, "
        "and map that selected class to its corresponding public option letter; do not guess a different class. Do not call another tool, do not add a class name in <answer>, "
        "and do not use any private label. Re-output the complete <think>...</think><answer>...</answer> response now.\n"
        f"Your previous public answer content: {previous_answer or 'not provided'}\n"
        f"Public question and choices:\n{public_question}"
    )


def retrieval_budget_finalization_prompt(sample: dict[str, Any]) -> str:
    public_context = ""
    if sample.get("question_type") == "option":
        public_context = "\nPublic question and choices:\n" + public_option_question(sample)
    if sample.get("language") == "zh":
        return (
            "检索预算已耗尽，上一条继续调用工具的响应无效。下一条响应不得包含任何工具调用；仅根据已经返回的公开检索证据完成最终回答。"
            "关键格式检查：下一条消息必须以 <think> 开始，绝不能以 {、[ 或工具调用 JSON 开始。现在立即输出最终回答，不得提出检索请求。"
            "输出完整的 <think>...</think><answer>...</answer>，并在 <think> 中使用以下带冒号的精确字段标签：证据：、排除的候选：、不确定性：；每个字段标签必须各自位于新的一行行首。"
            "在证据：字段中必须逐字引用至少一个已经返回的公开检索类别名称，并说明它如何支持所选公开选项。"
            "对于选项题，<answer> 必须且只能包含所选公开类别对应的一个字母 A、B、C 或 D，不得包含类别名。"
            + public_context
        )
    return (
        "The retrieval budget is exhausted and the previous tool call is invalid. The next response must contain no tool call; finalize using only the public retrieval evidence already returned. "
        "CRITICAL FORMAT CHECK: your next message must begin with <think>; it must never begin with {, [, or a tool-call JSON object. Finalize now and do not request another retrieval. "
        "Output the complete <think>...</think><answer>...</answer> response with these exact colon-terminated field labels: Evidence:, Rejected alternatives:, Uncertainty:. Put each field label at the start of its own new line. "
        "In Evidence:, quote at least one exact class name already returned by the public retrieval evidence and explain how it supports the selected public option. "
        "For an Option question, <answer> must contain exactly one letter A, B, C, or D corresponding to the selected public class, with no class name."
        + public_context
    )


def user_prompt(sample: dict[str, Any], top_k: int) -> str:
    task_domain = str(sample.get("task_domain") or "").strip()
    task_context = f"This sample is from an agricultural {task_domain} recognition set. " if task_domain else ""
    strategy_id, sequence, top_k = sample_strategy(sample, top_k)
    first_type = sequence[0]
    prompt = (
        f'The user asks: "{STUDENT_USER_QUERY}" '
        f"{task_context}"
        "Before retrieving, inspect the image and form two or three descriptive visual candidates. Your first assistant reply must be exactly one JSON "
        "agrinet_rag_search tool call and nothing else. Do not give a final answer until after a tool response. "
        f"Follow retrieval strategy {strategy_id} with preferred sequence {list(sequence)}. The first retrieval type must be {first_type}. "
        "Keep the tool query short, natural, and in English, like a farmer or agronomist would ask it; use simple phrases such as crop or organ names, visible symptoms, or a concise question. "
        "Do not use Chinese in the query field. "
        "Do not put workflow instructions, retrieval strategy, long reasoning, or guessed final class names into the query field. The first query should compare visible traits, not a class label. Put a concise `Visual Observation: ...; Candidate Analysis: ...` summary in the rationale so candidate formation is visible before RAG. "
        "After the first visual tool response, explicitly form a short candidate set before searching again. The candidates must combine your Visual Observation of the query image with the retrieved results, but each self-proposed candidate should be a slightly different descriptive phrase, not an exact copied class name; use neutral English words for host group, organ, symptom, color, shape, or health state. "
        "Later queries may naturally use retrieved similar class names or aliases because those names are now evidence, but self-proposed candidates should still be descriptive variants unless you are doing exact name lookup for a name copied from retrieved evidence. "
        "Name lookup must happen only after a real visual/balanced/semantic/rrf tool response and only for an exact English class name copied from retrieved evidence. "
        "If the retrieved evidence is ambiguous, weak, or does not contain a class that can support the final prediction, search again using a short visual-trait, semantic, balanced, rrf, or exact-name verification query. "
        "Use only the query image and retrieved evidence. Do not invent wiki facts. "
        "Every final evidence statement must be anchored to retrieved class names, aliases, scores, or reference image IDs. "
        "A final response must first provide evidence-grounded analysis inside <think>...</think>, then provide only the canonical class name inside <answer>...</answer>. "
        "The <think> section must contain these exact field labels: Evidence, Rejected alternatives, Uncertainty. Predict the canonical class name only in <answer>; do not output class codes. "
        "Keep the final visible response concise: under 900 characters in total, with no more than two short bullets per field. "
        f"Default top_k is {top_k}."
    )
    if sample.get("question_type") == "option":
        prompt += (
            " The public Option question is:\n"
            + public_option_question(sample)
            + "\nFIRST-TURN STOP RULE: Do not answer the Option question yet. Your entire response to this message must be exactly one "
              "agrinet_rag_search manual JSON object and must end at its final closing brace."
        )
    return prompt


def one_shot_example(top_k: int, first_type: str = "visual") -> str:
    return (
        "First-turn-only format example (this is one complete assistant message):\n"
        f'{{"name":"agrinet_rag_search","arguments":{{"query":"what disease is on this leaf","retrieval_type":"{first_type}","image":"query_image","top_k":{top_k},"rationale":"Visual Observation: broad green leaf with a healthy-looking surface; Candidate Analysis: healthy leaf or a subtle leaf condition."}}}}\n'
        "End the assistant message immediately after the final closing brace. Do not append reasoning tags, answer tags, analysis, or any final prediction on the first turn. "
        "A later final response is allowed only after an actual tool response has been received."
    )


def initial_retrieval_system_prompt(sample: dict[str, Any], top_k: int) -> str:
    strategy_id, sequence, top_k = sample_strategy(sample, top_k)
    first_type = sequence[0]
    required = json.dumps({
        "name": TOOL_NAME,
        "arguments": {
            "query": "short English visual-trait query",
            "retrieval_type": first_type,
            "image": "query_image",
            "top_k": top_k,
            "rationale": "Visual Observation: visible traits; Candidate Analysis: descriptive candidates.",
        },
    }, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are producing only phase one of an agricultural visual-retrieval trajectory. "
        "No diagnosis or final answer is requested in this phase. "
        "Inspect the query image and output exactly one manual agrinet_rag_search JSON object. "
        f"Use strategy {strategy_id}; the first retrieval type must be {first_type} and top_k must be {top_k}. "
        "The query must be short, natural English based only on visible host, organ, symptom, color, shape, or health traits. "
        "The rationale must contain Visual Observation and Candidate Analysis. "
        "Do not output a class code, exact class-name guess, explanation, diagnosis, answer, Markdown fence, or any text outside the JSON object. "
        f"Required shape: {required} "
        "End the assistant message immediately after the final closing brace. "
        f"Tool schema: {json.dumps(tool_schema(), ensure_ascii=False)}"
    )


def initial_retrieval_user_prompt(sample: dict[str, Any], top_k: int) -> str:
    task_domain = str(sample.get("task_domain") or "agricultural recognition").strip()
    _, sequence, top_k = sample_strategy(sample, top_k)
    return (
        f"Inspect this {task_domain} image for retrieval only. "
        f"Return exactly one agrinet_rag_search JSON object using retrieval_type {sequence[0]} and top_k {top_k}. "
        "Use visible traits in the English query and rationale. Do not diagnose or answer the recognition question in this phase."
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
    label_zh = sample.get("final_label_zh") or sample.get("class_name_zh") or ""
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


def generation_route(sample: dict[str, Any]) -> str:
    route = str(sample.get("generation_route") or "oracle_grounded")
    if route not in {"oracle_grounded", "blind_evidence"}:
        raise ValueError(f"unsupported generation_route: {route}")
    return route


def teacher_private_context(sample: dict[str, Any]) -> str:
    if generation_route(sample) == "oracle_grounded":
        return private_teacher_force_context(sample)
    return (
        "Blind evidence route. Infer the answer only from the public image, public choices, and actual tool responses. "
        "No private target label is available. Do not invent class names or evidence; if evidence is insufficient, continue with a descriptive query."
    )


def build_initial_messages(sample: dict[str, Any], image_path: Path, top_k: int, image_max_side: int = 0) -> list[dict[str, Any]]:
    strategy_id, sequence, top_k = sample_strategy(sample, top_k)
    first_type = sequence[0]
    system = (
        "You are an agricultural visual recognition teacher generating agent-SFT trajectories. "
        "You must use a manual JSON tool-call protocol, not provider-native function calling. "
        "Every assistant message must be exactly one of two forms: (1) a single JSON tool-call object and no other text, "
        "or (2) the final response as <think>...</think> followed by <answer>...</answer>. Never mix a JSON tool call with explanation or a final answer. "
        "Before the first retrieval, inspect the image and form descriptive candidates. The first assistant message must be a single agrinet_rag_search JSON call whose rationale is `Visual Observation: ...; Candidate Analysis: ...`. Do not output a bare arguments object. "
        f"Use strategy {strategy_id} and prefer the retrieval sequence {list(sequence)}. The first retrieval type must be {first_type}. "
        "The required shape is exactly: "
        f'{{"name":"agrinet_rag_search","arguments":{{"query":"...","retrieval_type":"{first_type}","image":"query_image","top_k":{top_k},"rationale":"..."}}}}. '
        f'The student-facing user request is: "{STUDENT_USER_QUERY}" Use it as the starting point for natural English tool queries, but keep each query shorter than the full instruction when possible. '
        "Good first-query examples are simple English phrases or questions such as `what disease is on this leaf`, `leaf spots and edge shape`, or `brown spots on crop leaf`. After the first retrieval, first state candidate hypotheses from Visual Observation plus retrieved results, then search again. Candidate hypotheses should be descriptive variants like `healthy-looking stone-fruit leaf`, `dark leaf-spot disease on a broadleaf crop`, or `rust-like lesions on a pome-fruit leaf`; they should not exactly copy a database class name such as `Cherry Normal leaf` unless the query is an exact name lookup for retrieved evidence. If a candidate is not already named in retrieved evidence, describe it neutrally with host/organ/symptom traits instead of writing a full database class name. Avoid Chinese, first-turn class-name guesses, and procedural text such as `retrieve top visual evidence before answering`. "
        "After tool responses, either emit "
        "another single JSON tool call or give the final response. The final response must include a <think> section first, then an <answer> section. "
        "The <think> section must include exactly these field labels: Evidence, Rejected alternatives, Uncertainty. For Open questions, the <answer> section must contain only the predicted canonical class name. For Option questions, it must contain only one option letter (A, B, C, or D). Predict class names, not class codes, in the reasoning. "
        "Keep the visible final response concise: the complete <think> section must be under 900 characters, with at most two short bullets under each field label. Do not repeat the query, tool results, or long descriptions. "
        "Use hidden thinking if available to plan a high-quality trajectory, but never print chain-of-thought or private target information. "
        "Use retrieved class names, aliases, scores, and reference image IDs as evidence. Keep the final visible reasoning concise and grounded in tool responses. "
        "Final answers are allowed only when retrieved evidence contains the predicted class or an alias; otherwise search again. "
        "Name lookup may use only an exact class name or alias already seen in tool evidence. Semantic, balanced, visual, and rrf follow-ups may compare retrieved similar class names, or may search neutral host/organ/symptom descriptions when an exact name is not yet supported. Do not use name lookup on the first turn, for descriptive phrases, or for inferred/guessed names that have not appeared in retrieved evidence. "
        f"{teacher_private_context(sample)} "
        f"{one_shot_example(top_k, first_type)} "
        "Tool schema: "
        f"{json.dumps(tool_schema(), ensure_ascii=False)}"
    )
    language = str(sample.get("language") or "en")
    question_type = str(sample.get("question_type") or "open")
    presentation = (
        "Write the visible reasoning and answer in Chinese. Use the Chinese field labels `证据`, `排除的候选`, and `不确定性`; do not use English section headings. Keep only the internal retrieval query in English. "
        if language == "zh" else "Write the visible final response in English. "
    )
    if question_type == "option":
        presentation += "Identify the best matching option from the visual and retrieved evidence, explain the class-level evidence naturally in <think>, then output only its A-D letter in <answer>. "
    if sample.get("trajectory_mode") == "stop_correction":
        presentation += (
            "Begin with a plausible but deliberately broad visual hypothesis. The retrieved evidence must narrow or change it. "
            "State the revision naturally as `Correction changed:` in English or `纠正并修改：` in Chinese, and explain the evidence that justified it. "
        )
    system += " " + presentation
    # Phase separation is intentional: the first provider request receives no
    # final-answer contract or public Option choices. Those are injected only
    # after a real tool response by api_tool_response_prompt.
    return [
        {"role": "system", "content": initial_retrieval_system_prompt(sample, top_k)},
        {"role": "user", "content": [{"type": "text", "text": initial_retrieval_user_prompt(sample, top_k)}, image_url_content(image_path, image_max_side)]},
    ]


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=raw, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            if not body.strip():
                raise UnknownTeacherDelivery(
                    "teacher API returned an empty response body "
                    f"(status={getattr(resp, 'status', '?')}, content_type={resp.headers.get('Content-Type', '')})"
                )
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                prefix = body[:240].replace("\\n", " ")
                raise UnknownTeacherDelivery(
                    "teacher API returned invalid JSON: "
                    f"{exc}; status={getattr(resp, 'status', '?')}; "
                    f"content_type={resp.headers.get('Content-Type', '')}; body_prefix={prefix!r}"
                ) from exc
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"POST {url} failed with HTTP {exc.code}: {body[:1000]}"
        if exc.code == 408 or exc.code == 429 or exc.code >= 500:
            # The provider has responded, but a timeout/rate-limit/server
            # error does not prove that the POST was not accepted upstream.
            # Do not replay an ambiguous teacher request automatically.
            raise UnknownTeacherDelivery(f"teacher API delivery unknown: {message}") from exc
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        raise UnknownTeacherDelivery(f"POST {url} delivery unknown: {exc}") from exc
    except (ConnectionError, OSError, TimeoutError) as exc:
        # urllib may surface a socket timeout/reset directly rather than as
        # URLError. It is still ambiguous whether the provider processed the
        # POST, so it must take the same no-replay path.
        raise UnknownTeacherDelivery(f"POST {url} delivery unknown: {exc}") from exc


def preflight_teacher_endpoint(base_url: str, api_key: str, timeout: int) -> None:
    """Check provider reachability before creating a pilot artifact."""
    url = f"{base_url.rstrip('/')}/models"
    req = request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"teacher preflight failed with HTTP {resp.status}")
    except error.HTTPError as exc:
        raise RuntimeError(f"teacher preflight failed with HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"teacher preflight failed: {exc}") from exc


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
            "unexpected_eof_while_reading",
            "eof occurred in violation of protocol",
            "ssl eof",
            "502",
            "503",
            "504",
            "429",
            "empty response body",
            "invalid json",
            "expecting value",
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
            if isinstance(exc, error.HTTPError) and (exc.code == 408 or exc.code == 429 or exc.code >= 500):
                raise UnknownTeacherDelivery(f"teacher API delivery unknown: HTTP {exc.code}") from exc
            if isinstance(exc, (UnknownTeacherDelivery, ConnectionError, OSError, TimeoutError)):
                # A timeout, connection close, empty body, or malformed body
                # does not prove the provider did not process the POST. Never
                # replay an ambiguous teacher request automatically.
                if isinstance(exc, UnknownTeacherDelivery):
                    raise
                raise UnknownTeacherDelivery(f"teacher API delivery unknown: {exc}") from exc
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
    # The first teacher turn is protocolically a manual JSON tool call. Use
    # provider JSON mode only for that phase; later turns must remain free to
    # emit the final <think>/<answer> response after real tool evidence.
    if not any(message.get("role") == "assistant" for message in messages):
        payload["response_format"] = {"type": "json_object"}
    # Several OpenAI-compatible relay endpoints reject the optional
    # reasoning_effort field with HTTP 429/invalid_request_error.  Avoid the
    # failed request up front for those relays; callers can still opt in via
    # YUNWU_SEND_REASONING_EFFORT=1 when a provider explicitly supports it.
    relay_host = parse.urlparse(base_url).hostname or ""
    known_relay = relay_host in {"api3.wlai.vip", "api.zhongzhuan.chat", "api.apiplus.org"}
    send_reasoning = os.environ.get("YUNWU_SEND_REASONING_EFFORT", "").lower() in {"1", "true", "yes"}
    if args.reasoning_effort != "none" and (not known_relay or send_reasoning):
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


def extract_single_json_object(text: str) -> Any:
    """Parse a manual tool call only when the entire assistant content is JSON."""
    stripped = text.strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("manual tool call must be one JSON object", stripped, 0)
    return parsed


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
            parsed = extract_single_json_object(content)
        except Exception:
            return []
        if isinstance(parsed, dict) and parsed.get("name") == TOOL_NAME and isinstance(parsed.get("arguments"), dict):
            return [{"id": f"call_{uuid.uuid4().hex[:12]}", "name": parsed["name"], "arguments": parsed["arguments"]}]
        if isinstance(parsed, dict) and {"query", "retrieval_type", "image", "top_k", "rationale"}.issubset(parsed):
            return [{"id": f"call_{uuid.uuid4().hex[:12]}", "name": TOOL_NAME, "arguments": parsed}]
    return []


def has_mixed_manual_tool_content(message: dict[str, Any]) -> bool:
    """Detect a manual tool JSON object followed by any non-whitespace suffix."""
    if message.get("tool_calls"):
        return False
    content = message.get("content") or ""
    if not isinstance(content, str) or not content.strip().startswith("{"):
        return False
    stripped = content.strip()
    try:
        parsed, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    is_tool = isinstance(parsed, dict) and (
        (parsed.get("name") == TOOL_NAME and isinstance(parsed.get("arguments"), dict))
        or {"query", "retrieval_type", "image", "top_k", "rationale"}.issubset(parsed)
    )
    return is_tool and bool(stripped[end:].strip())


def bounded_preflight_message_evidence(message: dict[str, Any]) -> dict[str, Any]:
    """Summarize a provider message without retaining its text or image-bearing request."""
    content = message.get("content") or ""
    content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)
    native_calls = message.get("tool_calls") or []
    parsed_calls = normalize_tool_calls(message)
    mixed = has_mixed_manual_tool_content(message)
    if mixed:
        classification = "mixed_manual_tool_content"
    elif native_calls:
        classification = "provider_native_tool_call"
    elif len(parsed_calls) == 1 and parsed_calls[0].get("name") == TOOL_NAME:
        classification = "single_manual_tool_call"
    elif re.search(r"<answer>.*?</answer>", content, flags=re.IGNORECASE | re.DOTALL):
        classification = "final_answer_without_manual_tool_call"
    elif content.strip():
        classification = "other_nonempty_content"
    else:
        classification = "empty_content"
    call = parsed_calls[0] if len(parsed_calls) == 1 else {}
    arguments = call.get("arguments") if isinstance(call, dict) else {}
    return {
        "classification": classification,
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "starts_with_json_object": content.lstrip().startswith("{"),
        "contains_think_tag": bool(re.search(r"<think>", content, flags=re.IGNORECASE)),
        "contains_answer_tag": bool(re.search(r"<answer>", content, flags=re.IGNORECASE)),
        "native_tool_call_count": len(native_calls),
        "parsed_tool_call_count": len(parsed_calls),
        "parsed_tool_name": call.get("name") if isinstance(call, dict) else None,
        "parsed_argument_keys": sorted(arguments) if isinstance(arguments, dict) else [],
    }


def write_preflight_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clamp_tool_args(arguments: dict[str, Any], default_top_k: int, sample: dict[str, Any] | None = None, sft_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    args = dict(arguments)
    _, sequence, strategy_top_k = sample_strategy(sample or {}, default_top_k)
    default_top_k = strategy_top_k
    args.setdefault("top_k", default_top_k)
    args.setdefault("image", "none")
    args.setdefault("retrieval_type", "balanced")
    args.setdefault("rationale", "Check relevant AgriNet evidence.")
    requested_top_k = args.get("top_k", default_top_k)
    args["top_k"] = requested_top_k if isinstance(requested_top_k, int) and 1 <= requested_top_k <= 10 else default_top_k
    if sample and sample.get("strategy_id"):
        args["top_k"] = default_top_k
    query = str(args.get("query") or "").strip()
    visible_messages = sft_messages or []
    # The first strict tool call is stored as an assistant JSON message, while
    # later calls use the explicit tool_call role. Count both representations
    # so strategy clamping advances from visual to its configured follow-up
    # instead of silently repeating the first retrieval type.
    prior_calls = 0
    for message in visible_messages:
        if message.get("role") == "tool_call":
            prior_calls += 1
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                parsed = extract_single_json_object(content)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("name") == TOOL_NAME:
                prior_calls += 1
    if sample and sample.get("strategy_id") and prior_calls < len(sequence):
        required_type = sequence[prior_calls]
        if required_type == "name" and not name_query_allowed(query, sample, visible_messages):
            retrieved_names = retrieved_class_names(visible_messages, limit=1)
            if retrieved_names:
                query = retrieved_names[0]
                args["query"] = query
        if required_type != "name" or name_query_allowed(query, sample, visible_messages):
            args["retrieval_type"] = required_type
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
    if sample is not None and sample.get("language") == "zh":
        # The JSON rationale is visible to the student; keep it Chinese while
        # retaining an English-only internal arguments.query for retrieval.
        args["rationale"] = "视觉观察与候选分析：根据图像形态、颜色和症状特征检索证据并比较候选。"
        rationale = args["rationale"]
    if contains_cjk(rationale) or (sample is not None and target_name_query_leaks(rationale, class_name_aliases(sample), visible_messages)):
        if sample is not None and sample.get("language") == "zh":
            args["rationale"] = "视觉观察与候选分析：根据当前图像和检索结果核对候选。"
        else:
            args["rationale"] = neutral_rationale(args.get("retrieval_type"), str(args.get("query") or "visible crop traits"))
    if args.get("ranker") is None:
        args.pop("ranker", None)
    return args


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


ZH_VISUAL_TERMS = {
    "leaf": "叶片", "fruit": "果实", "stem": "茎秆", "flower": "花部",
    "green": "绿色", "yellow": "黄化", "brown": "褐色", "black": "黑色", "white": "白色",
    "spot": "斑点", "spots": "斑点", "lesion": "病斑", "lesions": "病斑",
    "curl": "卷曲", "curled": "卷曲", "mold": "霉层", "rot": "腐烂", "healthy": "健康外观",
    "moth": "蛾类", "beetle": "甲虫", "aphid": "蚜虫", "caterpillar": "幼虫",
    "wings": "翅部", "antennae": "触角", "long": "细长", "round": "圆形",
    "corn": "玉米", "sunflower": "向日葵", "head": "花盘", "holes": "穿孔", "hole": "穿孔",
    "moldy": "霉变", "dried": "干枯", "dark": "深色", "darkened": "变黑",
    "wilted": "萎蔫", "necrotic": "坏死", "serrated": "锯齿边缘", "florets": "小花",
}


def chinese_visual_summary(query: str) -> str:
    values: list[str] = []
    for token in re.findall(r"[a-z]+", query.lower()):
        value = ZH_VISUAL_TERMS.get(token)
        if value and value not in values:
            values.append(value)
    if len(values) < 2:
        for fallback in ("器官轮廓", "表面状态"):
            if fallback not in values:
                values.append(fallback)
            if len(values) >= 2:
                break
    return "、".join(values[:5])


def english_fallback_query(retrieval_type: Any) -> str:
    if retrieval_type == "name":
        return "retrieved crop class name"
    if retrieval_type == "semantic":
        return "leaf shape color lesions and health state"
    return "green leaf shape edge lesions and health state"


def neutral_rationale(retrieval_type: Any, query: str = "visible crop traits") -> str:
    if retrieval_type == "visual":
        return f"Compare reference images for {query}, focusing on organ shape, color, lesions, and insect morphology."
    if retrieval_type == "semantic":
        return f"Compare symptom and morphology records related to {query}."
    if retrieval_type == "name":
        return f"Confirm the retrieved class name {query} against its recorded aliases."
    return f"Compare image and text evidence for {query} to separate the current candidates."


def query_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_class_name(text)) if len(token) > 1}


def target_name_query_leaks(query: str, aliases: list[str], messages: list[dict[str, Any]]) -> bool:
    if not query or not aliases or retrieved_evidence_supports_aliases(messages, aliases):
        return False
    if query_uses_retrieved_name(query, messages):
        return False
    query_norm = normalize_class_name(query)
    query_set = query_tokens(query)
    retrieved_tokens = set().union(*(query_tokens(term) for term in retrieved_name_terms(messages))) if messages else set()
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
        # A retrieved neighboring class may establish the host and organ. It is
        # then safe to ask a neutral health-state question about those visible
        # traits without exposing the exact unseen target name.
        supported_overlap = overlap & retrieved_tokens
        unsupported_overlap = overlap - retrieved_tokens
        if supported_overlap and unsupported_overlap <= {"healthy", "normal"}:
            continue
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


def visual_candidate_think(tool_response: dict[str, Any], candidates: list[str], language: str = "en") -> str:
    names = candidate_names_from_tool_response(tool_response, limit=3)
    if language == "zh":
        lines = ["<think>视觉观察：", "- 结合图中器官、颜色、形态和受害特征比较首次检索结果。", "", "检索到的相近类别："]
        lines.extend(f"- {name}" for name in names)
        lines.extend(["", "描述性候选："]); lines.extend(f"- {candidate}" for candidate in candidates)
        return "\n".join(lines) + "</think>"
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


def sample_candidate_think(tool_response: dict[str, Any], candidates: list[str], language: str = "en") -> str:
    names = candidate_names_from_tool_response(tool_response, limit=3)
    if language == "zh":
        lines = ["<think>视觉观察：", "- 先结合查询图像与首次检索证据，再比较紧凑的候选集合。", "", "检索到的相近类别："]
        lines.extend(f"- {name}" for name in names); lines.extend(["", "候选分析："]); lines.extend(f"- {candidate}" for candidate in candidates)
        lines.extend(["", "继续检索这些候选，核对图像特征后再确定规范名称。"]); return "\n".join(lines) + "</think>"
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
    language = str(sample.get("language") or "en")
    if not (not retrieval_ledgers and sample.get("strict_first_tool_call")):
        sft_messages.append({"role": "assistant", "content": pre_think_text or pre_tool_think(call_args, language, first_turn=not retrieval_ledgers)})
    first_role = "assistant" if (not retrieval_ledgers and sample.get("strict_first_tool_call")) else "tool_call"
    sft_messages.append({"role": first_role, "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
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
    api_messages.append(api_tool_response_prompt(sample, tool_response, ledger.get("visible_reference_images") or [], args.image_max_side))
    return tool_response, ledger


def pre_tool_think(call_args: dict[str, Any], language: str = "en", first_turn: bool = False) -> str:
    retrieval_type = str(call_args.get("retrieval_type") or "balanced")
    query = str(call_args.get("query") or "evidence").strip()
    rationale = str(call_args.get("rationale") or "").strip()
    candidate_rationale = "visual observation:" in rationale.lower() and "candidate analysis:" in rationale.lower()
    if language == "zh":
        if first_turn or retrieval_type == "visual" or candidate_rationale:
            visual = chinese_visual_summary(f"{query} {rationale}")
            thought = (
                f"视觉观察：图中主要可见{visual}。\n"
                f"候选分析：这些特征可能对应健康外观，也可能对应具有相近{visual}的病害或虫害。\n"
                f"证据检索：使用{retrieval_type}检索核对{visual}及其所在器官，区分这些候选。"
            )
        else:
            thought = "证据核对：补充检索以区分当前候选，并核对宿主、症状位置和形态特征。"
    elif first_turn or retrieval_type == "visual" or candidate_rationale:
        if candidate_rationale:
            thought = rationale + f"\nEvidence search: {retrieval_type} retrieval can distinguish these candidates through host, symptom location, and morphology."
        else:
            thought = (
                f"Visual Observation: The main visible pattern is {query}.\n"
                "Candidate Analysis: Plausible explanations include a healthy host-organ appearance and a visually similar disease or pest condition.\n"
                "Evidence search: Similar reference images can distinguish them through host, symptom location, and morphology."
            )
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
        # Keep the two-decimal contract as a string: JSON numbers cannot
        # preserve a trailing zero (1.40 would otherwise become 1.4).
        "score": f"{float(hit.get('distance') or 0.0):.2f}",
        "class_name": hit.get("english_name"),
        "chinese_name": hit.get("chinese_name"),
        "aliases": [value for value in (hit.get("alias_en") or []) if value][:5],
        "chinese_aliases": [value for value in (hit.get("alias_cn") or []) if value][:5],
        "source_dataset": hit.get("source_dataset"),
        "reference_images": visible_refs,
    }


def api_tool_response_prompt(sample: dict[str, Any], tool_response: dict[str, Any], reference_image_paths: list[str], image_max_side: int = 0) -> dict[str, Any]:
    continuation = (
        "如果证据仍不足，请只输出下一次 JSON 工具调用；如果证据已经充分，请用中文输出 <think>...</think> 和 <answer>...</answer>。"
        "<think> 中必须使用中文标题：证据、排除的候选、不确定性。<answer> 只包含中文规范类名。"
        "每条证据都应引用检索到的类别名称、别名、两位小数分数或参考图像编号。"
        if sample.get("language") == "zh" else
        "If evidence is still insufficient, output only the next JSON tool call. If evidence is sufficient, answer with <think>...</think> and <answer>...</answer>. "
        "Use the headings Evidence, Rejected alternatives, and Uncertainty. Put only the canonical English class name in <answer>. "
        "Anchor each evidence statement to retrieved class names, aliases, two-decimal scores, or reference image IDs."
    )
    if sample.get("question_type") == "option":
        public_question = public_option_question(sample)
        continuation = (
            "如果证据仍不足，请只输出下一次 JSON 工具调用；如果证据已经充分，请输出完整的 <think>...</think><answer>...</answer>。"
            "<think> 使用证据、排除的候选、不确定性三个标题，并根据检索证据比较公开选项。"
            "<answer> 必须且只能包含与所选公开类别对应的一个字母 A、B、C 或 D，不得包含类别名。\n"
            f"公开题面与选项：\n{public_question}"
            if sample.get("language") == "zh" else
            "If evidence is still insufficient, output only the next JSON tool call. If evidence is sufficient, output the complete <think>...</think><answer>...</answer> response. "
            "Use the headings Evidence, Rejected alternatives, and Uncertainty, and compare the public choices against retrieved evidence. "
            "The <answer> must contain exactly one letter A, B, C, or D corresponding to the selected public class, with no class name.\n"
            f"Public question and choices:\n{public_question}"
        )
    if sample.get("trajectory_mode") == "stop_correction":
        continuation += (
            " 在 Evidence 区域必须单独写一行‘纠正并修改：<初始假设> → <修正后的假设>’，明确说明本次检索证据如何把最初的宽泛视觉假设修改为更具体的类别；不能省略该行。"
            if sample.get("language") == "zh" else
            " In the Evidence section, you must include a separate line exactly in the form Correction changed: <initial broad hypothesis> -> <revised hypothesis>, stating how this retrieval revised the broad visual hypothesis; omitting this line is invalid."
        )
        desired = str(sample.get("desired_correction_type") or "")
        if desired == "evidence_confirmed":
            continuation += (" 本条目标是 evidence_confirmed：允许初始候选与最终类别保持同一方向，但必须明确写出检索证据确认了原先候选；不得伪造类别改变。"
                             if sample.get("language") == "zh" else
                             " This target is evidence_confirmed: the initial and final class direction may stay the same. Add a separate exact line `Correction type: evidence_confirmed` and explicitly state that retrieval confirmed the initial candidate; do not invent a class change.")
            continuation += (" 最终回答前必须再次检查并输出该行；如果证据不支持原候选，继续输出 JSON 检索调用，不要提交未经证据支持的答案。"
                             if sample.get("language") == "zh" else
                             " Before the final answer, re-check and output that exact correction-type line. If evidence does not support the initial candidate, emit another JSON retrieval call instead of submitting an unsupported answer.")
        elif desired == "narrowed":
            continuation += (" 本条目标是 narrowed：必须展示从较宽的候选范围收窄到检索支持的具体类别，并说明排除依据。"
                             if sample.get("language") == "zh" else
                             " This target is narrowed: add a separate exact line `Correction type: narrowed`, show a genuine narrowing from a broad candidate set to the retrieval-supported specific class, and state the exclusion basis.")
    text = (
        "Tool response JSON:\n"
        f"{json.dumps(tool_response, ensure_ascii=False)}\n\n"
        f"{teacher_private_context(sample)}\n\n"
        "Reference image IDs in the JSON correspond to the images attached after this text, in the same order. "
        f"{continuation} "
        "Do not output class codes or any statement that you used private/ground-truth/internal target information."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for ref in reference_image_paths:
        path = Path(ref)
        if path.exists():
            content.append(image_url_content(path, image_max_side))
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
    language = str(sample.get("language") or "en")
    if sample.get("question_type") != "option":
        query = "请识别图中的农业病虫害，并输出其规范名称。" if language == "zh" else STUDENT_USER_QUERY
        return {"role": "user", "content": "<image>\n" + query}
    return {"role": "user", "content": "<image>\n" + public_option_question(sample)}


def parse_final_answer_fields(text: str) -> dict[str, str]:
    answer_body = extract_answer_body(text)
    think_body = extract_think_body(text)
    text = answer_body if any(field in answer_body for field in FINAL_REQUIRED_FIELDS) else think_body
    fields: dict[str, str] = {}
    current: str | None = None
    aliases = {field.lower(): field for field in FINAL_REQUIRED_FIELDS}
    aliases.update({key.lower(): value for key, value in FINAL_FIELD_ALIASES.items()})
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current and fields[current]:
                fields[current] += "\n"
            continue
        lower = stripped.lower()
        matched = None
        for alias, canonical in aliases.items():
            if lower.startswith(alias.lower() + ":") or lower.startswith(alias.lower() + "："):
                matched = canonical
                value = re.split(r"[:：]", stripped, maxsplit=1)[1].strip()
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
    aliases.extend([
        sample.get("final_label_name"), sample.get("final_label_zh"),
        sample.get("class_name"), sample.get("class_name_zh"),
    ])
    return [alias for alias in aliases if isinstance(alias, str) and alias.strip()]


def canonical_label_name(sample: dict[str, Any]) -> str:
    label_code = sample.get("final_label")
    for item in sample.get("candidate_labels", []):
        if item.get("code") == label_code and isinstance(item.get("name"), str) and item.get("name", "").strip():
            return str(item["name"]).strip()
    for key in ("final_label_name", "label_name", "class_name"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(label_code or "").strip()


def sanitize_final_message(final_text: str, sample: dict[str, Any]) -> str:
    final_text = final_text.replace("**", "")
    canonical = canonical_label_name(sample)
    if sample.get("language") == "zh" and (sample.get("final_label_zh") or sample.get("class_name_zh")):
        canonical = str(sample.get("final_label_zh") or sample.get("class_name_zh"))
    if sample.get("question_type") == "option":
        canonical = str(sample["correct_option"])
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


def label_influence_audit(messages: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any]:
    supported_turn = None; used_turn = None; query_source = None; tool_turn = 0; supported = False
    for message in messages:
        if message.get("role") == "tool_response":
            tool_turn += 1
            parsed = parse_sft_json_content(message); results = parsed.get("results", []) if isinstance(parsed, dict) else []
            if any(any(class_name_matches(value, aliases) for value in result_aliases(result)) for result in results if isinstance(result, dict)):
                supported = True; supported_turn = supported_turn or tool_turn
        elif message.get("role") == "tool_call":
            parsed = parse_sft_json_content(message); args = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
            query = str(args.get("query") or "")
            if any(class_name_matches(query, [alias]) for alias in aliases):
                used_turn = used_turn or (tool_turn + 1); query_source = "retrieved_evidence" if supported else "unsupported"
    return {
        "target_first_supported_turn": supported_turn, "target_name_first_used_turn": used_turn,
        "target_name_query_source": query_source, "premature_target_query": premature_target_name_query(messages, aliases),
        "unsupported_target_rationale": False, "final_evidence_supported": retrieved_evidence_supports_aliases(messages, aliases),
    }


def correction_audit(sample: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if sample.get("trajectory_mode") != "stop_correction": return None
    first = next((extract_think_body(str(m.get("content", ""))) for m in messages if m.get("role") == "assistant"), "")
    final = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"), "")
    marker = "纠正并修改：" if sample.get("language") == "zh" else "Correction changed:"
    revised = final.split(marker, 1)[1].splitlines()[0].strip() if marker in final else ""
    visible = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "assistant")
    desired = str(sample.get("desired_correction_type") or "")
    if desired == "evidence_confirmed" and (re.search(r"Correction type\s*:\s*evidence_confirmed", visible, re.I) or "纠正类型：evidence_confirmed" in visible):
        change_type = "evidence_confirmed"
    elif desired == "narrowed" and (re.search(r"Correction type\s*:\s*narrowed", visible, re.I) or "纠正类型：narrowed" in visible):
        change_type = "narrowed"
    else:
        change_type = "class_changed" if revised and normalize_class_name(revised) not in normalize_class_name(first) else "narrowed"
    return {"initial_hypothesis": first[:500], "revised_hypothesis": revised[:500], "change_type": change_type,
            "change_reason": "retrieved evidence narrowed the visual candidate set", "supporting_retrieval_turn": 1 if revised else None}


def needs_more_evidence_before_final(sample: dict[str, Any], sft_messages: list[dict[str, Any]], final_text: str) -> bool:
    if generation_route(sample) == "blind_evidence":
        predicted = predicted_class_name(final_text)
        return predicted == "unknown" or not retrieved_evidence_supports_aliases(sft_messages, [predicted]) or not final_answer_has_evidence_anchor(final_text, sft_messages)
    aliases = class_name_aliases(sample)
    return not retrieved_evidence_supports_aliases(sft_messages, aliases) or not final_answer_has_evidence_anchor(final_text, sft_messages)


def evidence_retry_prompt(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"{teacher_private_context(sample)}\n\n"
            "The current visible trajectory is not sufficiently anchored to retrieved evidence. "
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
    if any(not row.get("ok") for row in retrieval_ledgers):
        reasons.append("failed_tool_call")
    _, preferred_sequence, _ = sample_strategy(sample, 3)
    actual_sequence = tuple(
        str(row.get("tool_call", {}).get("arguments", {}).get("retrieval_type") or "")
        for row in retrieval_ledgers
    )
    compared = min(len(actual_sequence), len(preferred_sequence))
    if sample.get("strategy_id") and actual_sequence[:compared] != preferred_sequence[:compared]:
        reasons.append("strategy_sequence_mismatch")
    if "name" in actual_sequence and actual_sequence[-1] != "name":
        reasons.append("name_confirmation_not_final")
    for message in sft_messages:
        if message.get("role") != "tool_call":
            continue
        content = str(message.get("content") or "")
        if "Predicted class name" in content or "Evidence" in content:
            reasons.append("tool_call_mixes_final_answer")
            break
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
    if sample.get("question_type") == "option":
        answer = extract_answer_body(assistant_final).strip()
        expected_option = str(sample.get("correct_option") or "")
        if answer not in {"A", "B", "C", "D"}:
            reasons.append("option_answer_not_single_letter")
        elif answer != expected_option:
            reasons.append("option_answer_mismatch")
        # Use the public option text for label validation, but never rewrite a
        # Blind answer from the private target after generation.
        choices = {
            letter: value
            for letter, value in re.findall(r"^([A-D])\. (.+)$", str(sft_messages[0].get("content", "")), re.MULTILINE)
        }
        predicted = choices.get(answer, answer)
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
    pre_tool_text = "\n".join(str(msg.get("content", "")) for msg in sft_messages if msg.get("role") == "assistant" and msg is not sft_messages[-1])
    if sample.get("language") == "en" and contains_cjk(pre_tool_text + "\n" + assistant_final):
        reasons.append("english_visible_text_contains_chinese")
    if sample.get("language") == "zh" and any(marker in pre_tool_text for marker in (
        "Visual Observation:", "Candidate Analysis:", "RAG Plan:", "Evidence search:"
    )):
        reasons.append("chinese_visible_reasoning_contains_english_template")
    if sample.get("language") == "zh" and (
        not all(re.search(rf"{marker}[:：]", assistant_final) for marker in ("证据", "排除的候选", "不确定性"))
        or any(marker in assistant_final for marker in ("Evidence:", "Rejected alternatives:", "Uncertainty:"))
    ):
        reasons.append("chinese_final_language_mismatch")
    first_think = next((str(msg.get("content", "")) for msg in sft_messages if msg.get("role") == "assistant"), "")
    if not any(marker in first_think for marker in ("Candidate Analysis:", "候选分析：")):
        reasons.append("missing_pre_rag_candidates")
    if sample.get("language") == "zh" and "视觉观察：" in first_think:
        observation = first_think.split("视觉观察：", 1)[1].split("候选分析：", 1)[0]
        if "、" not in observation:
            reasons.append("fewer_than_two_visible_traits")
    generic_markers = (
        "I note the visible organ, color, shape, lesion pattern, and health state.",
    )
    if any(marker in first_think for marker in generic_markers):
        reasons.append("generic_pre_rag_candidate_analysis")
    first_call = next((parse_sft_json_content(msg) for msg in sft_messages if msg.get("role") == "tool_call"), None)
    first_query = str((first_call or {}).get("arguments", {}).get("query", "")).lower().strip()
    if first_query in {"agricultural object or disease in the image", "crop leaf disease name", "crop leaf symptoms and disease"}:
        reasons.append("generic_first_retrieval_query")
    if sample.get("trajectory_mode") == "stop_correction" and not (
        "Correction changed:" in assistant_final or "纠正并修改：" in assistant_final
    ):
        reasons.append("missing_stop_correction_behavior")
    desired_correction = str(sample.get("desired_correction_type") or "")
    if desired_correction:
        audit = correction_audit(sample, sft_messages) or {}
        actual_correction = str(audit.get("change_type") or "")
        if actual_correction != desired_correction:
            reasons.append(f"correction_type_mismatch:{desired_correction}!={actual_correction or 'missing'}")
        if desired_correction == "narrowed":
            lowered = visible_text.lower()
            narrowing_terms = ("narrow", "specific", "exclude", "收窄", "具体", "排除")
            if not any(term in lowered for term in narrowing_terms):
                reasons.append("narrowed_missing_exclusion_basis")
    return not reasons, reasons


def run_sample(sample: dict[str, Any], args: argparse.Namespace, api_key: str, base_url: str) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    image_path = Path(sample["query_image"])
    _, _, sample_top_k = sample_strategy(sample, args.top_k)
    api_messages = build_initial_messages(sample, image_path, sample_top_k, args.image_max_side)
    sft_messages: list[dict[str, str]] = [sft_user_message(sample, sample_top_k)]
    raw_responses = []
    retrieval_ledgers = []
    correction_prompt_attempts = 0
    option_contract_retry_attempts = 0
    budget_finalization_attempts = 0
    final_contract_retry_attempts = 0

    for _ in range(args.max_tool_turns + 3):
        try:
            response = chat_completion(api_key, base_url, args.model, api_messages, args)
        except UnknownTeacherDelivery as exc:
            trace = {
                "sample_id": sample.get("sample_id"),
                "accepted": False,
                "rejection_reasons": ["unknown_delivery"],
                "delivery_status": "unknown",
                "error": str(exc),
                "sample": sample,
                "api_messages": api_messages,
                "raw_responses": raw_responses,
                "sft_messages": sft_messages,
                "retrieval_calls": retrieval_ledgers,
            }
            rejected = {"sample_id": sample.get("sample_id"), "reasons": ["unknown_delivery"], "delivery_status": "unknown", "error": str(exc), "trace": trace}
            return None, trace, retrieval_ledgers, rejected
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
        if has_mixed_manual_tool_content(message):
            reasons = ["mixed_manual_tool_content"]
            trace = {
                "sample_id": sample.get("sample_id"),
                "accepted": False,
                "rejection_reasons": reasons,
                "sample": sample,
                "api_messages": api_messages,
                "raw_responses": raw_responses,
                "sft_messages": sft_messages,
                "retrieval_calls": retrieval_ledgers,
            }
            rejected = {"sample_id": sample.get("sample_id"), "reasons": reasons, "trace": trace}
            return None, trace, retrieval_ledgers, rejected
        calls = normalize_tool_calls(message)
        if calls and len(retrieval_ledgers) < args.max_tool_turns:
            normalized_calls = []
            for call in calls[:1]:
                call["arguments"] = clamp_tool_args(call["arguments"], sample_top_k, sample, sft_messages)
                if should_force_adjacent_search(sample, sft_messages, retrieval_ledgers, args.max_tool_turns):
                    adjacent_args = adjacent_evidence_followup_args(sample, sft_messages, sample_top_k)
                    if adjacent_args is not None:
                        call["arguments"] = clamp_tool_args(adjacent_args, sample_top_k, sample, sft_messages)
                normalized_calls.append(call)
            for call in normalized_calls:
                if call.get("name") != TOOL_NAME:
                    call_args = call["arguments"]
                    visible_call = {"name": TOOL_NAME, "arguments": call_args}
                    if not (not retrieval_ledgers and sample.get("strict_first_tool_call")):
                        sft_messages.append({"role": "assistant", "content": pre_tool_think(call_args, str(sample.get("language") or "en"), first_turn=not retrieval_ledgers)})
                    first_role = "assistant" if (not retrieval_ledgers and sample.get("strict_first_tool_call")) else "tool_call"
                    sft_messages.append({"role": first_role, "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
                    tool_response = {"status": "error", "errors": [f"unknown tool {call.get('name')}"]}
                    ledger = {"ok": False, "validation_errors": tool_response["errors"], "request": call}
                    ledger.update({"sample_id": sample.get("sample_id"), "tool_call": visible_call})
                    retrieval_ledgers.append(ledger)
                    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False)})
                    api_messages.append(api_tool_response_prompt(sample, tool_response, [], args.image_max_side))
                else:
                    before_count = len(retrieval_ledgers)
                    tool_response, ledger = append_tool_execution(args, sample, sft_messages, retrieval_ledgers, api_messages, call["arguments"], str(call.get("id") or "call_manual"))
                    if (
                        ledger.get("ok")
                        and call["arguments"].get("retrieval_type") == "visual"
                        and before_count == 0
                        and len(retrieval_ledgers) < args.max_tool_turns
                        and sample_strategy(sample, args.top_k)[1][:2] == ("visual", "balanced")
                    ):
                        if args.candidate_followup_mode == "sample_candidates":
                            followup_args = sample_candidate_followup_args(sample, sample_top_k)
                            candidate_think = sample_candidate_think(tool_response, sample_candidate_names(sample, limit=6), str(sample.get("language") or "en")) if followup_args is not None else None
                        else:
                            followup_args = similar_candidate_followup_args(tool_response, sample_top_k)
                            candidate_think = visual_candidate_think(tool_response, descriptive_candidates_from_tool_response(tool_response, limit=3), str(sample.get("language") or "en")) if followup_args is not None else None
                        if followup_args is not None:
                            followup_args = clamp_tool_args(followup_args, sample_top_k, sample, sft_messages)
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
                            verify_args = neutral_candidate_followup_args(sample, sample_top_k)
                            if verify_args is not None:
                                verify_args = clamp_tool_args(verify_args, sample_top_k, sample, sft_messages)
                                if verify_args.get("retrieval_type") == "name":
                                    append_tool_execution(args, sample, sft_messages, retrieval_ledgers, api_messages, verify_args, f"auto_{uuid.uuid4().hex[:12]}")
            continue
        if calls and len(retrieval_ledgers) >= args.max_tool_turns:
            if budget_finalization_attempts < 1:
                budget_finalization_attempts += 1
                api_messages.append({"role": "user", "content": retrieval_budget_finalization_prompt(sample)})
                continue
            reasons = ["tool_call_after_budget_finalization"]
            trace = {
                "sample_id": sample.get("sample_id"), "accepted": False,
                "rejection_reasons": reasons, "sample": sample,
                "api_messages": api_messages, "raw_responses": raw_responses,
                "sft_messages": sft_messages, "retrieval_calls": retrieval_ledgers,
            }
            rejected = {"sample_id": sample.get("sample_id"), "reasons": reasons, "trace": trace}
            return None, trace, retrieval_ledgers, rejected
        content = message.get("content") or ""
        final_text = str(content).strip()
        if (sample.get("question_type") == "option"
                and option_contract_retry_attempts < 1
                and extract_answer_body(final_text).strip() not in {"A", "B", "C", "D"}):
            option_contract_retry_attempts += 1
            api_messages.append({"role": "user", "content": option_contract_retry_prompt(sample, final_text)})
            continue
        if (sample.get("trajectory_mode") == "stop_correction"
                and correction_prompt_attempts < 1
                and "Correction changed:" not in final_text
                and "纠正并修改：" not in final_text):
            correction_prompt_attempts += 1
            desired = str(sample.get("desired_correction_type") or "")
            marker = (
                ("纠正类型：evidence_confirmed" if desired == "evidence_confirmed" else "纠正类型：narrowed")
                if sample.get("language") == "zh"
                else ("Correction type: evidence_confirmed" if desired == "evidence_confirmed" else "Correction type: narrowed")
            )
            api_messages.append({"role": "user", "content": (
                f"Correction protocol retry: before answering, include one exact line `Correction changed: <initial broad hypothesis> -> <revised hypothesis>` and one exact line `{marker}` in <think>. For evidence_confirmed, explicitly say retrieval confirmed the initial candidate; do not invent a class change. The final <answer> must copy a class name from retrieved evidence."
                if sample.get("language") != "zh" else
                f"纠正协议重试：回答前必须在 <think> 中写出‘纠正并修改：<初始宽泛假设> → <修正后的假设>’以及‘{marker}’对应的中文标记；evidence_confirmed 不得伪造类别改变。<answer> 必须逐字使用检索证据中的类别名称。"
            )})
            continue
        if (sample.get("trajectory_mode") == "stop_correction"
                and final_contract_retry_attempts < 2
                and (not all(field in final_text for field in FINAL_REQUIRED_FIELDS)
                     or not final_answer_has_evidence_anchor(final_text, sft_messages))):
            final_contract_retry_attempts += 1
            api_messages.append({"role": "user", "content": (
                "Final contract retry: the previous final text was incomplete. Output the complete final response only, with each required field on its own line and the colon included exactly: Evidence:, Rejected alternatives:, Uncertainty:. Retain the required Correction changed: and Correction type: lines inside <think>; then output <answer> containing only a class name copied verbatim from retrieved evidence. Do not add prose or tool calls outside these tags."
                if sample.get("language") != "zh" else
                "最终契约重试：上一条最终文本不完整。现在只输出完整最终回答；<think> 中必须将以下字段分别置于新行行首并保留冒号：证据：、排除的候选：、不确定性：，同时保留精确的纠正并修改：和纠正类型：行；随后在 <answer> 中逐字输出检索证据中的类别名称。不要输出工具调用或标签外文字。"
            )})
            continue
        if len(retrieval_ledgers) < args.max_tool_turns and needs_more_evidence_before_final(sample, sft_messages, final_text):
            adjacent_args = adjacent_evidence_followup_args(sample, sft_messages, sample_top_k)
            if adjacent_args is not None:
                adjacent_args = clamp_tool_args(adjacent_args, sample_top_k, sample, sft_messages)
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

    if generation_route(sample) == "oracle_grounded" and sft_messages and sft_messages[-1].get("role") == "assistant":
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
            "strategy_id": sample.get("strategy_id", "legacy_visual"),
            "preferred_sequence": list(sample_strategy(sample, args.top_k)[1]),
            "retrieval_top_k": sample_top_k,
            "generation_route": generation_route(sample),
            "label_visible_to_teacher": generation_route(sample) == "oracle_grounded",
            "label_influence_audit": label_influence_audit(sft_messages, class_name_aliases(sample)),
            "correction_audit": correction_audit(sample, sft_messages),
            "desired_correction_type": sample.get("desired_correction_type"),
            "target_id": sample.get("target_id"),
            "candidate_index": sample.get("candidate_index"),
            "language": sample.get("language", "en"),
            "question_type": sample.get("question_type", "open"),
            "trajectory_mode": sample.get("trajectory_mode", "standard"),
            "correct_option": sample.get("correct_option"),
            "reserve": bool(sample.get("reserve", False)),
        },
    }
    return sft_row, trace, retrieval_ledgers, None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    provider_name: str,
    accepted: int,
    rejected: int,
    unknown_delivery: int = 0,
) -> None:
    manifest = {
        "artifact_version": "agrinet_rag_toolcall_v4_teacher_forced_pilot5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": args.plan_file or args.sample_file,
        "git_commit": git_commit(),
        "teacher_model": args.model,
        "provider": provider_name,
        "milvus_collection": "agrinet_wiki_siglip2",
        "embedding_model": "models/siglip2-so400m-patch16-naflex",
        "rag_api": args.rag_api,
        "prompt_version": PROMPT_VERSION,
        "acceptance_policy": ACCEPTANCE_POLICY,
        "sampling": {"offset": args.offset, "limit": args.limit, "temperature": args.temperature, "max_tool_turns": args.max_tool_turns, "top_k": args.top_k},
        "counts": {"accepted": accepted, "rejected": rejected, "unknown_delivery": unknown_delivery},
        "delivery_status": "unknown" if unknown_delivery else "complete",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_run_status(output_dir: Path, status: str, **fields: Any) -> None:
    """Persist a small lifecycle marker, including interrupted runs."""
    payload = {
        "schema_version": "agrinet.rag-pilot-run-status/v1",
        "status": status,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    destination = output_dir / "run_status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def install_interrupt_status_handler(output_dir: Path) -> None:
    """Fail closed when a bounded Pilot is externally interrupted."""
    def handle(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        preserved: dict[str, Any] = {}
        status_path = output_dir / "run_status.json"
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
            for key in (
                "model", "limit", "completed_samples",
                "accepted_observed", "rejected_observed",
                "progress_checkpoint", "training_eligible",
            ):
                if key in current:
                    preserved[key] = current[key]
        except (OSError, json.JSONDecodeError):
            pass
        write_run_status(
            output_dir,
            "terminated_unknown_delivery",
            delivery_status="unknown",
            signal=signal_name,
            exit_code=128 + signum,
            decision="do_not_retry_without_provider_request_resolution",
            **preserved,
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)
    write_run_status(
        output_dir,
        "running",
        delivery_status="in_progress",
        model=args.model,
        limit=args.limit,
        decision="do_not_retry_until_completed_or_provider_status_is_known",
    )
    install_interrupt_status_handler(output_dir)
    (output_dir / "tools" / "agrinet_rag_search.schema.json").write_text(json.dumps(tool_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    api_key, base_url, provider_name = resolve_api_config()
    if args.preflight_only:
        report: dict[str, Any] = {
            "schema": "agrinet.preflight_evidence/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider_name,
            "base_url": base_url,
            "model": args.model,
            "status": "started",
            "reachable": False,
            "visual_reachable": False,
            "manual_tool_call": False,
        }
        try:
            preflight_teacher_endpoint(base_url, api_key, max(1, int(args.teacher_timeout)))
            report["reachable"] = True
            result = {"provider": provider_name, "base_url": base_url, "reachable": True}
            if args.preflight_image:
                samples = read_plan_samples(args)
                if not samples:
                    report["failure_classification"] = "missing_sample"
                    raise RuntimeError("visual preflight requires at least one sample")
                sample = samples[0]
                image_path = Path(sample["query_image"])
                report.update({
                    "sample_id": sample.get("sample_id"),
                    "query_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                })
                messages = build_initial_messages(sample, image_path, int(args.top_k), args.image_max_side)
                response = chat_completion(api_key, base_url, args.model, messages, args)
                report["visual_reachable"] = True
                message = extract_message(response)
                evidence = bounded_preflight_message_evidence(message)
                report["response_evidence"] = evidence
                if evidence["classification"] == "mixed_manual_tool_content":
                    report["failure_classification"] = "mixed_manual_tool_content"
                    raise RuntimeError("visual preflight returned mixed manual tool JSON and additional assistant content")
                calls = normalize_tool_calls(message)
                if len(calls) != 1 or calls[0].get("name") != TOOL_NAME:
                    report["failure_classification"] = str(evidence["classification"])
                    raise RuntimeError("visual preflight returned no single manual agrinet_rag_search tool call")
                validate_tool_arguments(calls[0].get("arguments") or {})
                report["manual_tool_call"] = True
                result.update({"visual_reachable": True, "manual_tool_call": True, "model": args.model})
            report["status"] = "passed"
            write_preflight_report(args.preflight_report, report)
            write_run_status(output_dir, "preflight_passed", delivery_status="complete", preflight_report=args.preflight_report)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        except Exception as exc:
            report["status"] = "failed"
            report.setdefault("failure_classification", "transport_or_runtime_error")
            report["exception_type"] = type(exc).__name__
            write_preflight_report(args.preflight_report, report)
            write_run_status(
                output_dir,
                "preflight_failed",
                delivery_status="unknown" if isinstance(exc, UnknownTeacherDelivery) else "complete",
                failure_classification=report.get("failure_classification"),
                exception_type=type(exc).__name__,
            )
            raise
    samples = read_plan_samples(args)

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
            write_progress_checkpoint(output_dir, results)
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
                write_progress_checkpoint(output_dir, results)
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
    for route in ("oracle_grounded", "blind_evidence"):
        write_jsonl(output_dir / "train" / f"agent_sft.{route}.jsonl", [row for row in accepted_rows if row.get("metadata", {}).get("generation_route") == route])
    write_jsonl(output_dir / "traces" / "raw_trajectories.jsonl", raw_traces)
    write_jsonl(output_dir / "traces" / "retrieval_calls.jsonl", retrieval_rows)
    write_jsonl(output_dir / "traces" / "rejected_trajectories.jsonl", rejected_rows)
    for route in ("oracle_grounded", "blind_evidence"):
        write_jsonl(output_dir / "traces" / f"rejected_trajectories.{route}.jsonl", [row for row in rejected_rows if row.get("trace", {}).get("sample", {}).get("generation_route") == route])
    route_summary = {route: {"accepted": sum(row.get("metadata", {}).get("generation_route") == route for row in accepted_rows),
                             "rejected": sum(row.get("trace", {}).get("sample", {}).get("generation_route") == route for row in rejected_rows)}
                     for route in ("oracle_grounded", "blind_evidence")}
    (output_dir / "reports" / "route_summary.json").write_text(json.dumps(route_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unknown_delivery = sum(1 for row in rejected_rows if "unknown_delivery" in (row.get("reasons") or []))
    write_manifest(output_dir, args, provider_name, len(accepted_rows), len(rejected_rows), unknown_delivery)
    write_run_status(
        output_dir,
        "completed_with_unknown_delivery" if unknown_delivery else "completed",
        delivery_status="unknown" if unknown_delivery else "complete",
        accepted=len(accepted_rows),
        rejected=len(rejected_rows),
        unknown_delivery=unknown_delivery,
    )
    print(f"Wrote {len(accepted_rows)} accepted and {len(rejected_rows)} rejected trajectories to {output_dir}", flush=True)
    if unknown_delivery:
        return 2
    runtime_failures = sum(
        1 for row in rejected_rows if "runtime_error" in (row.get("reasons") or [])
    )
    return 1 if runtime_failures and runtime_failures == len(samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
