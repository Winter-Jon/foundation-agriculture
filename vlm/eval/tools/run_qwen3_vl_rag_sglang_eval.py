#!/usr/bin/env python3
"""Run AgriNet Qwen3-VL evaluation with SFT-style RAG tool-calling over an sglang OpenAI endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rag_distill.run_pilot import (
    STUDENT_USER_QUERY,
    clamp_tool_args,
    compact_hit,
    extract_message,
    image_url_content,
    sft_user_message,
)
from tools.rag_distill.schema import TOOL_NAME, tool_schema, validate_tool_arguments
from tools.rag_distill.terminal_contract import final_answer_only_correction
from agrinet.rag.hermes_protocol import is_pre_tool_think, parse_hermes_tool_calls
from eval_runner_common import SnapshotStore, load_jsonl as durable_load_jsonl, request_fingerprint, validate_manifest


DEFAULT_MAX_CONCURRENT = 24
# v4 adds public curated similar-class evidence to each model-visible hit.
# Bump the fingerprint so a resumed run cannot mix old and new evidence turns.
PROTOCOL_VERSION = "agrinet.hermes-rag-sglang-async/v5-native-json-multi-query-recovery-strict-similar-classes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Evaluation manifest JSONL.")
    parser.add_argument("--output", required=True, help="Prediction JSONL.")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving relative image paths.")
    parser.add_argument("--model", required=True, help="Served model name exposed by sglang.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible sglang API base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI-compatible API key for the local sglang endpoint.")
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077", help="Local AgriNet RAG API base URL.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--offset", type=int, default=0, help="Optional starting row offset (applied before limit).")
    parser.add_argument("--top-k", type=int, default=3, help="Default RAG top-k.")
    parser.add_argument("--max-tool-turns", type=int, default=3, help="Maximum model-driven tool turns.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Concurrent samples; tool turns within one sample remain ordered.")
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--snapshot-every", type=int, default=1, help="Progress reporting cadence; every completion is durable.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-protocol-trace", action="store_true", help="Persist bounded model-output forms for non-formal protocol smoke diagnostics.")
    parser.add_argument("--disable-forced-first-call", action="store_true", help="Do not inject the mandatory first retrieval when the model answers directly.")
    parser.add_argument(
        "--invalid-tool-call-policy", choices=("strict", "recovery"), default="strict",
        help="strict closes any malformed tool attempt with one terminal-answer request; recovery may use one fixed public visual fallback before the tool budget is exhausted.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def _resolve(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _eval_sample(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    label_aliases = [str(x).strip() for x in (row.get("label_aliases") or []) if str(x).strip()]
    label_name_zh = next((x for x in label_aliases if any('\u4e00' <= ch <= '\u9fff' for ch in x)), "")
    # Evaluation manifests store public Option choices as option_* fields,
    # whereas distillation plans use candidate_* fields.  Preserve the public
    # manifest order exactly; never derive or rearrange choices from a hidden
    # label at evaluation time.
    candidate_names = [str(x).strip() for x in (row.get("candidate_names") or row.get("option_names") or []) if str(x).strip()]
    candidate_codes = [str(x).strip() for x in (row.get("candidate_codes") or row.get("option_codes") or []) if str(x).strip()]
    candidate_labels = [
        {"code": code, "name": name}
        for code, name in zip(candidate_codes, candidate_names)
    ]
    return {
        "sample_id": row.get("source_sample_id") or row.get("id"),
        "query_image": str(image_path),
        "task_domain": row.get("task_domain"),
        "final_label": row.get("label_code"),
        "final_label_code": row.get("label_code"),
        "final_label_name": row.get("label_name"),
        "final_label_zh": label_name_zh,
        "candidate_labels": candidate_labels,
        "language": row.get("language") or "en",
        "question_type": row.get("question_type") or "open",
        "correct_option": row.get("option_answer") or None,
        "evaluation_question": row.get("question") or STUDENT_USER_QUERY,
    }


def _build_eval_messages(sample: dict[str, Any], image_path: Path, top_k: int) -> list[dict[str, Any]]:
    question = str(sample.get("evaluation_question") or STUDENT_USER_QUERY)
    # The frozen Swift native ``tool_call`` role is rendered as a bare JSON
    # assistant object, not Hermes XML.  Keep XML parsing only as historical
    # input recovery; asking for XML here creates an evaluation-only wire
    # mismatch and confounds strict protocol diagnostics.
    system = (
        "You are a helpful agricultural recognition assistant. When retrieval is needed, output exactly one bare JSON tool-call object and no other text: "
        f'{{"name":"{TOOL_NAME}","arguments":{{"query":"...","retrieval_type":"visual","image":"query_image","top_k":{top_k},"rationale":"..."}}}}. '
        "After a tool result, either output one bare JSON tool-call object or the trained final form <think>brief evidence</think><answer>...</answer>. Never use <tool_call> XML tags, markdown fences, or mix a call with an answer."
    )
    return [
        {"role": "system", "content": system + " Tool schema: " + json.dumps(tool_schema(), ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": [{"type": "text", "text": question}, image_url_content(image_path)]},
    ]



def _training_tool_response(tool_response: dict[str, Any]) -> dict[str, Any]:
    """Serialize public retrieval evidence in the frozen SFT schema.

    Student RAG rows use a native ``tool`` role whose results expose
    ``name``, ``name_zh``, and ``similarity``.  Evaluation must not add labels,
    option targets, or teacher-only fields.
    """
    results: list[dict[str, Any]] = []
    for hit in tool_response.get("results", []) or []:
        if isinstance(hit, dict):
            results.append({
                "rank": hit.get("rank"),
                "name": hit.get("class_name"),
                "name_zh": hit.get("chinese_name"),
                "similarity": hit.get("score"),
            })
            similar = hit.get("similar_classes")
            if isinstance(similar, list) and similar:
                # Keep only readable public names in the actual model-visible
                # tool turn. Internal IDs, local paths, and labels stay hidden.
                results[-1]["similar_classes"] = [
                    {"name": item.get("name"), "name_zh": item.get("name_zh")}
                    for item in similar[:5]
                    if isinstance(item, dict) and (item.get("name") or item.get("name_zh"))
                ]
    payload = {
        "source": "AgriNet public reference catalog",
        "retrieval_type": tool_response.get("retrieval_type"),
        "query": tool_response.get("query"),
        "results": results,
        "status": tool_response.get("status"),
    }
    # Error identifiers are public protocol state, not labels. Preserve them
    # so terminal-budget and invalid-call closures remain observable to the
    # model after serialization.
    for key in ("error", "message", "errors"):
        if key in tool_response:
            payload[key] = tool_response[key]
    return payload


def _tool_response_message(sample: dict[str, Any], tool_response: dict[str, Any], reference_images: list[str]) -> dict[str, Any]:
    """Render public evidence as the native ``tool`` turn used in SFT.

    Reference images are deliberately omitted: frozen student trajectories use
    JSON evidence only, and repeated reference-image tokens are not part of
    the trained state transition.
    """
    del sample, reference_images
    return {"role": "tool", "content": json.dumps(_training_tool_response(tool_response), ensure_ascii=False, separators=(",", ":"))}


def _terminal_tool_response_message(
    sample: dict[str, Any],
    tool_response: dict[str, Any],
    correction: str,
) -> dict[str, Any]:
    """Render the terminal closure in the SFT-observed tool state.

    Swift manual-JSON trajectories cannot represent ``tool -> user ->
    assistant``.  The closure dataset consequently appends the public
    label-blind instruction to the final tool observation.  Use the exact same
    state at evaluation time; an unmatched standalone user correction is a
    different transition and was shown to trigger further tool calls.
    """
    message = _tool_response_message(sample, tool_response, [])
    message["content"] = f"{message['content']}\n{correction}"
    return message



def _trim_reference_images(reference_image_paths: list[str], limit: int = 0) -> list[str]:
    # Keep reference-image IDs in the textual RAG response, but do not append
    # image payloads on every tool turn. Repeated vision tokens can otherwise
    # exceed the deployed model's 8192-token context after three RAG calls.
    if limit <= 0:
        return []
    trimmed: list[str] = []
    for ref in reference_image_paths:
        if not isinstance(ref, str) or not ref:
            continue
        trimmed.append(ref)
        if len(trimmed) >= limit:
            break
    return trimmed


def _build_eval_sft_messages() -> list[dict[str, str]]:
    return [sft_user_message({}, 0)]


def _chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    timeout: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _execute_rag_call(rag_api: str, image_path: Path, arguments: dict[str, Any], timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    retrieval_type = str(arguments["retrieval_type"])
    query = str(arguments.get("query") or "").strip()
    body: dict[str, Any] = {
        "top_k": int(arguments["top_k"]),
        "preset": retrieval_type,
    }
    if query:
        body["text"] = query
    if arguments.get("image") == "query_image":
        body["image_path"] = str(image_path)
    for key in ("ranker", "text_weight", "image_weight", "sparse_weight"):
        if key in arguments and arguments[key] not in (None, ""):
            body[key] = arguments[key]

    endpoint = f"{rag_api.rstrip('/')}/search/{retrieval_type}"
    response = requests.post(endpoint, json=body, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or []
    reference_images: list[str] = []
    compact_hits = [compact_hit(hit, reference_images) for hit in hits[: int(arguments["top_k"])]]
    tool_response = {
        "status": "success",
        "retrieval_type": retrieval_type,
        "query": query,
        "results": compact_hits,
    }
    ledger = {
        "ok": True,
        "endpoint": endpoint,
        "request": body,
        "raw_response": raw,
        "visible_reference_images": reference_images,
    }
    return tool_response, ledger


def _normalize_public_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


def _public_name_terms(tool_history: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for item in tool_history:
        for result in (item.get("tool_response") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            for key in ("class_name", "chinese_name"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    terms.add(_normalize_public_name(value))
            for key in ("aliases", "chinese_aliases"):
                for value in result.get(key) or []:
                    if isinstance(value, str) and value.strip():
                        terms.add(_normalize_public_name(value))
            for similar in result.get("similar_classes") or []:
                if isinstance(similar, dict):
                    for key in ("name", "name_zh"):
                        value = similar.get(key)
                        if isinstance(value, str) and value.strip():
                            terms.add(_normalize_public_name(value))
    return {term for term in terms if term}


def _call_signature(arguments: dict[str, Any]) -> str:
    fields = ("query", "retrieval_type", "image", "top_k", "ranker", "text_weight", "image_weight", "sparse_weight")
    return json.dumps({key: arguments.get(key) for key in fields if arguments.get(key) is not None}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _call_lineage_errors(arguments: dict[str, Any], tool_history: list[dict[str, Any]]) -> list[str]:
    """Reject non-public name lookups and no-value repeated calls.

    The check uses only model-emitted arguments and previous public tool turns;
    it never sees manifest labels, option targets, or teacher metadata.
    """
    errors: list[str] = []
    kind = arguments.get("retrieval_type")
    query = str(arguments.get("query") or "")
    if kind == "name":
        normalized = _normalize_public_name(query)
        if not tool_history:
            errors.append("name retrieval is not allowed before public retrieval evidence")
        elif not normalized or normalized not in _public_name_terms(tool_history):
            errors.append("name retrieval query must exactly copy a class name or alias from public evidence")
    signature = _call_signature(arguments)
    if any(_call_signature((item.get("tool_call") or {}).get("arguments") or {}) == signature for item in tool_history):
        errors.append("duplicate retrieval request would not add public evidence")
    return errors


def _fallback_final_answer(text: str) -> str:
    stripped = text.strip()
    if "<answer>" in stripped.lower():
        return stripped
    return f"<answer>{stripped}</answer>"


def _output_form(content: str, calls: list[dict[str, Any]], noncanonical: bool, malformed_reason: str | None = None) -> str:
    """Classify a model response without inspecting labels or retrieval truth."""
    stripped = content.strip()
    if calls:
        if noncanonical:
            return "mixed_or_noncanonical_tool_call"
        if stripped.startswith("<tool_call>"):
            return "xml_tool_call"
        if stripped.startswith("{"):
            return "bare_json_tool_call"
        return "recognized_tool_call"
    if malformed_reason:
        return f"invalid_tool_call:{malformed_reason}"
    if is_pre_tool_think(stripped):
        return "planning_think"
    if "<answer>" in stripped:
        return "answer"
    return "other"


def _parse_eval_tool_calls(content: str) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Parse strict Hermes calls, with a narrow evaluator-only recovery.

    Training validation deliberately rejects mixed assistant output.  At
    evaluation, a model can emit one otherwise-valid public call wrapped with
    incidental text.  Recover only that single, schema-valid call and record
    it as noncanonical; malformed or multiple calls remain invalid.
    """
    # Swift's manual-JSON template renders a training ``tool_call`` role as a
    # bare assistant JSON object. Prefer that exact served form; retain XML
    # parsing only for backwards-compatible recovery of historical outputs.
    try:
        bare = json.loads(content.strip())
    except (json.JSONDecodeError, TypeError):
        bare = None
    if (isinstance(bare, dict) and bare.get("name") == TOOL_NAME
            and isinstance(bare.get("arguments"), dict)
            and not validate_tool_arguments(bare["arguments"])):
        return [{"name": TOOL_NAME, "arguments": bare["arguments"]}], False, None
    # The SFT trace has a pure planning assistant turn followed by a bare JSON
    # tool turn.  The served model can occasionally concatenate the next
    # trained final-answer turn in the same generation.  Recover exactly one
    # schema-valid bare object from that mixed response, but mark it
    # noncanonical; multiple, malformed, or wrong-schema objects remain
    # strict failures below.  This is parallel to the existing XML mixed-call
    # recovery and never invents a retrieval call.
    decoder = json.JSONDecoder()
    recovered_bare: list[dict[str, Any]] = []
    for match in re.finditer(r"\{\s*\"name\"", content):
        try:
            candidate, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if (isinstance(candidate, dict) and candidate.get("name") == TOOL_NAME
                and isinstance(candidate.get("arguments"), dict)
                and not validate_tool_arguments(candidate["arguments"])):
            recovered_bare.append(candidate)
    if recovered_bare:
        # Several schema-valid calls in one generation are a noncanonical but
        # recoverable multi-query trajectory. Invalid or mixed-schema objects
        # are still rejected below; only validated calls are exposed.
        return [
            {"name": TOOL_NAME, "arguments": call["arguments"]}
            for call in recovered_bare
        ], True, None
    strict = parse_hermes_tool_calls(content, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments)
    if strict:
        if len(strict) == 1:
            return strict, False, None
        return [], False, "multiple_tool_calls"
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
    stripped = content.strip()
    if matches:
        if len(matches) != 1:
            return [], False, "multiple_tool_calls"
        try:
            call = json.loads(matches[0])
        except json.JSONDecodeError:
            return [], False, "malformed_xml_json"
        if not isinstance(call, dict):
            return [], False, "xml_call_not_object"
        if call.get("name") != TOOL_NAME:
            return [], False, "wrong_tool_name"
        if not isinstance(call.get("arguments"), dict) or validate_tool_arguments(call["arguments"]):
            return [], False, "invalid_tool_arguments"
        # A valid XML call embedded in other text is recoverable but is not a
        # canonical trained transition.
        return [{"name": TOOL_NAME, "arguments": call["arguments"]}], True, None
    if "<tool_call" in stripped.lower() or "</tool_call>" in stripped.lower():
        return [], False, "malformed_xml_envelope"

    # Do not flag ordinary final prose.  Bare JSON is a tool attempt only when
    # it begins like the trained wire form or mentions a tool-specific key.
    looks_like_bare_call = stripped.startswith("{") or any(token in stripped for token in ("agrinet_rag_search", '"arguments"', '"name"'))
    if not looks_like_bare_call:
        return [], False, None
    try:
        call = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if stripped[end:].lstrip().startswith("{"):
                return [], False, "multiple_bare_json_objects"
        return [], False, "malformed_bare_json"
    if not isinstance(call, dict):
        return [], False, "bare_call_not_object"
    if call.get("name") != TOOL_NAME:
        return [], False, "wrong_tool_name"
    if not isinstance(call.get("arguments"), dict) or validate_tool_arguments(call["arguments"]):
        return [], False, "invalid_tool_arguments"
    return [{"name": TOOL_NAME, "arguments": call["arguments"]}], False, None


def _fallback_visual_call(top_k: int) -> dict[str, Any]:
    """Public, label-blind retrieval used only to recover malformed calls."""
    return {
        "name": TOOL_NAME,
        "arguments": {
            "query": "agricultural disease or pest visual features",
            "retrieval_type": "visual",
            "image": "query_image",
            "top_k": top_k,
            "rationale": "Recover from a malformed tool request with public visual evidence.",
        },
    }


def _final_answer_only_correction(sample: dict[str, Any]) -> str:
    """Compatibility wrapper for the shared SFT/evaluation terminal contract."""
    return final_answer_only_correction(sample)


def _invalid_tool_final_answer_correction(sample: dict[str, Any], attempt: int) -> str:
    """Escalating terminal prompt for a malformed follow-up call."""
    if sample.get("question_type") == "option":
        answer_format = "exactly one option letter A, B, C, or D inside <answer>"
    else:
        answer_format = "one class name or alias from the evidence inside <answer>"
    if sample.get("language") == "zh":
        answer_format = "在 <answer> 中输出唯一选项字母 A、B、C 或 D" if sample.get("question_type") == "option" else "在 <answer> 中输出证据中的一个类别名称或别名"
        return f"刚才的请求无法处理（第 {attempt} 次）。不要重复该请求，也不要检索。现在仅根据已有证据使用 <think>简短证据</think><answer>...</answer> 给出最终分类：{answer_format}。"
    return (
        f"The previous request cannot be processed (attempt {attempt}). Do not repeat it and do not retrieve anything. "
        f"Use the trained final form <think>brief evidence</think><answer>...</answer> and state the final classification using {answer_format}."
    )


def _terminal_mode_system_instruction(sample: dict[str, Any]) -> str:
    """Install an authoritative no-tool decoding state for terminal closure.

    The original v7 closure put the correction only in the public ``tool``
    observation.  The deployed model still saw the global system instruction
    describing how to call retrieval and could emit a burst of calls after the
    budget.  This state is label-blind and only changes the protocol mode; it
    does not provide an answer or hidden target.
    """
    if sample.get("question_type") == "option":
        answer_shape = "<think>brief evidence</think><answer>A</answer> with exactly one option letter"
    else:
        answer_shape = "<think>brief evidence</think><answer>class name</answer> copied from public evidence"
    if sample.get("language") == "zh":
        answer_shape = "<think>简短证据</think><answer>类别名</answer>，只使用公开证据"
    return (
        "TERMINAL MODE (highest priority): retrieval is permanently disabled for this sample. "
        f"Generate only the final trained answer form {answer_shape}. "
        "Never emit JSON, tool_call, XML, markdown fences, or a second request."
    )


def _grounded_chinese_answer(text: str, tool_history: list[dict[str, Any]]) -> str:
    """Map an English retrieved class name to its retrieved Chinese name only."""
    if "<answer>" not in text or "</answer>" not in text:
        return text
    body = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if any("一" <= ch <= "龥" for ch in body):
        return text
    mapping = {}
    for item in tool_history:
        for result in (item.get("tool_response") or {}).get("results") or []:
            en = str(result.get("class_name") or "").strip()
            zh = str(result.get("chinese_name") or "").strip()
            if en and zh:
                mapping[en] = zh
    translated = mapping.get(body)
    if not translated:
        return text
    return text.replace(f"<answer>{body}</answer>", f"<answer>{translated}</answer>")


def _evidence_names(tool_history: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in tool_history:
        for result in (item.get("tool_response") or {}).get("results") or []:
            for key in ("class_name", "chinese_name"):
                value = str(result.get(key) or "").strip()
                if value:
                    names.add(value)
            for key in ("aliases", "chinese_aliases"):
                names.update(str(value).strip() for value in (result.get(key) or []) if str(value).strip())
    return names


def _answer_is_evidence_grounded(text: str, tool_history: list[dict[str, Any]], sample: dict[str, Any]) -> bool:
    if "<answer>" not in text or "</answer>" not in text:
        return False
    body = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if sample.get("question_type") == "option":
        return body in {"A", "B", "C", "D"}
    return body in _evidence_names(tool_history)


def _grounded_final_answer(text: str, tool_history: list[dict[str, Any]], sample: dict[str, Any]) -> str:
    """Ensure the visible answer is copied from the current tool evidence.

    This is a conservative last-mile guard: it may select only a returned class
    name/alias, and never translates or consults the hidden target label.
    """
    if sample.get("question_type") == "option" or "<answer>" not in text or "</answer>" not in text:
        return text
    body = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    names: list[tuple[str, str]] = []
    for item in tool_history:
        for result in (item.get("tool_response") or {}).get("results") or []:
            preferred = ("chinese_name", "chinese_aliases") if sample.get("language") == "zh" else ("class_name", "aliases")
            for key in preferred:
                values = result.get(key) if key.endswith("s") else [result.get(key)]
                for value in values or []:
                    value = str(value or "").strip()
                    if value:
                        names.append((value, str(result.get("rank") or 999)))
    exact = next((name for name, _ in names if name == body), None)
    if exact:
        return text
    normalized = re.sub(r"[_\s]+", " ", body).strip().lower()
    equivalent = next((name for name, _ in names if re.sub(r"[_\s]+", " ", name).strip().lower() == normalized), None)
    replacement = equivalent or (names[0][0] if names else None)
    if not replacement:
        return text
    return text.replace(f"<answer>{body}</answer>", f"<answer>{replacement}</answer>")


def _evaluate_sample(args: argparse.Namespace, row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
            image_path = _resolve(str(row["image_path"]), repo_root)
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")

            sample = _eval_sample(row, image_path)
            api_messages = _build_eval_messages(sample, image_path, args.top_k)
            sft_messages = _build_eval_sft_messages()
            final_text = ""
            tool_turns = 0
            forced_tool_turns = 0
            tool_history: list[dict[str, Any]] = []
            raw_messages: list[dict[str, Any]] = []
            protocol_trace: list[dict[str, Any]] = []
            error_text = ""
            language_correction_turns = 0
            answer_correction_turns = 0
            first_turn_reprompted = False
            planning_turns = 0
            protocol_errors: list[str] = []
            max_tool_turns_exceeded = 0
            terminal_answer_reprompts = 0
            terminal_answer_pending = False
            invalid_tool_reprompts = 0
            invalid_tool_call_policy = getattr(args, "invalid_tool_call_policy", "strict")
            malformed_tool_call_attempts = 0
            malformed_tool_call_reasons: dict[str, int] = {}
            noncanonical_recovered_calls = 0
            post_budget_tool_attempts = 0
            terminal_closure_used = 0
            terminal_closure_failed = 0
            queued_tool_calls: list[dict[str, Any]] = []
            terminal_mode_installed = False

            def install_terminal_mode() -> None:
                nonlocal terminal_mode_installed
                if terminal_mode_installed:
                    return
                api_messages[0]["content"] += "\n" + _terminal_mode_system_instruction(sample)
                terminal_mode_installed = True

            while True:
                if queued_tool_calls:
                    calls = [queued_tool_calls.pop(0)]
                    noncanonical_call = True
                    malformed_reason = None
                    content = json.dumps(calls[0], ensure_ascii=False, separators=(",", ":"))
                else:
                    try:
                        response = _chat_completion(
                            args.api_base,
                            args.api_key,
                            args.model,
                            api_messages,
                            # Terminal closure only needs a short think/answer
                            # pair.  The full formal budget lets a model that
                            # has entered a tool-call loop generate hundreds
                            # of additional tokens before the evaluator can
                            # reject the attempt.  Cap only this correction
                            # request; normal reasoning and retrieval turns
                            # retain the configured generation budget.
                            max_new_tokens=(
                                min(args.max_new_tokens, 128)
                                if terminal_answer_pending else args.max_new_tokens
                            ),
                            temperature=args.temperature,
                            timeout=args.request_timeout,
                            # Frozen RAG rows contain one pure assistant planning
                            # turn before their first tool call. Keep thinking
                            # enabled for that trained transition; disable it only
                            # when a terminal-only correction must close the run.
                            chat_template_kwargs={"enable_thinking": False} if terminal_answer_pending else None,
                        )
                    except Exception as exc:
                        error_text = f"sglang request failed: {exc}"
                        final_text = f"<answer>{error_text}</answer>"
                        break

                    raw_messages.append(response)
                    message = extract_message(response)
                    content = str(message.get("content") or "")
                    calls, noncanonical_call, malformed_reason = _parse_eval_tool_calls(content)
                    if len(calls) > 1 and noncanonical_call:
                        queued_tool_calls.extend(calls[1:])
                        calls = calls[:1]
                if getattr(args, "capture_protocol_trace", False):
                    protocol_trace.append({
                        "tool_turns_before": tool_turns,
                        "form": _output_form(content, calls, noncanonical_call, malformed_reason),
                        "content_chars": len(content),
                        # Diagnostic smokes need the actual wire form to
                        # distinguish schema failures from mixed planning/JSON
                        # output.  Formal runs never enable this flag; cap the
                        # payload so repeated failures stay inspectable.
                        "content_preview": content[:2048],
                    })
                forced_call = False
                if not calls and is_pre_tool_think(content) and planning_turns == 0 and tool_turns == 0:
                    planning_turns += 1
                    api_messages.append({"role": "assistant", "content": content})
                    # The native/manual-JSON SFT trajectory is a pure planning
                    # turn followed by a bare JSON assistant object.  Asking
                    # for Hermes XML here reintroduces the exact wire-format
                    # mismatch this evaluator is intended to diagnose.
                    api_messages.append({
                        "role": "user",
                        "content": (
                            "Continue with exactly one bare JSON object with name agrinet_rag_search and its arguments; "
                            "do not answer yet, and do not use XML tags, markdown, or explanation."
                            if sample.get("language") != "zh" else
                            "继续时只输出一个包含 name=agrinet_rag_search 与 arguments 的裸 JSON 对象；暂时不要回答，"
                            "不要使用 XML 标签、Markdown 或解释。"
                        ),
                    })
                    continue
                invalid_tool_call = malformed_reason is not None
                if invalid_tool_call:
                    protocol_errors.append("invalid_hermes_tool_call")
                    malformed_tool_call_attempts += 1
                    malformed_tool_call_reasons[malformed_reason] = malformed_tool_call_reasons.get(malformed_reason, 0) + 1
                if noncanonical_call:
                    protocol_errors.append("noncanonical_hermes_tool_call")
                    noncanonical_recovered_calls += 1
                if (calls or invalid_tool_call) and tool_turns >= args.max_tool_turns:
                    post_budget_tool_attempts += 1
                # A terminal-only correction is a single, fixed state
                # transition. Any further tool attempt (valid or malformed)
                # is an explicit evaluation failure; never turn it into an
                # additional retrieval, fallback, or repeated prompt.
                if terminal_answer_pending and (calls or invalid_tool_call):
                    error_text = "model emitted a tool call despite terminal-answer correction"
                    final_text = f"<answer>{error_text}</answer>"
                    terminal_closure_failed += 1
                    break
                # The trained manual-JSON policy occasionally emits a final
                # answer directly on its first generation. Give it one
                # explicit protocol-only retry before treating the sample as
                # a no-retrieval failure (or using the optional forced call).
                if not calls and not invalid_tool_call and tool_turns == 0 and not first_turn_reprompted:
                    first_turn_reprompted = True
                    api_messages.append({
                        "role": "user",
                        "content": (
                            "Protocol correction: this is the first tool turn. Do not answer yet. Output exactly one bare JSON object with name agrinet_rag_search and its arguments; no XML tags, markdown, explanation, or <answer> tags."
                            if sample.get("language") != "zh" else
                            "协议校正：这是第一轮工具调用。现在不要回答。只输出一个包含 name=agrinet_rag_search 与 arguments 的裸 JSON 对象，不要输出 XML 标签、Markdown、解释或 <answer> 标签。"
                        ),
                    })
                    continue
                # The SFT trajectories contain an assistant observation/planning
                # message before the tool_call role. Some OpenAI-compatible
                # backends instead let the model jump directly to a final answer.
                # Keep the benchmark genuinely RAG-enabled by executing the
                # protocol's mandatory first visual retrieval in that case, and
                # record the fallback separately from model-emitted calls.
                if not calls and not invalid_tool_call and tool_turns == 0 and not args.disable_forced_first_call:
                    calls = [_fallback_visual_call(args.top_k)]
                    forced_call = True
                if (invalid_tool_call and invalid_tool_call_policy == "recovery"
                        and tool_turns > 0 and tool_turns < args.max_tool_turns):
                    # Preserve a genuine RAG trajectory when the model asks for
                    # another tool but its wire format is malformed.  The
                    # fallback is fixed, public, and label-blind; it avoids
                    # turning a transport/protocol typo into a missing answer.
                    calls = [_fallback_visual_call(args.top_k)]
                    forced_call = True

                # A malformed manual tool call is not a final answer.  Close
                # it with a public protocol error and request a final answer
                # from the evidence already available.  This applies after a
                # valid retrieval as well as on the first turn; no extra tool
                # execution is performed.
                if invalid_tool_call and not calls:
                    if invalid_tool_reprompts < 1:
                        invalid_tool_reprompts += 1
                        terminal_answer_pending = True
                        install_terminal_mode()
                        terminal_closure_used += 1
                        # Do not replay malformed assistant content into the
                        # model context: it is deliberately absent from SFT
                        # targets and reintroduces the forbidden XML/bare-JSON
                        # continuation prior.  Instead install the same
                        # native trainable terminal state as the budget path:
                        # a canonical assistant tool-call placeholder followed
                        # by a public non-executing tool observation containing
                        # the terminal answer contract. The raw attempt remains
                        # in protocol_trace for strict auditing.
                        terminal_call = {
                            "name": TOOL_NAME,
                            "arguments": {
                                "query": "invalid request not executed",
                                "retrieval_type": "visual",
                                "image": "query_image",
                                "top_k": args.top_k,
                                "rationale": "Public protocol rejection; no retrieval was executed.",
                            },
                        }
                        api_messages.append({
                            "role": "assistant",
                            "content": json.dumps(terminal_call, ensure_ascii=False, separators=(",", ":")),
                        })
                        api_messages.append(_terminal_tool_response_message(
                            sample,
                            {
                                "status": "error",
                                "error": "invalid_tool_call",
                                "message": "The tool call was invalid and was not executed. Use evidence already returned.",
                            },
                            _invalid_tool_final_answer_correction(sample, invalid_tool_reprompts),
                        ))
                        continue
                    error_text = "model repeated an invalid tool call despite final-answer correction"
                    final_text = f"<answer>{error_text}</answer>"
                    terminal_closure_failed += 1
                    break

                if calls and tool_turns < args.max_tool_turns:
                    call = calls[0]
                    emitted_args = dict(call.get("arguments") or {})
                    call_args = clamp_tool_args(emitted_args, args.top_k, sample, sft_messages)
                    visible_call = {"name": TOOL_NAME, "arguments": call_args}
                    sft_messages.append({"role": "tool_call", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})

                    validation_errors = [
                        *validate_tool_arguments(call_args),
                        *_call_lineage_errors(emitted_args, tool_history),
                    ]
                    if validation_errors:
                        tool_response = {"status": "error", "errors": validation_errors}
                        ledger = {"ok": False, "validation_errors": validation_errors, "request": visible_call, "visible_reference_images": []}
                    else:
                        try:
                            tool_response, ledger = _execute_rag_call(args.rag_api, image_path, call_args, timeout=args.request_timeout)
                        except Exception as exc:
                            tool_response = {
                                "status": "error",
                                "retrieval_type": call_args.get("retrieval_type"),
                                "query": call_args.get("query"),
                                "error": str(exc),
                            }
                            ledger = {"ok": False, "request": visible_call, "error": str(exc), "visible_reference_images": []}

                    tool_turns += 1
                    forced_tool_turns += int(forced_call)
                    tool_history.append({"tool_call": visible_call, "tool_response": tool_response, "ledger": ledger, "forced": forced_call})
                    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append(
                        _tool_response_message(
                            sample,
                            tool_response,
                            _trim_reference_images(ledger.get("visible_reference_images") or []),
                        )
                    )
                    continue

                # A model-generated call after the tool budget is exhausted is
                # not an answer.  The previous implementation fell through to
                # _fallback_final_answer and serialized the XML call inside
                # <answer>, which disproportionately invalidated option items.
                # Preserve the ordered transcript, then allow one terminal-only
                # correction; a repeated call is an explicit evaluation error.
                if calls or ("<tool_call>" in content and tool_turns >= args.max_tool_turns):
                    max_tool_turns_exceeded += 1
                    protocol_errors.append("max_tool_turns_exceeded")
                    if terminal_answer_reprompts < 1:
                        terminal_answer_reprompts += 1
                        # A single generation may contain several recovered
                        # JSON calls.  Only the first one is observable in the
                        # current assistant turn; discard the remaining
                        # parser queue when entering terminal mode.  Leaving
                        # it populated bypasses the terminal system state and
                        # deterministically turns a valid final answer into a
                        # false post-budget failure.
                        queued_tool_calls.clear()
                        # Close the manual tool turn with a public rejection
                        # response.  Leaving the emitted call unmatched makes
                        # the chat state incomplete and Qwen tends to emit a
                        # further call; this response neither executes a
                        # fourth retrieval nor exposes hidden information.
                        terminal_answer_pending = True
                        install_terminal_mode()
                        terminal_closure_used += 1
                        # A schema-valid over-budget call can be replayed in
                        # the same assistant JSON form used by successful SFT
                        # tool transitions before attaching the public error.
                        terminal_call = calls[0] if calls else None
                        terminal_content = (json.dumps(terminal_call, ensure_ascii=False, separators=(",", ":"))
                                            if terminal_call is not None else content)
                        api_messages.append({"role": "assistant", "content": terminal_content})
                        api_messages.append(_terminal_tool_response_message(
                            sample,
                            {
                                "status": "error",
                                "error": "tool_budget_exhausted",
                                "message": "No additional retrieval is available. Use the evidence already returned.",
                            },
                            _final_answer_only_correction(sample),
                        ))
                        continue
                    error_text = (
                        f"model emitted a tool call after max_tool_turns={args.max_tool_turns} "
                        "despite terminal-answer correction"
                    )
                    final_text = f"<answer>{error_text}</answer>"
                    terminal_closure_failed += 1
                    break

                answer_body = content.split("<answer>", 1)[-1].split("</answer>", 1)[0] if "<answer>" in content else ""
                if (answer_correction_turns < 1 and tool_history
                        and not _answer_is_evidence_grounded(content, tool_history, sample)):
                    answer_correction_turns += 1
                    if sample.get("question_type") == "option":
                        correction = ("答案校正：这是选择题。只根据题目选项和检索证据，在 <answer> 中输出唯一选项字母 A、B、C 或 D，不要输出类别名称。"
                                      if sample.get("language") == "zh" else
                                      "Answer correction: this is multiple choice. Based only on the options and retrieved evidence, output exactly one option letter A, B, C, or D inside <answer>, not a class name.")
                    else:
                        correction = ("答案校正：最终 <answer> 必须逐字使用检索结果中的中文规范类别名或中文别名；不得自行翻译、改写、补充未出现在证据中的名称。若当前证据不足以支持原答案，请选择当前检索结果中最匹配的中文类别名，不要猜测题目标注。"
                                      if sample.get("language") == "zh" else
                                      "Answer correction: the final <answer> must copy an English class name or alias exactly from the retrieved evidence; do not invent, translate, or rewrite a label not present in evidence. If the original answer is unsupported, choose the closest class name currently present in the retrieved results rather than guessing the hidden label.")
                    api_messages.append({"role": "user", "content": correction})
                    continue
                if (sample.get("language") == "zh" and sample.get("question_type") != "option"
                        and language_correction_turns == 0
                        and (not any("一" <= ch <= "龥" for ch in answer_body)
                             or any(marker in content for marker in ("Evidence:", "Rejected alternatives:", "Uncertainty:")))):
                    language_correction_turns += 1
                    api_messages.append({
                        "role": "user",
                        "content": "语言校正：这是中文 query。请将刚才的最终回答完整改写为中文；只保留中文思维链标题‘证据、排除的候选、不确定性’，并在 <answer> 中输出检索证据支持的中文规范类别名称。不要重新检索，不要输出英文标题。",
                    })
                    continue
                final_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                final_text = _fallback_final_answer(final_text)
                if sample.get("language") == "zh" and sample.get("question_type") != "option":
                    final_text = _grounded_chinese_answer(final_text, tool_history)
                final_text = _grounded_final_answer(final_text, tool_history, sample)
                break

            out = dict(row)
            out["prediction"] = final_text.strip()
            out["tool_turns"] = tool_turns
            out["forced_tool_turns"] = forced_tool_turns
            out["tool_history"] = tool_history
            out["protocol"] = {
                "format": "swift-hermes/v1", "planning_turns": planning_turns,
                "valid_tool_calls": tool_turns - forced_tool_turns,
                "protocol_errors": protocol_errors,
                "has_invalid_tool_call": bool(malformed_tool_call_attempts),
                "has_protocol_event": bool(protocol_errors),
                "malformed_tool_call_attempts": malformed_tool_call_attempts,
                "malformed_tool_call_reasons": malformed_tool_call_reasons,
                "noncanonical_recovered_calls": noncanonical_recovered_calls,
                "post_budget_tool_attempts": post_budget_tool_attempts,
                "terminal_closure_used": terminal_closure_used,
                "terminal_closure_failed": terminal_closure_failed,
                "forced_fallback_turns": forced_tool_turns,
                "answer_format_corrections": answer_correction_turns,
                "invalid_tool_call_policy": invalid_tool_call_policy,
                "max_tool_turns_exceeded": max_tool_turns_exceeded,
                "terminal_answer_reprompts": terminal_answer_reprompts,
                "invalid_tool_reprompts": invalid_tool_reprompts,
            }
            if getattr(args, "capture_protocol_trace", False):
                out["protocol_trace"] = protocol_trace
            out["student_user_query"] = STUDENT_USER_QUERY
            if error_text:
                out["error"] = error_text
            return out


async def run(args: argparse.Namespace) -> None:
    if args.max_concurrent < 1 or args.request_retries < 0 or args.snapshot_every < 1:
        raise ValueError("--max-concurrent and --snapshot-every must be positive; --request-retries must be non-negative")
    manifest = Path(args.manifest)
    rows = durable_load_jsonl(manifest)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    ids = validate_manifest(rows)
    fingerprint = request_fingerprint(manifest=manifest, protocol=PROTOCOL_VERSION, model=args.model, parameters={
        "offset": args.offset, "limit": args.limit, "top_k": args.top_k, "max_tool_turns": args.max_tool_turns,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "disable_forced_first_call": args.disable_forced_first_call,
        "invalid_tool_call_policy": args.invalid_tool_call_policy, "terminal_policy": "one-terminal-closure-v1",
    })
    store = SnapshotStore(Path(args.output), fingerprint, resume=args.resume)
    completed = store.completed(set(ids))
    semaphore = asyncio.Semaphore(args.max_concurrent)
    lock = asyncio.Lock()
    count = len(completed)
    root = Path(args.repo_root).resolve()

    async def one(row: dict[str, Any]) -> None:
        nonlocal count
        item_id = str(row["id"])
        if item_id in completed:
            return
        try:
            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(args.request_retries + 1):
                    try:
                        result = await asyncio.to_thread(_evaluate_sample, args, row, root)
                        if result.get("error"):
                            # A completed strict state-machine failure is
                            # model/protocol evidence, not a transient request
                            # failure.  Persist it exactly once (including the
                            # optional bounded protocol trace) rather than
                            # replaying the same sample and replacing that
                            # evidence with an opaque outer retry error.
                            # Only retry failures where no model response was
                            # obtained from SGLang.
                            if str(result["error"]).startswith("sglang request failed:"):
                                raise RuntimeError(str(result["error"]))
                            result["request_attempts"] = attempt + 1
                            break
                        result["request_attempts"] = attempt + 1
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise RuntimeError(f"sample failed after {args.request_retries + 1} attempts: {last_error}")
        except Exception as exc:
            result = dict(row)
            result.update({"prediction": "", "error": str(exc), "request_attempts": args.request_retries + 1, "tool_history": []})
        async with lock:
            completed[item_id] = result
            store.append(result)
            count += 1
            if count % args.snapshot_every == 0 or count == len(rows):
                print(f"[{count}/{len(rows)}] {item_id}", flush=True)

    await asyncio.gather(*(one(row) for row in rows))
    store.finalize(ids, completed)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
