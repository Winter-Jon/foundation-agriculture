#!/usr/bin/env python3
"""Run AgriNet Qwen3-VL evaluation with SFT-style RAG tool-calling over an sglang OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from tools.rag_distill.run_pilot import (
    api_tool_response_prompt,
    STUDENT_USER_QUERY,
    clamp_tool_args,
    compact_hit,
    extract_message,
    image_url_content,
    normalize_tool_calls,
    one_shot_example,
    sft_user_message,
    user_prompt,
)
from tools.rag_distill.schema import TOOL_NAME, tool_schema, validate_tool_arguments


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
    parser.add_argument("--top-k", type=int, default=3, help="Default RAG top-k.")
    parser.add_argument("--max-tool-turns", type=int, default=3, help="Maximum model-driven tool turns.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=300)
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
    candidate_names = [str(x).strip() for x in (row.get("candidate_names") or []) if str(x).strip()]
    candidate_codes = [str(x).strip() for x in (row.get("candidate_codes") or []) if str(x).strip()]
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
    }


def _build_eval_messages(sample: dict[str, Any], image_path: Path, top_k: int) -> list[dict[str, Any]]:
    system = (
        "You are an agricultural visual recognition assistant using manual JSON tool-calling. "
        "You must use a manual JSON tool-call protocol, not provider-native function calling. "
        "Every assistant message must be exactly one of two forms: (1) a single JSON tool-call object and no other text, "
        "or (2) the final response as <think>...</think> followed by <answer>...</answer>. Never mix a JSON tool call with explanation or a final answer. "
        "The first assistant message must be a single agrinet_rag_search JSON call. Do not output a bare arguments object. "
        f'The required shape is exactly: {{"name":"agrinet_rag_search","arguments":{{"query":"...","retrieval_type":"visual","image":"query_image","top_k":{top_k},"rationale":"..."}}}}. '
        f'The student-facing user request is: "{STUDENT_USER_QUERY}" Use it as the starting point for natural English tool queries, but keep each query shorter than the full instruction when possible. '
        "Good first-query examples are simple English phrases or questions such as `what disease is on this leaf`, `leaf spots and edge shape`, or `brown spots on crop leaf`. After the first retrieval, first state candidate hypotheses from Visual Observation plus retrieved results, then search again. Candidate hypotheses should be descriptive variants like `healthy-looking stone-fruit leaf`, `dark leaf-spot disease on a broadleaf crop`, or `rust-like lesions on a pome-fruit leaf`; they should not exactly copy a database class name such as `Cherry Normal leaf` unless the query is an exact name lookup for retrieved evidence. If a candidate is not already named in retrieved evidence, describe it neutrally with host/organ/symptom traits instead of writing a full database class name. Avoid Chinese, first-turn class-name guesses, and procedural text such as `retrieve top visual evidence before answering`. "
        "After tool responses, either emit another single JSON tool call or give the final response. The final response must include a <think> section first, then an <answer> section. "
        "The <think> section must include exactly these field labels: Evidence, Rejected alternatives, Uncertainty. The <answer> section must contain only the predicted canonical class name. Predict class names, not class codes. "
        "Use retrieved class names, aliases, scores, and reference image IDs as evidence. Keep the final visible reasoning concise and grounded in tool responses. "
        "Final answers are allowed only when retrieved evidence contains the predicted class or an alias; otherwise search again. "
        "Name lookup may use only an exact class name or alias already seen in tool evidence. Semantic, balanced, visual, and rrf follow-ups may compare retrieved similar class names, or may search neutral host/organ/symptom descriptions when an exact name is not yet supported. Do not use name lookup on the first turn, for descriptive phrases, or for inferred/guessed names that have not appeared in retrieved evidence. "
        f"{one_shot_example(top_k)} "
        "Tool schema: "
        f"{json.dumps(tool_schema(), ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user_prompt(sample, top_k)}, image_url_content(image_path)]},
    ]


def _trim_reference_images(reference_image_paths: list[str], limit: int = 2) -> list[str]:
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


def _chat_completion(api_base: str, api_key: str, model: str, messages: list[dict[str, Any]], *, max_new_tokens: int, temperature: float, timeout: int) -> dict[str, Any]:
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        },
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


def _fallback_final_answer(text: str) -> str:
    stripped = text.strip()
    if "<answer>" in stripped.lower():
        return stripped
    return f"<answer>{stripped}</answer>"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(Path(args.manifest))
    if args.limit:
        rows = rows[: args.limit]

    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            image_path = _resolve(str(row["image_path"]), repo_root)
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")

            sample = _eval_sample(row, image_path)
            api_messages = _build_eval_messages(sample, image_path, args.top_k)
            sft_messages = _build_eval_sft_messages()
            final_text = ""
            tool_turns = 0
            tool_history: list[dict[str, Any]] = []
            raw_messages: list[dict[str, Any]] = []
            error_text = ""

            while True:
                try:
                    response = _chat_completion(
                        args.api_base,
                        args.api_key,
                        args.model,
                        api_messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        timeout=args.request_timeout,
                    )
                except Exception as exc:
                    error_text = f"sglang request failed: {exc}"
                    final_text = f"<answer>{error_text}</answer>"
                    break

                raw_messages.append(response)
                message = extract_message(response)
                calls = normalize_tool_calls(message)

                if calls and tool_turns < args.max_tool_turns:
                    call = calls[0]
                    call_args = clamp_tool_args(dict(call.get("arguments") or {}), args.top_k, sample, sft_messages)
                    visible_call = {"name": TOOL_NAME, "arguments": call_args}
                    sft_messages.append({"role": "tool_call", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})

                    validation_errors = validate_tool_arguments(call_args)
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
                    tool_history.append({"tool_call": visible_call, "tool_response": tool_response, "ledger": ledger})
                    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False)})
                    api_messages.append(
                        api_tool_response_prompt(
                            sample,
                            tool_response,
                            _trim_reference_images(ledger.get("visible_reference_images") or []),
                        )
                    )
                    continue

                content = message.get("content") or ""
                final_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                final_text = _fallback_final_answer(final_text)
                break

            out = dict(row)
            out["prediction"] = final_text.strip()
            out["tool_turns"] = tool_turns
            out["tool_history"] = tool_history
            out["student_user_query"] = STUDENT_USER_QUERY
            if error_text:
                out["error"] = error_text
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] {row['id']} tool_turns={tool_turns}", flush=True)


if __name__ == "__main__":
    main()
