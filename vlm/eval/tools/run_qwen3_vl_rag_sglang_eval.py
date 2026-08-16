#!/usr/bin/env python3
"""Run AgriNet Qwen3-VL evaluation with SFT-style RAG tool-calling over an sglang OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests

from tools.rag_distill.run_pilot import (
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
    parser.add_argument("--offset", type=int, default=0, help="Optional starting row offset (applied before limit).")
    parser.add_argument("--top-k", type=int, default=3, help="Default RAG top-k.")
    parser.add_argument("--max-tool-turns", type=int, default=3, help="Maximum model-driven tool turns.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--disable-forced-first-call", action="store_true", help="Do not inject the mandatory first retrieval when the model answers directly.")
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
    option_task = sample.get("question_type") == "option"
    chinese = sample.get("language") == "zh"
    answer_contract = (
        ("<answer> 中只能包含题目选项字母 A、B、C 或 D。" if chinese else
         "The <answer> section must contain only the selected option letter (A, B, C, or D). ")
        if option_task else
        ("<answer> 中只能包含检索证据支持的中文规范类别名称；可在括号中保留英文原名，但回答主体必须为中文。"
         if chinese else
         "The <answer> section must contain only the predicted canonical class name. ")
    )
    language_contract = (
        "这是中文 query。除工具 arguments.query 外，所有可见思维链、Evidence 字段、排除候选、不确定性和答案都必须使用中文；"
        "<think> 中必须使用中文标题：证据、排除的候选、不确定性。工具查询仍使用简短英文。"
        if chinese else
        "This is an English query. All visible reasoning and the final answer must be in English; do not use Chinese."
    )
    system = ("You are an agricultural visual recognition assistant using manual JSON tool-calling. "
        "You must use a manual JSON tool-call protocol, not provider-native function calling. "
        "Every assistant message must be exactly one of two forms: (1) a single JSON tool-call object and no other text, "
        "or (2) the final response as <think>...</think> followed by <answer>...</answer>. Never mix a JSON tool call with explanation or a final answer. "
        "The first assistant message must be a single agrinet_rag_search JSON call. Do not output a bare arguments object. "
        f'The required shape is exactly: {{"name":"agrinet_rag_search","arguments":{{"query":"...","retrieval_type":"visual","image":"query_image","top_k":{top_k},"rationale":"..."}}}}. '
        f'The student-facing user request is: "{STUDENT_USER_QUERY}" Use it as the starting point for natural English tool queries, but keep each query shorter than the full instruction when possible. '
        "Good first-query examples are simple English phrases or questions such as `what disease is on this leaf`, `leaf spots and edge shape`, or `brown spots on crop leaf`. After the first retrieval, first state candidate hypotheses from Visual Observation plus retrieved results, then search again. Candidate hypotheses should be descriptive variants like `healthy-looking stone-fruit leaf`, `dark leaf-spot disease on a broadleaf crop`, or `rust-like lesions on a pome-fruit leaf`; they should not exactly copy a database class name such as `Cherry Normal leaf` unless the query is an exact name lookup for retrieved evidence. If a candidate is not already named in retrieved evidence, describe it neutrally with host/organ/symptom traits instead of writing a full database class name. Avoid Chinese, first-turn class-name guesses, and procedural text such as `retrieve top visual evidence before answering`. "
        "After tool responses, either emit another single JSON tool call or give the final response. The final response must include a <think> section first, then an <answer> section. "
    )
    system += " " + language_contract + " "
    system += ("The <think> section must include exactly these Chinese field labels: 证据, 排除的候选, 不确定性。 " if chinese else
               "The <think> section must include exactly these field labels: Evidence, Rejected alternatives, Uncertainty. ")
    system += answer_contract
    system += ("Use retrieved class names, aliases, scores, and reference image IDs as evidence. Keep the final visible reasoning concise and grounded in tool responses. "
               "Final answers are allowed only when retrieved evidence contains the predicted class or an alias; otherwise search again. "
               "Name lookup may use only an exact class name or alias already seen in tool evidence. Semantic, balanced, visual, and rrf follow-ups may compare retrieved similar class names, or may search neutral host/organ/symptom descriptions when an exact name is not yet supported. Do not use name lookup on the first turn, for descriptive phrases, or for inferred/guessed names that have not appeared in retrieved evidence. ")
    system += f"{one_shot_example(top_k)} Tool schema: {json.dumps(tool_schema(), ensure_ascii=False)}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user_prompt(sample, top_k).replace(STUDENT_USER_QUERY, question)}, image_url_content(image_path)]},
    ]


def _tool_response_message(sample: dict[str, Any], tool_response: dict[str, Any], reference_images: list[str]) -> dict[str, Any]:
    """Render a public-only continuation for blind evaluation.

    ``api_tool_response_prompt`` is intentionally a distillation helper and may
    append an Oracle teacher-forcing context.  Evaluation must never expose a
    manifest label, alias, or target-derived option mapping to the model.
    """
    chinese = sample.get("language") == "zh"
    option = sample.get("question_type") == "option"
    if chinese:
        continuation = (
            "如果证据不足，只输出下一次 JSON 工具调用；否则输出 <think>...</think><answer>...</answer>。"
            "<think> 必须使用标题：证据、排除的候选、不确定性。"
            + ("<answer> 只能是 A、B、C 或 D 中的一个选项字母。" if option else "<answer> 只能是检索证据中出现的中文规范类别名称或中文别名。")
        )
    else:
        continuation = (
            "If evidence is insufficient, output only the next JSON tool call; otherwise output <think>...</think><answer>...</answer>. "
            "Use Evidence, Rejected alternatives, and Uncertainty headings. "
            + ("The <answer> must contain exactly one option letter A, B, C, or D." if option else "The <answer> must copy an English class name or alias appearing in retrieved evidence.")
        )
    text = (
        "Tool response JSON:\n" + json.dumps(tool_response, ensure_ascii=False) + "\n\n"
        + "Reference image IDs in this public result correspond to the attached images. " + continuation
        + " Do not use hidden labels, ground truth, or private target information."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for ref in reference_images:
        path = Path(ref)
        if path.exists():
            content.append(image_url_content(path))
    return {"role": "user", "content": content}


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


def _fallback_final_answer(text: str) -> str:
    stripped = text.strip()
    if "<answer>" in stripped.lower():
        return stripped
    return f"<answer>{stripped}</answer>"


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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(Path(args.manifest))
    if args.offset:
        rows = rows[args.offset :]
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
            forced_tool_turns = 0
            tool_history: list[dict[str, Any]] = []
            raw_messages: list[dict[str, Any]] = []
            error_text = ""
            language_correction_turns = 0
            answer_correction_turns = 0
            first_turn_reprompted = False

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
                        # Qwen3-VL's native template otherwise prepends a
                        # thinking block before the protocol's first JSON
                        # call. Disable thinking only for the first turn;
                        # later turns retain the normal final-answer contract.
                        chat_template_kwargs={"enable_thinking": False} if tool_turns == 0 else None,
                    )
                except Exception as exc:
                    error_text = f"sglang request failed: {exc}"
                    final_text = f"<answer>{error_text}</answer>"
                    break

                raw_messages.append(response)
                message = extract_message(response)
                calls = normalize_tool_calls(message)
                forced_call = False
                # The trained manual-JSON policy occasionally emits a final
                # answer directly on its first generation. Give it one
                # explicit protocol-only retry before treating the sample as
                # a no-retrieval failure (or using the optional forced call).
                if not calls and tool_turns == 0 and not first_turn_reprompted:
                    first_turn_reprompted = True
                    api_messages.append({
                        "role": "user",
                        "content": (
                            "Protocol correction: this is the first turn. Do not answer yet. Output exactly one JSON object with name agrinet_rag_search and its arguments; no markdown, explanation, or <answer> tags."
                            if sample.get("language") != "zh" else
                            "协议校正：这是第一轮。现在不要回答。只输出一个 name 为 agrinet_rag_search 且包含 arguments 的 JSON 对象，不要输出 Markdown、解释或 <answer> 标签。"
                        ),
                    })
                    continue
                # The SFT trajectories contain an assistant observation/planning
                # message before the tool_call role. Some OpenAI-compatible
                # backends instead let the model jump directly to a final answer.
                # Keep the benchmark genuinely RAG-enabled by executing the
                # protocol's mandatory first visual retrieval in that case, and
                # record the fallback separately from model-emitted calls.
                if not calls and tool_turns == 0 and not args.disable_forced_first_call:
                    calls = [{
                        "name": TOOL_NAME,
                        "arguments": {
                            "query": "agricultural disease or pest visual features",
                            "retrieval_type": "visual",
                            "image": "query_image",
                            "top_k": args.top_k,
                            "rationale": "Establish visual candidates before the final identification.",
                        },
                    }]
                    forced_call = True

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
                    forced_tool_turns += int(forced_call)
                    tool_history.append({"tool_call": visible_call, "tool_response": tool_response, "ledger": ledger, "forced": forced_call})
                    sft_messages.append({"role": "tool_response", "content": json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))})
                    api_messages.append({"role": "assistant", "content": json.dumps(visible_call, ensure_ascii=False)})
                    api_messages.append(
                        _tool_response_message(
                            sample,
                            tool_response,
                            _trim_reference_images(ledger.get("visible_reference_images") or []),
                        )
                    )
                    continue

                content = message.get("content") or ""
                answer_body = content.split("<answer>", 1)[-1].split("</answer>", 1)[0] if "<answer>" in content else ""
                if (answer_correction_turns < 2 and tool_history
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
            out["student_user_query"] = STUDENT_USER_QUERY
            if error_text:
                out["error"] = error_text
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] {row['id']} tool_turns={tool_turns}", flush=True)


if __name__ == "__main__":
    main()
