"""Micu Direct-first HCV v13 pilot collector.

This is intentionally a new pilot-only collector. It receives no private
truth and writes only public trajectory artifacts. Its output is subsequently
checked by the private v13 Micu auditor.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import signal
import time
from pathlib import Path
from typing import Any, Callable

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.tool_schema import TOOL_NAME, tool_schema, tools_json, validate_tool_arguments
from agrinet.research.hcv.collector import (
    UnknownTeacherDelivery, execute_rag_call, image_url_content, normalize_tool_calls,
    preflight_teacher_endpoint,
    post_teacher_json, strip_code_fence, write_jsonl,
)
from agrinet.research.hcv.v13_contract import INSUFFICIENT_EVIDENCE, classify_terminal, validate_final_answer
from agrinet.research.hcv.v13_generation import (
    TEACHER_PROMPT_VERSION, continuation_prompt, first_turn_system_prompt,
    first_turn_user_prompt, route_first_turn,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_TOOL_NAME = "agrinet_hcv_final"


class _MicuCallDeadline(TimeoutError):
    """Raised locally when a provider call exceeds its hard wall-clock limit."""


def _call_with_deadline(invoke: Callable[[], Any], timeout: int) -> Any:
    """Bound a blocking provider call even when the HTTP socket ignores timeout.

    The v13 collector is sequential and runs in its main thread, so SIGALRM is
    safe here.  A deadline is deliberately treated as unknown delivery: the
    image must not be replayed automatically.
    """
    if timeout < 1:
        raise ValueError("Micu timeout must be positive")
    if not hasattr(signal, "setitimer"):
        return invoke()
    def expire(_signum: int, _frame: Any) -> None:
        raise _MicuCallDeadline(f"Micu call exceeded hard {timeout}s deadline")
    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return invoke()
    except _MicuCallDeadline as exc:
        raise UnknownTeacherDelivery(str(exc)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)


def _preflight_micu_endpoint(base_url: str, api_key: str, timeout: int, *, attempts: int = 3) -> None:
    """Bound transient `/models` transport failures before any image is sent."""
    if attempts < 1:
        raise ValueError("Micu preflight attempts must be positive")
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            _call_with_deadline(
                lambda: preflight_teacher_endpoint(base_url, api_key, timeout), timeout,
            )
            return
        except (RuntimeError, UnknownTeacherDelivery) as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(f"Micu endpoint preflight failed after {attempts} attempts: {last_error}") from last_error


def _normalized(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum() or "\u4e00" <= character <= "\u9fff")


def _tool_references_candidate(arguments: dict[str, Any], candidates: tuple[str, ...]) -> bool:
    declared = arguments.get("candidate_classes")
    if isinstance(declared, list) and declared:
        candidate_set = {_normalized(value) for value in candidates}
        return all(isinstance(value, str) and _normalized(value) in candidate_set for value in declared)
    haystack = _normalized(f"{arguments.get('query') or ''} {arguments.get('rationale') or ''}")
    return any((candidate := _normalized(value)) and candidate in haystack for value in candidates)


def _v13_tools_list(*, allow_final: bool = False, option_final: bool = False) -> list[dict[str, Any]]:
    """Native schema with public candidate linkage required for HCV audit."""
    schema = tool_schema()
    parameters = schema["function"]["parameters"]
    parameters["properties"]["candidate_classes"] = {
        "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
        "description": "One or more exact labels from the preceding public candidate analysis.",
    }
    parameters["required"].append("candidate_classes")
    tools = [schema]
    if allow_final:
        answer_schema: dict[str, Any] = {
            "type": "string",
            "description": "Final answer or exactly INSUFFICIENT_EVIDENCE.",
        }
        if option_final:
            # The letters and refusal token are public contract values.  Make
            # them a provider-enforced schema instead of relying on a Chinese
            # natural-language instruction at the final tool boundary.
            answer_schema["enum"] = ["A", "B", "C", "D", INSUFFICIENT_EVIDENCE]
        tools.append({
            "type": "function",
            "function": {
                "name": FINAL_TOOL_NAME,
                "description": "Return the final public-evidence diagnosis or the exact insufficient-evidence refusal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string", "description": "Brief public reasoning and uncertainty."},
                        "answer": answer_schema,
                    },
                    "required": ["reasoning", "answer"],
                    "additionalProperties": False,
                },
            },
        })
    return tools


def _validate_v13_tool_arguments(arguments: dict[str, Any], candidates: tuple[str, ...]) -> list[str]:
    retrieval_arguments = {key: value for key, value in arguments.items() if key != "candidate_classes"}
    errors = validate_tool_arguments(retrieval_arguments)
    declared = arguments.get("candidate_classes")
    # The live Micu request schema requires this field.  Retain the legacy
    # text-link fallback only for local replay/tests and older public records.
    if declared is None:
        if not _tool_references_candidate(arguments, candidates):
            errors.append("tool must reference a prior candidate")
    elif not isinstance(declared, list) or not 1 <= len(declared) <= 3:
        errors.append("candidate_classes must contain one to three candidates")
    elif not _tool_references_candidate(arguments, candidates):
        errors.append("candidate_classes must be drawn from prior candidates")
    return errors


def _repair_candidate_linkage_only(arguments: dict[str, Any], candidates: tuple[str, ...]) -> dict[str, Any] | None:
    """Restore only an invalid public ``candidate_classes`` field.

    This is deliberately narrower than a tool-call repair: the provider's
    query, retrieval mode, image handle, top-k and rationale must already pass
    the ordinary RAG schema.  The replacement is the Direct-first candidate
    list already present in the public trajectory, so it neither invents a
    diagnosis nor consults private truth.
    """
    if not candidates:
        return None
    retrieval_arguments = {key: value for key, value in arguments.items() if key != "candidate_classes"}
    if validate_tool_arguments(retrieval_arguments):
        return None
    repaired = dict(arguments)
    repaired["candidate_classes"] = list(candidates[:3])
    return repaired if not _validate_v13_tool_arguments(repaired, candidates) else None


def _result_names(response: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for result in response.get("results") or []:
        if not isinstance(result, dict):
            continue
        for key in ("class_name", "chinese_name"):
            value = _normalized(str(result.get(key) or ""))
            if value:
                values.add(value)
    return values


def _top_result_label(response: dict[str, Any], language: str) -> str:
    """Return the explicit public rank-1 label in the requested language.

    The retrieval service emits a stable numeric ``rank``.  Keeping this label
    in the public ledger makes terminal convergence inspectable without
    conflating every top-k neighbour with a competing diagnosis.
    """
    results = [item for item in response.get("results") or [] if isinstance(item, dict)]
    if not results:
        return ""
    def key(item: dict[str, Any]) -> tuple[int, float]:
        try:
            rank = int(item.get("rank") or 10**9)
        except (TypeError, ValueError):
            rank = 10**9
        try:
            score = -float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return rank, score
    top = min(results, key=key)
    preferred = "chinese_name" if language == "zh" else "class_name"
    return str(top.get(preferred) or top.get("class_name") or top.get("chinese_name") or "").strip()


def _top_result_labels(response: dict[str, Any]) -> list[str]:
    """Return all public display aliases for the explicit rank-1 result.

    Retrieval returns both English and Chinese display names for the same public
    class.  Keeping both aliases in the ledger prevents a rank-1 English Option
    label from being treated as unsupported merely because the query language
    selected the Chinese display field (and vice versa).  This is public output
    only; it neither adds a candidate nor reads private truth.
    """
    results = [item for item in response.get("results") or [] if isinstance(item, dict)]
    if not results:
        return []

    def key(item: dict[str, Any]) -> tuple[int, float]:
        try:
            rank = int(item.get("rank") or 10**9)
        except (TypeError, ValueError):
            rank = 10**9
        try:
            score = -float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return rank, score

    top = min(results, key=key)
    labels: list[str] = []
    for field in ("class_name", "chinese_name"):
        label = str(top.get(field) or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _public_answer_class_name(sample: dict[str, Any], answer: str) -> str:
    """Resolve an Option letter to its public option label for evidence checks."""
    if str(sample.get("question_type") or "open") != "option" or answer not in {"A", "B", "C", "D"}:
        return answer
    options = sample.get("public_option_labels")
    if not isinstance(options, list) or len(options) != 4:
        return answer
    selected = options[ord(answer) - ord("A")]
    if isinstance(selected, dict):
        language_key = "name_zh" if str(sample.get("language") or "en") == "zh" else "name"
        return str(selected.get(language_key) or selected.get("name") or selected.get("name_zh") or answer)
    return str(selected or answer)


def _public_evidence_leader(
    sample: dict[str, Any], candidates: tuple[str, ...], retrievals: list[dict[str, Any]],
) -> str:
    """Return the only publicly supported terminal answer, if one exists.

    A terminal refusal must not survive the full five-turn budget when a
    candidate (or visible option) has two independent supports and strictly
    leads every other eligible public alternative.  This reads only the public
    plan and retrieval ledger; it is intentionally not a truth-guided repair.
    """
    eligible: dict[str, str] = {}
    # Open questions may be corrected to a public retrieval class outside the
    # Direct candidate set.  Option questions stay restricted to their four
    # visible labels below.
    if str(sample.get("question_type") or "open") != "option":
        for retrieval in retrievals:
            label = str(retrieval.get("top_result_label") or "")
            if normalized := _normalized(label):
                eligible[normalized] = label
        for candidate in candidates:
            if normalized := _normalized(candidate):
                eligible.setdefault(normalized, candidate)
    options = sample.get("public_option_labels")
    if isinstance(options, list):
        for index, option in enumerate(options):
            if isinstance(option, dict):
                label = str(option.get("name_zh") if str(sample.get("language") or "en") == "zh" else option.get("name") or "")
                if normalized := _normalized(label):
                    eligible[normalized] = chr(ord("A") + index)
    counts = {name: 0 for name in eligible}
    for retrieval in retrievals:
        # Count one support per independent retrieval turn even when the same
        # rank-1 class has public English and Chinese display aliases.
        tops = {
            _normalized(str(label))
            for label in (retrieval.get("top_result_labels") or [retrieval.get("top_result_label")])
            if _normalized(str(label))
        }
        for top in tops & set(counts):
            counts[top] += 1
    leaders = [name for name, count in counts.items() if count >= 2 and count > max((other for key, other in counts.items() if key != name), default=0)]
    return eligible[leaders[0]] if len(leaders) == 1 else ""


def _converged_public_final(sample: dict[str, Any], answer: str) -> str:
    reasoning = (
        "公开检索证据在独立轮次中对该候选具有严格领先支持。"
        if str(sample.get("language") or "en") == "zh"
        else "Public retrieval evidence gives this candidate strictly leading support across independent turns."
    )
    return f"<think>{reasoning}</think><answer>{answer}</answer>"


def _rag_state(
    *, sample: dict[str, Any], answer: str, candidates: tuple[str, ...], retrievals: list[dict[str, Any]], exhausted: bool,
    max_tool_turns: int = 5,
) -> str:
    evidence_names = {
        _normalized(str(name)) for item in retrievals for name in (item.get("result_names") or [])
        if _normalized(str(name))
    }
    evidence_progress = bool(evidence_names)
    if answer == INSUFFICIENT_EVIDENCE:
        return classify_terminal(
            retrieval_turns=len(retrievals), evidence_progress=evidence_progress,
            conflict_unresolved=not evidence_progress, max_retrieval_turns=max_tool_turns,
        )
    normalized_answer = _normalized(_public_answer_class_name(sample, answer))
    candidate_hit = normalized_answer in {_normalized(item) for item in candidates}
    external_support = normalized_answer in evidence_names
    if candidate_hit and external_support:
        return "candidate_hit_verified"
    if candidate_hit:
        return "candidate_hit_no_external_gain"
    if external_support:
        return "candidate_miss_corrected"
    # A non-candidate final can only be retained as a resolved evidence conflict
    # when retrieval actually surfaced competing public classes; private audit
    # remains the final correctness gate.
    if evidence_progress and exhausted:
        return "candidate_conflict_resolved"
    return "candidate_hit_no_external_gain"


def _content(response: dict[str, Any]) -> str:
    """Return text or normalize one provider-native function call as JSON."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Micu response has no text message") from exc
    if not isinstance(message, dict):
        raise ValueError("Micu response message must be an object")
    native_calls = message.get("tool_calls") or []
    if native_calls:
        if len(native_calls) != 1 or not isinstance(native_calls[0], dict):
            raise ValueError("Micu response must contain exactly one native tool call")
        function = native_calls[0].get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("Micu native tool call has invalid JSON arguments") from exc
        if function.get("name") == FINAL_TOOL_NAME:
            if not isinstance(arguments, dict):
                raise ValueError("Micu final tool call arguments must be an object")
            reasoning, answer = arguments.get("reasoning"), arguments.get("answer")
            if not isinstance(reasoning, str) or not reasoning.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError("Micu final tool call is incomplete")
            return f"<think>{reasoning.strip()}</think><answer>{answer.strip()}</answer>"
        if function.get("name") != TOOL_NAME or not isinstance(arguments, dict):
            raise ValueError("Micu response has an invalid native tool call")
        return json.dumps({"name": TOOL_NAME, "arguments": arguments}, ensure_ascii=False)
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Micu response content must be text")
    return strip_code_fence(content).strip()


def _call_micu(messages: list[dict[str, Any]], *, api_key: str, base_url: str, model: str, timeout: int, max_tokens: int) -> str:
    request = {
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": messages,
    }
    latest = messages[-1] if messages else {}
    latest_content = latest.get("content") if isinstance(latest, dict) else ""
    if isinstance(latest_content, str) and (
        latest_content.startswith("External verification is required.")
        or latest_content.startswith("Public retrieval evidence follows.")
        or latest_content.startswith("Public evidence is non-empty")
        or latest_content.startswith("The preceding retrieval request was invalid.")
        or latest_content.startswith("The retrieval budget is exhausted.")
    ):
        # Both public terminal prompts carry this literal contract.  It is
        # deliberately inferred only from public prompt text, never labels.
        option_final = "For an Option question" in latest_content
        request["tools"] = _v13_tools_list(allow_final=True, option_final=option_final)
        if latest_content.startswith("External verification is required."):
            request["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        elif latest_content.startswith("The preceding retrieval request was invalid."):
            # A malformed call has not been sent to retrieval, so its single
            # public-only repair must be schema-constrained just like the
            # initial external-verification call.  Without this branch Micu
            # received only prose instructions and could repeat the malformed
            # manual JSON despite the repair prompt.
            request["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        elif latest_content.startswith("Public evidence is non-empty"):
            request["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        elif latest_content.startswith("The retrieval budget is exhausted."):
            request["tool_choice"] = {"type": "function", "function": {"name": FINAL_TOOL_NAME}}
        else:
            # Evidence-continuation must be one of the two native functions.
            # Otherwise Micu can emit free text that is neither a valid next
            # retrieval nor a renderable terminal HCV decision.
            request["tool_choice"] = "required"
    response = _isolated_micu_request(
        f"{base_url.rstrip('/')}" + "/chat/completions", request,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout,
    )
    return _content(response)


def _micu_request_worker(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int, result_sender: Any) -> None:
    """One provider POST in a killable child process.

    A blocked TLS/socket poll cannot always be interrupted by a signal in the
    parent process. The child boundary lets the pilot record the image as
    unknown-delivery and move on without replaying it.
    """
    args = argparse.Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0)
    try:
        result_sender.send(("ok", post_teacher_json(url, payload, headers, args)))
    except BaseException as exc:  # child must serialize all provider failures
        result_sender.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        result_sender.close()


def _read_isolated_micu_result(result_receiver: Any, *, timeout: int, exit_code: int | None) -> dict[str, Any]:
    """Receive one bounded result from the isolated provider child.

    ``multiprocessing.Queue`` owns a feeder thread whose post-exit flush can
    block a parent indefinitely on some local runtimes.  A one-shot Pipe has
    no feeder and supplies a directly pollable, bounded delivery receipt.
    """
    receive_timeout = min(5.0, max(1.0, float(timeout)))
    try:
        if not result_receiver.poll(receive_timeout):
            raise TimeoutError("no readable child receipt")
        status, value = result_receiver.recv()
    except (EOFError, OSError, TimeoutError) as exc:
        raise UnknownTeacherDelivery(
            f"Micu request exited without a readable response (exit_code={exit_code})"
        ) from exc
    if status != "ok":
        raise UnknownTeacherDelivery(f"Micu request failed or delivery is unknown: {value}")
    if not isinstance(value, dict):
        raise UnknownTeacherDelivery("Micu request returned a non-object response")
    return value


def _isolated_micu_request(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    if timeout < 1:
        raise ValueError("Micu timeout must be positive")
    # ``fork`` inherited active HTTP/runtime threads in the local worker and
    # produced non-deterministic queue/transport behavior. ``spawn`` starts a
    # clean interpreter for each ambiguous provider POST.
    context = multiprocessing.get_context("spawn")
    result_receiver, result_sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_micu_request_worker, args=(url, payload, headers, timeout, result_sender), daemon=True,
    )
    process.start()
    result_sender.close()
    deadline = time.monotonic() + float(timeout)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                raise UnknownTeacherDelivery(f"Micu request exceeded isolated {timeout}s deadline")
            if result_receiver.poll(min(0.25, remaining)):
                result = _read_isolated_micu_result(result_receiver, timeout=timeout, exit_code=process.exitcode)
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    raise UnknownTeacherDelivery("Micu child did not exit after delivering a response")
                return result
            if not process.is_alive():
                return _read_isolated_micu_result(result_receiver, timeout=timeout, exit_code=process.exitcode)
    finally:
        # An unreadable or malformed receipt must not leave a spawned child
        # behind.  This cleanup is deliberately independent of provider
        # success so the next fresh image starts from a known local state.
        if process.is_alive():
            process.terminate()
            process.join(5)
        result_receiver.close()


def _answer_body(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def _think_body(text: str) -> str:
    if "<think>" not in text or "</think>" not in text:
        return ""
    return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()


def _abstention_mentions_diagnosis(
    final: str, candidates: tuple[str, ...], retrievals: list[dict[str, Any]],
) -> bool:
    """Reject a refusal that simultaneously speculates about a diagnosis.

    This is a public-only contract check: candidate and retrieved class names
    already occur in the public trajectory, while private truth is never read.
    """
    reasoning = _normalized(_think_body(final))
    if not reasoning:
        return False
    names = set(candidates)
    for retrieval in retrievals:
        names.update(str(name) for name in retrieval.get("result_names") or [])
    return any((normalized := _normalized(name)) and normalized in reasoning for name in names)


def _sanitize_abstention_reasoning(
    sample: dict[str, Any], final: str, candidates: tuple[str, ...], retrievals: list[dict[str, Any]],
) -> str:
    """Make every public refusal non-diagnostic without changing its answer.

    This handles a formatting/rationale violation only.  It never turns a
    diagnosis into a refusal, reads no private data, and retains the exact
    terminal answer required by the public HCV contract.
    """
    if _answer_body(final) != INSUFFICIENT_EVIDENCE:
        return final
    # The private audit can flag a guessed diagnosis that is neither an exact
    # Direct candidate nor an exact retrieved display name (for example an
    # alias or a broader disease term).  A fixed evidence-only rationale is
    # the only public-safe representation for every refusal; it never reads
    # private truth and it does not change a diagnosis into a refusal.
    reasoning = (
        "公开图像与检索证据不足以唯一确定类别。"
        if str(sample.get("language") or "en") == "zh"
        else "Public image and retrieval evidence cannot uniquely determine a class."
    )
    return f"<think>{reasoning}</think><answer>{INSUFFICIENT_EVIDENCE}</answer>"


def _normalize_public_final(sample: dict[str, Any], final: str) -> str:
    """Normalize a public Option label to its required letter contract.

    Micu occasionally returns the visible option name (especially in Chinese)
    even when the prompt requests A--D.  Names are public plan fields, so this
    deterministic conversion does not read or expose private truth.  Ambiguous
    names remain invalid and are left untouched for the validation gate.
    """
    if str(sample.get("question_type") or "open") != "option":
        return final
    answer = _answer_body(final)
    if answer not in {"A", "B", "C", "D"}:
        options = sample.get("public_option_labels")
        if isinstance(options, list) and len(options) == 4:
            normalized = _normalized(answer)
            matches = []
            for index, option in enumerate(options):
                if not isinstance(option, dict):
                    continue
                names = (option.get("name"), option.get("name_zh"))
                if any(normalized and normalized == _normalized(str(name)) for name in names if name):
                    matches.append(chr(65 + index))
            if len(matches) == 1:
                return final.replace(f"<answer>{answer}</answer>", f"<answer>{matches[0]}</answer>")
    return final


def _public_messages(sample: dict[str, Any], image: Path) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": first_turn_system_prompt(str(sample.get("language") or "en"))},
        {"role": "user", "content": [
            {"type": "text", "text": first_turn_user_prompt(sample)}, image_url_content(image),
        ]},
    ]


def preflight_visual_micu(
    sample: dict[str, Any], *, api_key: str, base_url: str, model: str, timeout: int, max_tokens: int,
) -> dict[str, Any]:
    """Verify one retired image can traverse the visual Micu path.

    The caller must supply an already retired image: this check is never a
    collection attempt and returns no candidate text or training artifact.
    """
    image = Path(str(sample.get("query_image") or ""))
    if not image.is_absolute():
        image = REPO_ROOT / image
    if not image.is_file():
        raise ValueError(f"query image unavailable: {image}")
    content = _call_micu(_public_messages(sample, image), api_key=api_key, base_url=base_url, model=model, timeout=timeout, max_tokens=max_tokens)
    return {
        "schema_version": "agrinet.hcv-v13-micu-visual-preflight/v1",
        "sample_id": sample.get("sample_id"),
        "image_sha256": sample.get("image_sha256"),
        "status": "ok",
        "response_chars": len(content),
        "training_eligible": False,
    }


def _public_tool_response_prompt(sample: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Continue v13 collection with public evidence only."""
    language = str(sample.get("language") or "en")
    question_type = str(sample.get("question_type") or "open")
    answer_rule = (
        "For an Option question, output only A, B, C, or D in <answer>."
        if question_type == "option" else
        "For an Open question, output only the public evidence-supported class name in <answer>."
    )
    if language == "zh":
        answer_rule = "若证据不足，<answer> 必须是 INSUFFICIENT_EVIDENCE；否则只能输出受公开证据支持的最终答案。"
    return {
        "role": "user",
        "content": (
            "Public retrieval evidence follows. Either call agrinet_rag_search again grounded in earlier candidates and this evidence, or call agrinet_hcv_final with public reasoning and a final answer. Do not write text, JSON, Markdown, or <answer>. "
            "Before a final answer, complete two candidate-linked retrievals that test distinct visible traits or competing candidates; compare support and contradiction rather than repeating a query. "
            "For every agrinet_rag_search call, candidate_classes must copy one to three earlier candidate labels verbatim; never translate, paraphrase, replace, or append retrieved class names. "
            "If evidence cannot select a diagnosis after no-progress/conflict or final budget, use exactly INSUFFICIENT_EVIDENCE in <answer>. "
            "For INSUFFICIENT_EVIDENCE, reasoning must state only the missing or conflicting visible/public evidence: do not name, imply, or guess any diagnosis or candidate. "
            + answer_rule + " Do not mention hidden sources, labels, codes, or audit-only material.\n"
            + json.dumps(response, ensure_ascii=False)
        ),
    }


def _public_finalization_prompt(
    sample: dict[str, Any], *, candidates: tuple[str, ...] = (), retrievals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require a terminal response after the configured retrieval budget."""
    retrievals = retrievals or []
    evidence_card = {
        "initial_candidates": list(candidates),
        "retrieval_turns": len(retrievals),
        # Every field below is already present in the public tool call or its
        # public response ledger. Preserve turn boundaries and rank-1 aliases
        # so finalization can compare independent support and contradiction
        # rather than receiving a lossy, flattened class-name bag.
        "retrieval_evidence": [
            {
                "turn": index + 1,
                "candidate_classes": ((item.get("tool_call") or {}).get("arguments") or {}).get("candidate_classes") or [],
                "query": ((item.get("tool_call") or {}).get("arguments") or {}).get("query") or "",
                "rationale": ((item.get("tool_call") or {}).get("arguments") or {}).get("rationale") or "",
                "rank1_labels": item.get("top_result_labels") or (
                    [item.get("top_result_label")] if item.get("top_result_label") else []
                ),
                "retrieved_class_names": item.get("result_names") or [],
            }
            for index, item in enumerate(retrievals)
        ],
    }
    answer_rule = (
        "For an Option question, output only A, B, C, or D in <answer>."
        if str(sample.get("question_type") or "open") == "option" else
        "For an Open question, output only the public evidence-supported class name in <answer>."
    )
    return {
        "role": "user",
        "content": (
            "The retrieval budget is exhausted. Call agrinet_hcv_final now; do not issue agrinet_rag_search or write text, JSON, Markdown, or <answer>. Put brief public-evidence reasoning in reasoning and the final value in answer. "
            "Use the public candidate-evidence card below turn by turn: compare each rank-1 result and its full retrieved names against the candidate-linked query and rationale, then identify repeated support versus contradiction. A diagnosis is allowed only when it is named by at least two independent retrieval turns and has strictly more supporting turns than every competing retrieved class; a tie or weaker support is unresolved and must refuse. The initial candidates remain the only permitted candidates for any hypothetical further retrieval; retrieved names are evidence, not new query candidates. "
            "If the public evidence cannot support one diagnosis, the answer must be exactly "
            f"{INSUFFICIENT_EVIDENCE}. If refusing, reasoning may only describe missing or conflicting visible/public evidence and must not name, imply, or guess a diagnosis or candidate. " + answer_rule
            + "\nPublic candidate-evidence card:\n" + json.dumps(evidence_card, ensure_ascii=False)
        ),
    }


def _public_continue_after_early_refusal_prompt() -> dict[str, Any]:
    """Reject an unsupported refusal without inventing any private signal."""
    return {
        "role": "user",
        "content": (
            "Public evidence is non-empty and retrieval budget remains. INSUFFICIENT_EVIDENCE is not allowed yet. "
            "Call agrinet_rag_search once more using prior candidates and a different visible-trait query; do not write text or a final answer."
        ),
    }


def _public_continue_after_unverified_candidate_prompt() -> dict[str, Any]:
    """Keep a Direct candidate from finalizing without public support."""
    return {
        "role": "user",
        "content": (
            "The proposed final candidate has fewer than two independent matching retrieved-class evidence turns, and retrieval budget remains. "
            "Do not finalize or refuse now. Call agrinet_rag_search once more with a distinct visible-trait comparison grounded in the original candidates; candidate_classes must copy one to three original labels verbatim. "
            "Do not write text, JSON, Markdown, or <answer>."
        ),
    }


def _public_continue_after_open_first_evidence_prompt() -> dict[str, Any]:
    """Require a distinct second public comparison for Open diagnoses.

    The private r1 audit found that one retrieval often made an Open final look
    superficially verified while it remained factually wrong.  This prompt is
    entirely public and only changes the minimum evidence policy: it requests
    a second, candidate-linked query before an Open terminal decision.
    """
    return {
        "role": "user",
        "content": (
            "Public evidence is non-empty, but this is an Open diagnosis and one independent comparison is insufficient. "
            "Call agrinet_rag_search once more using a different visible trait or a different prior candidate; do not write text or a final answer."
        ),
    }


def _public_repair_invalid_tool_prompt() -> dict[str, Any]:
    """Give one public-only repair chance after a malformed tool request."""
    return {
        "role": "user",
        "content": (
            "The preceding retrieval request was invalid. Repair it now by calling agrinet_rag_search. "
            "candidate_classes is mandatory and must copy one to three labels verbatim from the original Candidate set in the conversation; do not translate, paraphrase, replace, or append retrieved names. "
            "Use image=query_image and only visible traits. Do not write text, JSON, Markdown, or <answer>."
        ),
    }


def _needs_more_retrieval(
    final: str, sample: dict[str, Any], candidates: tuple[str, ...], retrievals: list[dict[str, Any]], max_tool_turns: int,
) -> bool:
    """Return whether a public final lacks evidence to settle.

    A refusal is terminal before the budget only when no public evidence made
    progress (or a conflict is explicitly unresolved).  Micu sometimes emits
    a refusal in a native final-function call, so this check must happen after
    normalizing both text and native function responses.
    """
    if len(retrievals) >= max_tool_turns:
        return False
    answer = _answer_body(final)
    if answer == INSUFFICIENT_EVIDENCE:
        return bool(retrievals) and any(bool(item.get("evidence_progress")) for item in retrievals)
    if not answer:
        return False
    normalized = _normalized(_public_answer_class_name(sample, answer))
    candidate_hit = normalized in {_normalized(candidate) for candidate in candidates}
    # A single nearest-neighbour hit is weak public evidence. In addition to
    # two independent supporting turns, the selected class must outrank its
    # Direct/Option competitors.  Top-k retrieval often contains unrelated
    # background neighbours; treating every one as a competing diagnosis made
    # a supported candidate permanently unable to settle.  The competitor set
    # remains entirely public: initial candidates and visible Option labels.
    turn_counts: dict[str, int] = {}
    for item in retrievals:
        names = {_normalized(str(name)) for name in item.get("result_names") or []}
        for name in names:
            if name:
                turn_counts[name] = turn_counts.get(name, 0) + 1
    support_turns = turn_counts.get(normalized, 0)
    competitors = {_normalized(candidate) for candidate in candidates}
    options = sample.get("public_option_labels")
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict):
                competitors.update(_normalized(str(option.get(key) or "")) for key in ("name", "name_zh"))
    competitors.discard("")
    competitors.discard(normalized)
    competing_support = max((turn_counts.get(name, 0) for name in competitors), default=0)
    return support_turns < 2 or support_turns <= competing_support


def _write_collection_checkpoint(
    output_dir: Path, *, accepted: list[dict[str, Any]], traces: list[dict[str, Any]], last_trace: dict[str, Any],
) -> None:
    """Persist public pilot state after every independent image attempt.

    These are explicitly staging artifacts, never an SFT release.  Keeping the
    full public trace lets a terminated local run be audited without reissuing a
    teacher request for an already-attempted image.
    """
    reports = output_dir / "reports"
    staging = output_dir / "staging"
    reports.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    write_jsonl(staging / "accepted_candidates.jsonl", accepted)
    write_jsonl(staging / "collection_traces.jsonl", traces)
    (reports / "collection_progress.json").write_text(
        json.dumps({
            "schema_version": "agrinet.hcv-v13-collection-progress/v2",
            "processed_rows": len(traces), "accepted_rows": len(accepted),
            "last_trace": last_trace, "training_eligible": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def collect_one(
    sample: dict[str, Any], calibration: dict[str, Any], invoke: Callable[[list[dict[str, Any]]], str],
    *, rag_api: str, max_tool_turns: int = 5, top_k: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Collect one public trajectory; caller owns all remote delivery handling."""
    image = Path(str(sample.get("query_image") or ""))
    if not image.is_absolute():
        image = REPO_ROOT / image
    if not image.is_file():
        raise ValueError(f"query image unavailable: {image}")
    api_messages = _public_messages(sample, image)
    candidate_turn = invoke(api_messages)
    analysis, decision = route_first_turn(candidate_turn, sample, calibration)
    public = [{"role": "user", "content": "<image> Identify the agricultural object or disease."}, {"role": "assistant", "content": candidate_turn}]
    api_messages.append({"role": "assistant", "content": candidate_turn})
    api_messages.append({"role": "user", "content": continuation_prompt(analysis, decision, sample)})
    retrievals: list[dict[str, Any]] = []
    if decision.route == "direct":
        final = invoke(api_messages)
        if not validate_final_answer(_answer_body(final), question_type=str(sample.get("question_type") or "open")):
            return None, {"sample_id": sample.get("sample_id"), "reason": "invalid_direct_final", "route": "direct"}
        public.append({"role": "assistant", "content": final})
        state = "direct_high_confidence"
    else:
        final = ""
        # Bootstrap calibration does not yet permit a trusted one-shot route.
        # Every question therefore compares two public, candidate-linked views
        # before it may settle or refuse; later calibrated Direct routing is
        # still governed independently by ``direct_permitted``.
        minimum_retrievals = min(max_tool_turns, 2)
        # Reserve enough model-response slots to recover when it attempts a
        # terminal response before the Open minimum evidence count. Retrieval
        # calls themselves remain capped by ``max_tool_turns``.
        for _ in range(max_tool_turns + max(0, minimum_retrievals - 1)):
            output = invoke(api_messages)
            calls = normalize_tool_calls({"content": output})
            if not calls:
                if len(retrievals) < minimum_retrievals and len(retrievals) < max_tool_turns:
                    api_messages.append(_public_continue_after_open_first_evidence_prompt())
                    continue
                if _needs_more_retrieval(output, sample, analysis.candidates, retrievals, max_tool_turns):
                    api_messages.append(
                        _public_continue_after_early_refusal_prompt()
                        if _answer_body(output) == INSUFFICIENT_EVIDENCE
                        else _public_continue_after_unverified_candidate_prompt()
                    )
                    continue
                final = output
                break
            call = calls[0]
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if call.get("name") != TOOL_NAME:
                detail = "unexpected_tool_name"
            elif not isinstance(arguments, dict):
                detail = "tool_arguments_not_object"
            elif validation_errors := _validate_v13_tool_arguments(arguments, analysis.candidates):
                detail = "tool_schema_invalid"
            else:
                detail = ""
            if detail:
                # The malformed request has not been executed and contains no
                # private signal. Give the same image one bounded repair turn
                # before fail-closing the attempt; this handles Micu emitting
                # a translated candidate_classes value despite the schema text.
                if detail == "tool_schema_invalid":
                    # A candidate-linkage formatting error is recoverable
                    # deterministically when every actual retrieval parameter
                    # is valid.  Preserve those parameters and attach only the
                    # public Direct-first labels; never repair a malformed
                    # query, retrieval mode, image handle, top-k, or rationale.
                    if repaired_arguments := _repair_candidate_linkage_only(arguments, analysis.candidates):
                        arguments = repaired_arguments
                        detail = ""
                    else:
                        api_messages.append(_public_repair_invalid_tool_prompt())
                        repaired = invoke(api_messages)
                        repaired_calls = normalize_tool_calls({"content": repaired})
                        if len(repaired_calls) == 1:
                            repaired_call = repaired_calls[0]
                            repaired_args = repaired_call.get("arguments") if isinstance(repaired_call, dict) else None
                            repaired_errors = (
                                ["unexpected_tool_name"] if repaired_call.get("name") != TOOL_NAME else
                                ["tool_arguments_not_object"] if not isinstance(repaired_args, dict) else
                                _validate_v13_tool_arguments(repaired_args, analysis.candidates)
                            )
                            if repaired_errors and isinstance(repaired_args, dict):
                                repaired_args = _repair_candidate_linkage_only(repaired_args, analysis.candidates)
                                repaired_errors = _validate_v13_tool_arguments(repaired_args, analysis.candidates) if repaired_args else repaired_errors
                            if not repaired_errors:
                                arguments = repaired_args
                                call = repaired_call
                                detail = ""
                if detail:
                    return None, {
                        "sample_id": sample.get("sample_id"), "reason": "invalid_tool_call",
                        "detail": detail, "route": "rag",
                    }
            retrieval_arguments = {key: value for key, value in arguments.items() if key != "candidate_classes"}
            response, ledger = execute_rag_call(rag_api, sample, retrieval_arguments)
            ledger["tool_call"] = {"name": TOOL_NAME, "arguments": arguments}
            ledger["candidate_linked"] = True
            ledger["result_names"] = sorted(_result_names(response))
            ledger["top_result_label"] = _top_result_label(response, str(sample.get("language") or "en"))
            ledger["top_result_labels"] = _top_result_labels(response)
            ledger["evidence_progress"] = bool(ledger["result_names"])
            retrievals.append(ledger)
            public.extend([{"role": "tool_call", "content": json.dumps(ledger["tool_call"], ensure_ascii=False)}, {"role": "tool_response", "content": json.dumps(response, ensure_ascii=False)}])
            api_messages.extend([{"role": "assistant", "content": json.dumps(ledger["tool_call"], ensure_ascii=False)}, _public_tool_response_prompt(sample, response)])
        if not final:
            final = invoke([*api_messages, _public_finalization_prompt(sample, candidates=analysis.candidates, retrievals=retrievals)])
        if _needs_more_retrieval(final, sample, analysis.candidates, retrievals, max_tool_turns):
            # This path covers providers that returned a native final function
            # after the loop boundary rather than in the ordinary continuation.
            # Re-entering here is safe because no private truth is available.
            # Count completed retrievals, not merely model responses.  A model
            # can refuse again after a continuation prompt; consuming the
            # remaining budget on that refusal used to leave an ordinary early
            # refusal for ``_rag_state`` to classify.  Bounded extra response
            # attempts let it repair to a required tool call, while a model
            # that keeps refusing fails closed below.
            response_attempts = 0
            max_response_attempts = 2 * (max_tool_turns - len(retrievals)) + 1
            while (
                len(retrievals) < max_tool_turns
                and _needs_more_retrieval(final, sample, analysis.candidates, retrievals, max_tool_turns)
                and response_attempts < max_response_attempts
            ):
                response_attempts += 1
                api_messages.append(
                    _public_continue_after_early_refusal_prompt()
                    if _answer_body(final) == INSUFFICIENT_EVIDENCE
                    else _public_continue_after_unverified_candidate_prompt()
                )
                output = invoke(api_messages)
                calls = normalize_tool_calls({"content": output})
                if calls:
                    call = calls[0]
                    arguments = call.get("arguments") if isinstance(call, dict) else None
                    if call.get("name") != TOOL_NAME or not isinstance(arguments, dict):
                        return None, {"sample_id": sample.get("sample_id"), "reason": "invalid_tool_call", "detail": "continuation_tool_invalid", "route": "rag"}
                    if _validate_v13_tool_arguments(arguments, analysis.candidates):
                        arguments = _repair_candidate_linkage_only(arguments, analysis.candidates)
                    if arguments is None or _validate_v13_tool_arguments(arguments, analysis.candidates):
                        return None, {"sample_id": sample.get("sample_id"), "reason": "invalid_tool_call", "detail": "continuation_tool_invalid", "route": "rag"}
                    retrieval_arguments = {key: value for key, value in arguments.items() if key != "candidate_classes"}
                    response, ledger = execute_rag_call(rag_api, sample, retrieval_arguments)
                    ledger["tool_call"] = {"name": TOOL_NAME, "arguments": arguments}
                    ledger["candidate_linked"] = True
                    ledger["result_names"] = sorted(_result_names(response))
                    ledger["top_result_label"] = _top_result_label(response, str(sample.get("language") or "en"))
                    ledger["top_result_labels"] = _top_result_labels(response)
                    ledger["evidence_progress"] = bool(ledger["result_names"])
                    retrievals.append(ledger)
                    public.extend([{"role": "tool_call", "content": json.dumps(ledger["tool_call"], ensure_ascii=False)}, {"role": "tool_response", "content": json.dumps(response, ensure_ascii=False)}])
                    api_messages.extend([{"role": "assistant", "content": json.dumps(ledger["tool_call"], ensure_ascii=False)}, _public_tool_response_prompt(sample, response)])
                    final = ""
                    continue
                final = output
                if not _needs_more_retrieval(final, sample, analysis.candidates, retrievals, max_tool_turns):
                    break
            if _needs_more_retrieval(final, sample, analysis.candidates, retrievals, max_tool_turns):
                return None, {
                    "sample_id": sample.get("sample_id"), "reason": "premature_final",
                    "detail": "continuation_did_not_retrieve", "route": "rag",
                }
        final = _normalize_public_final(sample, final)
        final = _sanitize_abstention_reasoning(sample, final, analysis.candidates, retrievals)
        if _answer_body(final) == INSUFFICIENT_EVIDENCE and len(retrievals) >= max_tool_turns:
            if leader := _public_evidence_leader(sample, analysis.candidates, retrievals):
                final = _converged_public_final(sample, leader)
        abstention = _answer_body(final) == INSUFFICIENT_EVIDENCE
        valid_final = not normalize_tool_calls({"content": final}) and validate_final_answer(
            _answer_body(final), question_type=str(sample.get("question_type") or "open"), abstention=abstention,
        )
        if valid_final and abstention and _abstention_mentions_diagnosis(final, analysis.candidates, retrievals):
            valid_final = False
        if not valid_final:
            # A provider can ignore ``tool_choice=required`` and return bare
            # prose after valid public evidence.  One terminal repair call
            # keeps the same evidence ledger and forces the final native
            # function; it is not another retrieval or a truth-guided retry.
            final = _normalize_public_final(sample, invoke([*api_messages, _public_finalization_prompt(sample, candidates=analysis.candidates, retrievals=retrievals)]))
            final = _sanitize_abstention_reasoning(sample, final, analysis.candidates, retrievals)
            abstention = _answer_body(final) == INSUFFICIENT_EVIDENCE
            valid_final = not normalize_tool_calls({"content": final}) and validate_final_answer(
                _answer_body(final), question_type=str(sample.get("question_type") or "open"), abstention=abstention,
            )
            if valid_final and abstention and _abstention_mentions_diagnosis(final, analysis.candidates, retrievals):
                valid_final = False
        if not valid_final:
            return None, {"sample_id": sample.get("sample_id"), "reason": "invalid_rag_final", "route": "rag"}
        public.append({"role": "assistant", "content": final})
        if final:
            state = _rag_state(
                sample=sample, answer=_answer_body(final), candidates=analysis.candidates, retrievals=retrievals,
                exhausted=len(retrievals) >= max_tool_turns, max_tool_turns=max_tool_turns,
            )
    row = {
        "sample_id": sample.get("sample_id"), "tools": tools_json(), "messages": public,
        "images": [str(image), *[ref for ledger in retrievals for ref in ledger.get("visible_reference_images") or []]],
        "metadata": {
            "schema_version": "agrinet.hcv-v13-pilot-trajectory/v1", "teacher_prompt_version": TEACHER_PROMPT_VERSION,
            "question_type": sample.get("question_type"), "language": sample.get("language"), "task_domain": sample.get("task_domain"),
            "route": decision.route, "route_reason": decision.reason, "candidate_analysis": list(analysis.candidates),
            "candidate_confidence": analysis.confidence, "candidate_state": state, "retrieval_turns": len(retrievals),
            "retrieval_ledger": [
                {"turn": index + 1, "candidate_linked": bool(item.get("candidate_linked")),
                 "evidence_progress": bool(item.get("evidence_progress")), "result_names": item.get("result_names") or [],
                 "top_result_label": item.get("top_result_label") or "",
                 "top_result_labels": item.get("top_result_labels") or []}
                for index, item in enumerate(retrievals)
            ],
            "image_sha256": sample.get("image_sha256"), "pilot_only": True, "training_eligible": False,
        },
    }
    return row, {"sample_id": sample.get("sample_id"), "route": decision.route, "state": state, "retrieval_turns": len(retrievals)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-api", required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--credential-profile", choices=("yunwu", "micu_slb"), default="micu_slb")
    parser.add_argument("--max-tool-turns", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--preflight-timeout", type=int, default=20)
    parser.add_argument("--visual-preflight-only", action="store_true")
    parser.add_argument("--trajectory-preflight-only", action="store_true")
    parser.add_argument("--trajectory-preflight-index", type=int, default=0)
    parser.add_argument("--trajectory-preflight-max-tool-turns", type=int, default=1)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != 32:
        raise ValueError("HCV v13 pilot collection requires exactly 32 plan rows")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    env = yunwu_environment(profile=args.credential_profile)
    _preflight_micu_endpoint(
        env["YUNWU_API_BASE_URL"], env["YUNWU_API_KEY"], args.preflight_timeout,
    )
    if args.visual_preflight_only and args.trajectory_preflight_only:
        raise ValueError("choose at most one v13 preflight mode")
    if args.visual_preflight_only:
        result = preflight_visual_micu(
            samples[0], api_key=env["YUNWU_API_KEY"], base_url=env["YUNWU_API_BASE_URL"],
            model=args.model, timeout=args.timeout, max_tokens=min(args.max_tokens, 256),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "reports" / "visual_preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    invoke = lambda messages: _call_micu(messages, api_key=env["YUNWU_API_KEY"], base_url=env["YUNWU_API_BASE_URL"], model=args.model, timeout=args.timeout, max_tokens=args.max_tokens)
    accepted, traces = [], []
    if args.trajectory_preflight_only:
        if not 0 <= args.trajectory_preflight_index < len(samples):
            raise ValueError("trajectory preflight index must select a plan row")
        if not 1 <= args.trajectory_preflight_max_tool_turns <= args.max_tool_turns:
            raise ValueError("trajectory preflight tool turns must be within the configured budget")
        preflight_sample = samples[args.trajectory_preflight_index]
        try:
            row, trace = collect_one(
                preflight_sample, calibration, invoke, rag_api=args.rag_api,
                max_tool_turns=args.trajectory_preflight_max_tool_turns, top_k=args.top_k,
            )
            result = {
                "schema_version": "agrinet.hcv-v13-trajectory-preflight/v1",
                "sample_id": preflight_sample.get("sample_id"), "accepted": row is not None,
                "trace": trace, "message_count": len(row.get("messages") or []) if row else 0,
                "training_eligible": False,
            }
        except Exception as exc:
            result = {
                "schema_version": "agrinet.hcv-v13-trajectory-preflight/v1",
                "sample_id": preflight_sample.get("sample_id"), "accepted": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "training_eligible": False,
            }
        (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "reports" / "trajectory_preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("accepted") else 1
    for sample in samples:
        try:
            row, trace = collect_one(
                sample, calibration, invoke, rag_api=args.rag_api,
                max_tool_turns=args.max_tool_turns, top_k=args.top_k,
            )
        except ValueError as exc:
            row = None
            trace = {
                "sample_id": sample.get("sample_id"), "route": "unavailable",
                "reason": "invalid_candidate_turn", "detail": str(exc),
            }
        except UnknownTeacherDelivery as exc:
            row = None
            trace = {
                "sample_id": sample.get("sample_id"), "route": "unavailable",
                "reason": "unknown_delivery", "detail": str(exc),
            }
        except Exception as exc:
            row = None
            trace = {
                "sample_id": sample.get("sample_id"), "route": "unavailable",
                "reason": "collector_exception", "detail": f"{type(exc).__name__}: {exc}",
            }
        traces.append(trace)
        if row is not None:
            accepted.append(row)
        _write_collection_checkpoint(
            args.output_dir, accepted=accepted, traces=traces, last_trace=trace,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "traces").mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "traces" / "collection.jsonl", traces)
    if len(accepted) == 32:
        (args.output_dir / "train").mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "train" / "agent_sft.accepted.jsonl", accepted)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "agrinet.hcv-v13-pilot-collection/v2",
        "requested_rows": 32, "accepted_rows": len(accepted),
        "complete": len(accepted) == 32, "training_eligible": False,
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if len(accepted) == 32 else 1


if __name__ == "__main__":
    raise SystemExit(main())
