"""E3.5 public cascade controller.

This module intentionally has no implicit provider fallback: a live caller must
explicitly bind a teacher, isolated private auditor, and the typed local RAG
surface.  That keeps an offline audit or unit test from accidentally spending
teacher budget.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from agrinet.rag.e35_classifier_cascade import (DELIVERY_FAILURES, ROUTES,
    cascade_outcome, public_classifier_card, public_teacher_input,
    validate_rag_terminal)
from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.e35_budget import BudgetExhausted, BudgetObservedOverrun, E35TokenBudget, uncached_input_tokens
from agrinet.rag.e35_private import validate_parent_protocol
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget, image_url_content, parse_teacher_action
from agrinet.rag.e35_transport import transport_image

ToolRunner = Callable[[dict[str, Any]], dict[str, Any]]
AuditRunner = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]
RagRunner = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
_HCV_PROTOCOLS = {"agrinet.e39-hcv-cascade/v1", "agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}


class LocalRagToolFailure(RuntimeError):
    """A local RAG call failed after its teacher tool call was delivered."""

def _terminal_answer(content: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", content, flags=re.I | re.S)
    return match.group(1).strip() if match else content.strip()

_TOOLS = {
    "agrinet_classifier_predict": {"type": "object", "properties": {}, "additionalProperties": False},
    "agrinet_classifier_expand": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False},
    "agrinet_rag_search": {"type": "object", "properties": {"query": {"type": "string"}, "retrieval_type": {"enum": ["visual", "semantic"]}, "rationale": {"type": "string"}}, "required": ["query", "retrieval_type", "rationale"], "additionalProperties": False},
}

# Public, stable controller diagnostics.  These identify protocol failures
# without exposing private-audit prose, truth, fold, or witness designation.
_CONTRACT_ERRORS = {
    "invalid_response": "invalid_terminal_response",
    "route tool contract violation": "route_tool_contract",
    "private audit must close accept or reject": "private_audit_terminal",
    "private RAG witness must reject pre-RAG terminal": "private_route_coverage",
    "private RAG witness requires actual RAG call and response": "missing_rag_call_response",
    "private audit receipt already exists": "private_receipt_conflict",
    "tool action has no native assistant tool-call message": "native_tool_call_missing",
    "native tool-call identity changed unexpectedly": "native_tool_call_identity",
    "teacher must make exactly one tool call": "multiple_tool_calls",
    "E3.9 final must be legal Hermes": "hermes_final_format",
    "E3.9 HCV fields are missing or unordered": "hcv_fields",
    "E3.9 requires three visual observations": "hcv_visual_observations",
    "E3.9 uncertainty is incomplete": "hcv_uncertainty",
    "E3.9 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.9 Direct cannot call tools": "route_tool_contract",
    "E3.9 Direct Open requires two or three hypotheses": "hcv_hypotheses",
    "E3.9 Classifier tool order is invalid": "hcv_tool_order",
    "E3.9 Classifier cannot call RAG": "hcv_tool_order",
    "E3.9 RAG tool order is invalid": "hcv_tool_order",
    "E3.9 Option answer must be class name — letter": "option_terminal_format",
    "E3.9 pre-tool think is invalid": "hermes_pretool_format",
    "E3.10 final must be legal Hermes": "hermes_final_format",
    "E3.10 HCV fields are missing or unordered": "hcv_fields",
    "E3.10 requires three visual observations": "hcv_visual_observations",
    "E3.10 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.10 uncertainty is incomplete": "hcv_uncertainty",
    "E3.10 Direct Open requires two or three hypotheses": "hcv_hypotheses",
    "E3.10 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.10 Option answer must be class name — letter": "option_terminal_format",
    "E3.10 Direct cannot call tools": "route_tool_contract",
    "E3.10 Classifier tool order is invalid": "hcv_tool_order",
    "E3.10 RAG tool order is invalid": "hcv_tool_order",
    "E3.10 tool call lacks preceding Hermes planning": "hermes_pretool_format",
    "E3.10 native tool response is invalid": "native_tool_call_missing",
    "E3.10 native tool call must not mix planning text": "hermes_pretool_format",
    "E3.10 RAG search limit exceeded": "hcv_tool_order",
    "E3.11 final must be legal Hermes": "hermes_final_format",
    "E3.11 HCV fields are missing or unordered": "hcv_fields",
    "E3.11 requires three visual observations": "hcv_visual_observations",
    "E3.11 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.11 uncertainty is incomplete": "hcv_uncertainty",
    "E3.11 Option answer must be class name — letter": "option_terminal_format",
    "E3.11 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.11 Direct cannot call tools": "route_tool_contract",
    "E3.11 Classifier tool order is invalid": "hcv_tool_order",
    "E3.11 RAG tool order is invalid": "hcv_tool_order",
    "E3.11 tool call lacks preceding Hermes planning": "hermes_pretool_format",
    "E3.11 native tool response is invalid": "native_tool_call_missing",
    "E3.12 final must be legal Hermes": "hermes_final_format",
    "E3.12 HCV fields are missing or unordered": "hcv_fields",
    "E3.12 requires three visual observations": "hcv_visual_observations",
    "E3.12 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.12 uncertainty is incomplete": "hcv_uncertainty",
    "E3.12 Option answer must be class name — letter": "option_terminal_format",
    "E3.12 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.12 Direct cannot call tools": "route_tool_contract",
    "E3.12 Classifier tool order is invalid": "hcv_tool_order",
    "E3.12 RAG tool order is invalid": "hcv_tool_order",
    "E3.12 tool call lacks Hermes planning": "hermes_pretool_format",
    "E3.12 native tool response is invalid": "native_tool_call_missing",
    "E3.13 final must be legal Hermes": "hermes_final_format",
    "E3.13 HCV fields are missing or unordered": "hcv_fields",
    "E3.13 requires three visual observations": "hcv_visual_observations",
    "E3.13 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.13 uncertainty is incomplete": "hcv_uncertainty",
    "E3.13 Option answer must be class name — letter": "option_terminal_format",
    "E3.13 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.13 Direct cannot call tools": "route_tool_contract",
    "E3.13 Classifier tool order is invalid": "hcv_tool_order",
    "E3.13 RAG tool order is invalid": "hcv_tool_order",
    "E3.13 tool call lacks Hermes planning": "hermes_pretool_format",
    "E3.13 native tool response is invalid": "native_tool_call_missing",
    "E3.14 final must be legal Hermes": "hermes_final_format",
    "E3.14 HCV fields are missing or unordered": "hcv_fields",
    "E3.14 requires three visual observations": "hcv_visual_observations",
    "E3.14 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.14 uncertainty is incomplete": "hcv_uncertainty",
    "E3.14 Option answer must be class name — letter": "option_terminal_format",
    "E3.14 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.14 Direct cannot call tools": "route_tool_contract",
    "E3.14 Classifier tool order is invalid": "hcv_tool_order",
    "E3.14 RAG tool order is invalid": "hcv_tool_order",
    "E3.14 tool call lacks Hermes planning": "hermes_pretool_format",
    "E3.14 native tool response is invalid": "native_tool_call_missing",
    "E3.15 final must be legal Hermes": "hermes_final_format",
    "E3.15 HCV fields are missing or unordered": "hcv_fields",
    "E3.15 requires three visual observations": "hcv_visual_observations",
    "E3.15 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.15 uncertainty is incomplete": "hcv_uncertainty",
    "E3.15 Option answer must be class name — letter": "option_terminal_format",
    "E3.15 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.15 Direct cannot call tools": "route_tool_contract",
    "E3.15 Classifier tool order is invalid": "hcv_tool_order",
    "E3.15 RAG tool order is invalid": "hcv_tool_order",
    "E3.15 tool call lacks Hermes planning": "hermes_pretool_format",
    "E3.15 native tool response is invalid": "native_tool_call_missing",
    "E3.16 final must be legal Hermes": "hermes_final_format",
    "E3.16 HCV fields are missing or unordered": "hcv_fields",
    "E3.16 requires three visual observations": "hcv_visual_observations",
    "E3.16 requires two rejected alternatives": "hcv_rejected_alternatives",
    "E3.16 uncertainty is incomplete": "hcv_uncertainty",
    "E3.16 Option answer must be class name — letter": "option_terminal_format",
    "E3.16 Option reasoning must compare every public option": "hcv_option_coverage",
    "E3.16 Direct cannot call tools": "route_tool_contract",
    "E3.16 Classifier tool order is invalid": "hcv_tool_order",
    "E3.16 RAG tool order is invalid": "hcv_tool_order",
    "E3.16 tool call lacks Hermes planning": "hermes_pretool_format",
    "E3.16 native tool response is invalid": "native_tool_call_missing",
    "E3.16 RAG requires an actual search response": "missing_rag_call_response",
    "E3.16 RAG retrieval_type is required": "route_tool_contract",
    "E3.16 RAG nearest alternative is missing": "rag_nearest_alternative",
    "E3.16 RAG evidence linkage is incomplete": "rag_evidence_linkage",
    "E3.16 RAG nearest alternative is not rejected": "rag_nearest_alternative",
    "E3.16 abstention lacks evidence limitation": "rag_abstention_evidence",
    "E3.19 abstention lacks missing visible trait": "rag_abstention_missing_visible_trait",
    "E3.19 abstention lacks matching RAG limitation": "rag_abstention_rag_limitation",
    "E3.19 final must be legal Hermes": "hermes_final_format",
}


def _contract_error_code(exc: ValueError) -> str | None:
    """Project delivered controller/validator errors into stable public codes.

    E3.17 wraps the E3.16 discriminative validator, so its detailed messages
    are intentionally version-prefixed rather than duplicated in the global
    table.  They occur after a durable teacher response and must never become
    delivery ambiguity eligible for replay.
    """
    message=str(exc)
    if message.startswith("E3.19 "):
        # E3.19 inherits the E3.16 discriminative base but adds explicit
        # closure/refusal diagnostics.  Preserve the granular public code
        # rather than collapsing a delivered validator failure to generic.
        return _CONTRACT_ERRORS.get(message) or _CONTRACT_ERRORS.get(message.replace("E3.19", "E3.16", 1)) or "e319_rag_contract"
    return _CONTRACT_ERRORS.get(message) or ("e317_rag_contract" if message.startswith("E3.17 ") else None)

def _validate_tool_arguments(name: str, arguments: Any, *, predicted: bool) -> None:
    if name not in _TOOLS or not isinstance(arguments, dict):
        raise ValueError("route tool contract violation")
    schema = _TOOLS[name]
    fields, required = set(schema.get("properties", {})), set(schema.get("required", []))
    if set(arguments) - fields or not required.issubset(arguments):
        raise ValueError("route tool contract violation")
    if name == "agrinet_classifier_expand" and not predicted:
        raise ValueError("route tool contract violation")
    if name == "agrinet_classifier_expand" and not isinstance(arguments.get("reason"), str):
        raise ValueError("route tool contract violation")
    if name == "agrinet_rag_search":
        if arguments.get("retrieval_type") not in {"visual", "semantic"} or any(not isinstance(arguments.get(key), str) or not arguments[key].strip() for key in ("query", "rationale")):
            raise ValueError("route tool contract violation")

def _request(row: dict[str, Any], route: str, messages: list[dict[str, Any]], model: str, max_tokens: int, *, rag_called: bool = False) -> dict[str, Any]:
    public = public_teacher_input(row, route)
    allowed = public.get("tools", [])
    payload: dict[str, Any] = {"model": model, "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "messages": messages}
    if allowed:
        payload["tools"] = [{"type": "function", "function": {"name": name, "description": name, "parameters": _TOOLS[name]}} for name in allowed]
        # E3.9 forced tool_choice caused providers to emit empty assistant
        # content, contradicting its own Hermes pre-tool contract. E3.10 uses
        # explicit stage prompts plus the controller sequence validator instead.
        force_first = ("agrinet_classifier_predict" if row.get("e39_protocol") == "agrinet.e39-hcv-cascade/v1" and route in {"classifier", "rag"} else ("agrinet_rag_search" if route == "rag" else None))
        if row.get("e39_protocol") in {"agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}:
            from agrinet.rag.hermes_protocol import is_pre_tool_think
            planned = bool(messages and messages[-1].get("role") == "assistant" and is_pre_tool_think(messages[-1].get("content")))
            prior_tool = any(message.get("role") == "tool" for message in messages)
            native_names = [str(call.get("function", {}).get("name")) for message in messages
                            for call in (message.get("tool_calls") or []) if isinstance(call, dict)]
            if planned and route == "classifier":
                force_first = ("agrinet_classifier_predict" if not prior_tool
                               else ("agrinet_classifier_expand" if native_names.count("agrinet_classifier_expand") == 0 else None))
            elif planned and route == "rag":
                # Every RAG retrieval is preceded by a standalone Hermes plan.
                # In particular, do not turn a completed first retrieval into
                # auto mode: the model may legitimately plan a second/third
                # discriminative query.
                force_first = ("agrinet_classifier_predict" if not prior_tool else
                               ("agrinet_rag_search" if native_names.count("agrinet_rag_search") < 3 else None))
            else:
                force_first = None
            # After a native tool response, a fresh E3.10 response must be a
            # final Hermes answer or standalone plan; it cannot immediately
            # call another tool without exposing that plan.
            payload["tool_choice"] = ({"type": "function", "function": {"name": force_first}}
                                      if force_first else ("none" if prior_tool else "auto"))
        else:
            payload["tool_choice"] = ({"type": "function", "function": {"name": force_first}}
                                      if force_first and not (rag_called or (route == "classifier" and any(m.get("role") == "tool" for m in messages))) else "auto")
    return payload

def _initial_messages(row: dict[str, Any], route: str, *, transport_max_side: int = 0) -> list[dict[str, Any]]:
    public = public_teacher_input(row, route)
    text = public["question"]
    if "options" in public:
        text += "\n" + "\n".join(f"{item['label']}. {item['name']}" for item in public["options"])
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("frozen source row lacks its image binding")
    # The filesystem path is used only to attach the image bytes; it is never
    # serialized as public prompt text or durable public output.
    image = transport_image(Path(image_path), max_side=transport_max_side)[0] if transport_max_side else image_url_content(Path(image_path))
    return [{"role": "system", "content": public["system"]}, {"role": "user", "content": [{"type": "text", "text": text}, image]}]

def _ledger_payload(payload: dict[str, Any], *, sample_id: str, route: str, turn: int) -> dict[str, Any]:
    """Persist only public text/tool metadata, never image bytes or local paths."""
    messages = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            content = [{"type": item.get("type"), "image_sha256": "bound"} if isinstance(item, dict) and item.get("type") == "image_url" else item for item in content]
        # Native tool exchanges must be retained for an auditable continuation,
        # but keep the same image/path exclusion as ordinary prompt messages.
        projected = {"role": message.get("role"), "content": content}
        if "tool_calls" in message:
            projected["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            projected["tool_call_id"] = message["tool_call_id"]
        messages.append(projected)
    return {"operation": "generation", "sample_id": sample_id, "route": route, "turn": turn,
            "messages": messages, "tools": [item.get("function", {}).get("name") for item in payload.get("tools", [])]}


def _append_native_tool_exchange(messages: list[dict[str, Any]], raw: dict[str, Any],
                                 action: dict[str, Any], result: dict[str, Any], *,
                                 split_pretool_think: bool = False) -> None:
    """Append an OpenAI-compatible assistant tool call and its matched result.

    Tool-call IDs are provider protocol state, not display text.  Rewriting the
    call as JSON in an assistant message breaks the next native continuation.
    """
    try:
        message = raw["choices"][0]["message"]
        calls = message["tool_calls"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("tool action has no native assistant tool-call message") from exc
    if (not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict)
            or calls[0].get("id") != action.get("tool_call_id")):
        raise ValueError("native tool-call identity changed unexpectedly")
    content = message.get("content")
    if split_pretool_think:
        from agrinet.rag.hermes_protocol import is_pre_tool_think
        if not is_pre_tool_think(content):
            raise ValueError("E3.12 tool call lacks Hermes planning")
        # This is an exact normalization of provider wire framing, not a
        # synthesized thought: preserve the provider's text verbatim and its
        # original native call identity in adjacent messages.
        messages.append({"role": "assistant", "content": content})
        content = None
    messages.append({"role": "assistant", "content": content,
                     "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": action["tool_call_id"],
                     "content": json.dumps(result, ensure_ascii=False)})

def _e39_pretool_contract(raw: dict[str, Any]) -> None:
    """E3.9 teaches explicit Hermes planning before every real tool action."""
    try:
        content = raw["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("E3.9 pre-tool think is invalid") from exc
    from agrinet.rag.hermes_protocol import is_pre_tool_think
    if not isinstance(content, str) or not content.strip() or not is_pre_tool_think(content):
        raise ValueError("E3.9 pre-tool think is invalid")

def _e310_tool_turn_contract(messages: list[dict[str, Any]], raw: dict[str, Any]) -> None:
    """E3.10 puts Hermes planning on the preceding assistant turn."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("E3.10 tool call lacks preceding Hermes planning")
    from agrinet.rag.hermes_protocol import is_pre_tool_think
    if not is_pre_tool_think(messages[-1].get("content")):
        raise ValueError("E3.10 tool call lacks preceding Hermes planning")
    try:
        content = raw["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("E3.10 native tool response is invalid") from exc
    if content not in (None, ""):
        raise ValueError("E3.10 native tool call must not mix planning text")

def _e312_tool_turn_contract(messages: list[dict[str, Any]], raw: dict[str, Any]) -> bool:
    """Accept only an existing pure Hermes plan, standalone or wire-combined."""
    from agrinet.rag.hermes_protocol import is_pre_tool_think
    try:
        content = raw["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("E3.12 native tool response is invalid") from exc
    if content in (None, ""):
        if not messages or not is_pre_tool_think(messages[-1].get("content")):
            raise ValueError("E3.12 tool call lacks Hermes planning")
        return False
    if is_pre_tool_think(content):
        return True
    raise ValueError("E3.12 tool call lacks Hermes planning")

def _e310_next_step_message(*, route: str, tool_name: str) -> str | None:
    if route == "rag" and tool_name == "agrinet_classifier_predict":
        return ("Use the actual public classifier card now returned. Do not answer and do not call a tool in this turn. "
                "Output only one nonempty <think>...</think> planning turn that names the unresolved candidate difference and the discriminative RAG query you will make next.")
    return None


def _write_private_audit_record(*, root: Path, row: dict[str, Any], context: dict[str, Any],
                                audit: dict[str, Any]) -> None:
    """Persist a non-secret private-audit receipt only after ledger delivery.

    The provider response is already durable at this point. A filesystem problem
    must therefore be a controller-quality failure, never an ambiguous provider
    delivery that becomes eligible for recovery.
    """
    work_id, route = context["work_id"], context["route"]
    receipt = {"sample_id": row["sample_id"], "work_id": work_id,
               "attempt_ordinal": context["attempt_ordinal"], "route": route,
               "decision": audit["decision"],
               "requires_rag": bool(audit.get("requires_rag")),
               "rag_evidence_insufficient": bool(audit.get("rag_evidence_insufficient")),
               "private_registry_sha256": audit.get("private_registry_sha256"),
               "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    target = root / "private" / work_id.replace(":", "_") / route / "audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("private audit receipt already exists") from exc
        comparable = ("sample_id", "work_id", "attempt_ordinal", "route", "decision",
                      "requires_rag", "rag_evidence_insufficient", "private_registry_sha256")
        if all(existing.get(key) == receipt.get(key) for key in comparable):
            return
        raise ValueError("private audit receipt already exists")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")

def collect_round(*, manifest: Path, source: Path, output: Path, output_root: Path,
                  teacher: ToolRunner, private_audit: AuditRunner, rag_search: RagRunner,
                  model: str = "gpt-5.6-terra", max_tokens: int = 4096) -> dict[str, Any]:
    """Execute one immutable manifest with at most its declared four workers.

    This function neither retries nor creates a replenishment manifest. A
    completed output is exclusive (`x` mode), which prevents accidental replay.
    """
    if output.exists(): raise ValueError("E3.5 round output is immutable and already exists")
    plan = json.loads(Path(manifest).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = {row.get("sample_id"): row for row in rows}
    items = plan.get("work_items")
    if plan.get("schema_version") not in {"agrinet.e35-cascade-manifest/v1", "agrinet.e35-cascade-manifest/v2", "agrinet.e39-hcv-cascade-manifest/v1", "agrinet.e310-hcv-cascade-manifest/v1", "agrinet.e311-hcv-cascade-manifest/v1", "agrinet.e312-hcv-cascade-manifest/v1", "agrinet.e313-hcv-cascade-manifest/v1", "agrinet.e314-hcv-cascade-manifest/v1", "agrinet.e315-option-format-repair-manifest/v1", "agrinet.e316-rag-discriminator-manifest/v1", "agrinet.e316-rag-discriminator-canary-manifest/v1", "agrinet.e317-all-unknown-rag-audit-manifest/v1", "agrinet.e318-all-unknown-512-rag-audit-manifest/v1", "agrinet.e319-rag-closure-audit-manifest/v1"} or not isinstance(items, list):
        raise ValueError("invalid E3.5 immutable collection manifest")
    if len(source_rows) != len(rows) or any(item.get("sample_id") not in source_rows for item in items):
        raise ValueError("manifest work item lacks a unique frozen source row")
    workers = int(plan.get("workers", 0))
    if workers != 4 or plan.get("automatic_replay_allowed") is not False: raise ValueError("E3.5 worker/replay contract mismatch")
    def one(item: dict[str, Any]) -> dict[str, Any]:
        return run_work_item(work_item=item, row=source_rows[item["sample_id"]], output_root=output_root, teacher=teacher,
                             private_audit=private_audit, rag_search=rag_search, model=model, max_tokens=max_tokens,
                             intent_limit=int(plan.get("micu_intent_limit", 8000)),
                             controls=dict(plan.get("collection_controls") or {}))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(one, items))
    payload = {"schema_version": "agrinet.e35-cascade-round-outcomes/v1", "campaign_id": plan.get("campaign_id"),
               "round": plan.get("round"), "manifest_sha256": __import__("hashlib").file_digest(Path(manifest).open("rb"), "sha256").hexdigest(),
               "outcomes": outcomes, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload

def run_work_item(*, work_item: dict[str, Any], row: dict[str, Any], output_root: Path,
                  teacher: ToolRunner, private_audit: AuditRunner, rag_search: RagRunner,
                  model: str = "gpt-5.6-terra", max_tokens: int = 4096, intent_limit: int = 8000,
                  controls: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Direct -> Classifier -> RAG once, stopping on any delivery ambiguity."""
    if work_item.get("sample_id") != row.get("sample_id") or work_item.get("round") not in {"R0", "R1", "R2"}:
        raise ValueError("work item is not bound to its frozen source row")
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    budget = GlobalMicuBudget(root / "global_micu_intents.jsonl", limit=intent_limit)
    ledger = E35Ledger(root / "ledgers" / str(work_item["work_id"]).replace(":", "_"),
                       work_id=str(work_item["work_id"]), attempt_ordinal=int(work_item.get("attempt_ordinal", 0)), intent_limit=intent_limit)
    controls = controls or {}
    resume_route = str(work_item.get("resume_route") or "direct")
    if resume_route not in ROUTES:
        raise ValueError("E3.5 continuation has an invalid resume route")
    routes = ROUTES[ROUTES.index(resume_route):]
    transport_max_side = int(controls.get("transport_image_max_side", 0))
    max_public_turns = int(controls.get("max_public_turns_per_route", 8))
    reserve_by_kind = controls.get("reservation_uncached_tokens") or {}
    token_budget = None
    if controls:
        cap = int(controls.get("uncached_input_token_cap", 0))
        if cap <= 0 or transport_max_side <= 0 or max_public_turns <= 0:
            raise ValueError("E3.5 budget controls are incomplete")
        token_budget = E35TokenBudget(root / "token_budget.jsonl", uncached_input_token_cap=cap)
    stages: list[dict[str, Any]] = []
    for route in routes:
        messages = _initial_messages(row, route, transport_max_side=transport_max_side)
        trace: list[dict[str, Any]] = []
        predicted = False
        expand_calls = 0
        rag_called = False
        rag_calls = 0
        try:
            for turn in range(1, max_public_turns + 1):
                request = _request(row, route, messages, model, max_tokens, rag_called=rag_called)
                budget_key = f"{work_item['work_id']}:{route}:generation:{turn}"
                if token_budget is not None:
                    token_budget.reserve(budget_key, uncached_input_tokens=int(reserve_by_kind["generation"][route]), metadata={"route": route, "kind": "generation", "turn": turn})
                budget.reserve(f"{work_item['work_id']}:{route}:generation:{turn}")
                request_id, raw = ledger.call(kind="generation", key=f"{route}:generation:{turn}",
                    payload=_ledger_payload(request, sample_id=row["sample_id"], route=route, turn=turn),
                    invoke=lambda: teacher(request))
                if token_budget is not None:
                    observed = uncached_input_tokens(raw)
                    if observed is not None: token_budget.settle(budget_key, uncached_input_tokens=observed)
                action = parse_teacher_action(raw)
                if (row.get("e39_protocol") in {"agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} and action["type"] == "final"
                        and __import__("agrinet.rag.hermes_protocol", fromlist=["is_pre_tool_think"]).is_pre_tool_think(action["content"])):
                    # A planning turn is not a terminal answer. It is recorded
                    # verbatim, then the next native request may issue one tool.
                    messages.append({"role": "assistant", "content": action["content"]})
                    continue
                if action["type"] == "final":
                    if (row.get("e39_protocol") in {"agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} and route == "rag"
                            and not rag_called):
                        raise ValueError("E3.14 RAG tool order is invalid")
                    trajectory = {"route": route, "answer": action["content"], "tool_calls": [event["call"] for event in trace if "call" in event], "tool_trace": trace, "messages": messages + [{"role": "assistant", "content": action["content"]}]}
                    audit_budget_key = f"{work_item['work_id']}:{route}:private-audit:1"
                    if token_budget is not None:
                        token_budget.reserve(audit_budget_key, uncached_input_tokens=int(reserve_by_kind["private_audit"]), metadata={"route": route, "kind": "private_audit"})
                    budget.reserve(f"{work_item['work_id']}:{route}:private-audit:1")
                    audit_context = {"work_id": str(work_item["work_id"]),
                                     "attempt_ordinal": int(work_item.get("attempt_ordinal", 0)),
                                     "route": route}
                    audit_request_id, audit = ledger.call(kind="private_audit", key=f"{route}:private-audit",
                        payload={"operation": "private_parent_audit", "sample_id": row["sample_id"], "route": route,
                                 "parent_request_id": request_id}, invoke=lambda: private_audit(row, trajectory, audit_context))
                    if token_budget is not None and isinstance(audit.get("_provider_response"), dict):
                        observed = uncached_input_tokens(audit["_provider_response"])
                        if observed is not None: token_budget.settle(audit_budget_key, uncached_input_tokens=observed)
                    if audit.get("decision") not in {"accept", "reject"}: raise ValueError("private audit must close accept or reject")
                    _write_private_audit_record(root=root, row=row, context=audit_context, audit=audit)
                    validate_parent_protocol(row, trajectory, audit)
                    if _terminal_answer(action["content"]) == "INSUFFICIENT_EVIDENCE":
                        trajectory["private_rag_evidence_insufficient"] = bool(audit.get("rag_evidence_insufficient"))
                        validate_rag_terminal(trajectory)
                    # Persist the delivered public trajectory before controller
                    # validation.  A validation reject is quality evidence,
                    # not a reason to hide the public trace needed to diagnose
                    # the stable contract code below.
                    path = root / "public" / str(work_item["work_id"]).replace(":", "_") / route / "trajectory.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    contract_error = None
                    if row.get("e39_protocol") in _HCV_PROTOCOLS:
                        try:
                            if row.get("e39_protocol") == "agrinet.e319-rag-closure-audit/v1":
                                from agrinet.rag.e319_rag_closure_audit import validate_e319_trajectory
                                validate_e319_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e318-all-unknown-512-rag-audit/v1":
                                from agrinet.rag.e318_all_unknown_512_audit import validate_e318_trajectory
                                validate_e318_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e317-all-unknown-rag-audit/v1":
                                from agrinet.rag.e317_all_unknown_rag_audit import validate_e317_trajectory
                                validate_e317_trajectory(row, trajectory)
                            elif row.get("e39_protocol") in {"agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1"}:
                                from agrinet.rag.e316_rag_discriminator import validate_e316_trajectory
                                validate_e316_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e315-option-format-repair/v1":
                                from agrinet.rag.e315_option_format_repair import validate_e315_trajectory
                                validate_e315_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e314-hcv-cascade/v1":
                                from agrinet.rag.e314_hcv_cascade import validate_e314_trajectory
                                validate_e314_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e313-hcv-cascade/v1":
                                from agrinet.rag.e313_hcv_cascade import validate_e313_trajectory
                                validate_e313_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e312-hcv-cascade/v1":
                                from agrinet.rag.e312_hcv_cascade import validate_e312_trajectory
                                validate_e312_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e311-hcv-cascade/v1":
                                from agrinet.rag.e311_hcv_cascade import validate_e311_trajectory
                                validate_e311_trajectory(row, trajectory)
                            elif row.get("e39_protocol") == "agrinet.e310-hcv-cascade/v1":
                                from agrinet.rag.e310_hcv_cascade import validate_e310_trajectory
                                validate_e310_trajectory(row, trajectory)
                            else:
                                from agrinet.rag.e39_hcv_cascade import validate_e39_trajectory
                                validate_e39_trajectory(row, trajectory)
                        except ValueError as exc:
                            # E3.11 alone preserves a delivered Direct parent
                            # for legal escalation when its private auditor
                            # rejects it and only an internal HCV observation
                            # is incomplete. Outer Hermes and all tool errors
                            # remain fail-closed.
                            continuable = {"E3.11 requires three visual observations", "E3.11 requires two rejected alternatives", "E3.11 uncertainty is incomplete", "E3.11 Option reasoning must compare every public option"}
                            if not (row.get("e39_protocol") == "agrinet.e311-hcv-cascade/v1" and route == "direct" and audit["decision"] == "reject" and str(exc) in continuable):
                                raise
                            contract_error = _CONTRACT_ERRORS[str(exc)]
                    path = root / "public" / str(work_item["work_id"]).replace(":", "_") / route / "trajectory.json"; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    stages.append({"route": route, "delivery_status": "delivered", "private_audit": audit["decision"], "requires_rag": bool(audit.get("requires_rag")), "request_id": request_id, "private_audit_request_id": audit_request_id, "parent_path": str(path), "contract_error": contract_error})
                    break
                name, arguments = action.get("name"), action.get("arguments")
                split_pretool = False
                if row.get("e39_protocol") in _HCV_PROTOCOLS:
                    if row.get("e39_protocol") in {"agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}:
                        split_pretool = _e312_tool_turn_contract(messages, raw)
                    elif row.get("e39_protocol") in {"agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1"}:
                        _e310_tool_turn_contract(messages, raw)
                    else:
                        _e39_pretool_contract(raw)
                    if (route in {"classifier", "rag"} and not predicted and name != "agrinet_classifier_predict"):
                        raise ValueError("E3.9 tool sequence must start with classifier predict")
                    if row.get("e39_protocol") in {"agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}:
                        if name == "agrinet_classifier_predict" and predicted:
                            raise ValueError("E3.10 Classifier tool order is invalid")
                        if name == "agrinet_classifier_expand" and (route != "classifier" or expand_calls >= 1):
                            raise ValueError("E3.10 Classifier tool order is invalid")
                    if route == "rag" and name == "agrinet_rag_search" and rag_calls >= int(controls.get("max_rag_searches", 3)):
                        version = "E3.10" if row.get("e39_protocol") == "agrinet.e310-hcv-cascade/v1" else "E3.9"
                        raise ValueError(f"{version} RAG search limit exceeded")
                if name not in public_teacher_input(row, route).get("tools", []): raise ValueError("route tool contract violation")
                _validate_tool_arguments(str(name), arguments, predicted=predicted)
                if name == "agrinet_classifier_predict": result = public_classifier_card(row)
                elif name == "agrinet_classifier_expand": result = public_classifier_card(row, expanded=True)
                else:
                    # This runs after `ledger.call` has durably recorded the
                    # provider response.  Never turn an HTTP-500/GPU failure
                    # here into provider unknown_delivery.
                    try:
                        result = rag_search(row, arguments)
                    except Exception as exc:
                        raise LocalRagToolFailure(type(exc).__name__) from exc
                if name == "agrinet_classifier_predict": predicted = True
                if name == "agrinet_classifier_expand": expand_calls += 1
                if name == "agrinet_rag_search": rag_called = True
                if name == "agrinet_rag_search": rag_calls += 1
                trace.append({"call": {"name": name, "arguments": arguments}, "response": result})
                _append_native_tool_exchange(messages, raw, action, result, split_pretool_think=split_pretool)
                if row.get("e39_protocol") in {"agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}:
                    instruction = _e310_next_step_message(route=route, tool_name=str(name))
                    if instruction: messages.append({"role": "user", "content": instruction})
            else: raise ValueError("invalid_response")
        except BudgetObservedOverrun:
            return {"work_id": work_item["work_id"], "delivery_status": "budget_shortfall", "request_id": None, "winner": False, "final_route": route, "quality_status": "provider_usage_overrun", "parent_path": None}
        except BudgetExhausted:
            return {"work_id": work_item["work_id"], "delivery_status": "budget_shortfall", "request_id": None, "winner": False, "final_route": route, "quality_status": None, "parent_path": None}
        except DeliveryUnresolved as exc:
            stages.append({"route": route, "delivery_status": "unknown_delivery", "request_id": exc.request_id or f"unresolved:{work_item['work_id']}:{route}", "parent_path": None, "error_type": type(exc).__name__})
        except LocalRagToolFailure as exc:
            stages.append({"route": route, "delivery_status": "tool_shortfall",
                           "quality_status": "local_rag_failure", "request_id": request_id,
                           "parent_path": None, "error_type": str(exc)})
        except ValueError as exc:
            # Controller-detectable violations have a delivered provider
            # response and must never be admitted to delivery recovery.  The
            # specialized table retains stable public diagnostics, while a
            # future validator's detailed text is still a controller-quality
            # error rather than a provider ambiguity.  Actual upstream
            # ambiguity reaches the preceding DeliveryUnresolved branch.
            contract_error = _contract_error_code(exc) or "controller_contract_error"
            # The provider response and private audit are durable at this
            # point.  Keep the public trajectory for exact diagnosis even if
            # the subsequent controller validation rejects it.
            path = root / "public" / str(work_item["work_id"]).replace(":", "_") / route / "trajectory.json"
            stages.append({"route": route, "delivery_status": "delivered", "private_audit": "route_contract_reject", "request_id": f"rejected:{work_item['work_id']}:{route}", "parent_path": str(path) if path.exists() else None, "error_type": type(exc).__name__, "contract_error": contract_error})
        except Exception as exc:
            stages.append({"route": route, "delivery_status": "unknown_delivery", "request_id": f"unresolved:{work_item['work_id']}:{route}", "parent_path": None, "error_type": type(exc).__name__})
        # A private rejection is the sole legal reason to advance.  Do not
        # validate a deliberately incomplete prefix as though it were final.
        if stages[-1].get("delivery_status") in DELIVERY_FAILURES:
            return cascade_outcome(work_item=work_item, stages=stages)
        if row.get("e39_protocol") in {"agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}:
            if route == "direct" and stages[-1].get("private_audit") == "reject":
                continue
            private_fields={
                "agrinet.e314-hcv-cascade/v1": "e314_route_coverage",
                "agrinet.e316-rag-discriminator/v1": "e316_route_coverage",
                "agrinet.e316-rag-discriminator-canary/v1": "e316_route_coverage",
                "agrinet.e317-all-unknown-rag-audit/v1": "e317_route_coverage",
                "agrinet.e318-all-unknown-512-rag-audit/v1": "e318_route_coverage",
                "agrinet.e319-rag-closure-audit/v1": "e319_route_coverage",
            }
            private_designation=(row.get("private") or {}).get(private_fields.get(row.get("e39_protocol"), ""))
            if route == "classifier" and stages[-1].get("private_audit") == "reject" and (bool(stages[-1].get("requires_rag")) or private_designation == "rag"):
                continue
            stages[-1]["terminal_reject"] = True
            return cascade_outcome(work_item=work_item, stages=stages)
        if stages[-1].get("private_audit") != "reject" or route == "rag":
            return cascade_outcome(work_item=work_item, stages=stages)
    return cascade_outcome(work_item=work_item, stages=stages)
