"""E3.5 public cascade controller.

This module intentionally has no implicit provider fallback: a live caller must
explicitly bind a teacher, isolated private auditor, and the typed local RAG
surface.  That keeps an offline audit or unit test from accidentally spending
teacher budget.
"""
from __future__ import annotations

import json
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


class LocalRagToolFailure(RuntimeError):
    """A local RAG call failed after its teacher tool call was delivered."""

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
}

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
        payload["tool_choice"] = ({"type": "function", "function": {"name": "agrinet_rag_search"}}
                                  if route == "rag" and not rag_called else "auto")
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
                                 action: dict[str, Any], result: dict[str, Any]) -> None:
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
    messages.append({"role": "assistant", "content": message.get("content"),
                     "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": action["tool_call_id"],
                     "content": json.dumps(result, ensure_ascii=False)})


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
                      "rag_evidence_insufficient", "private_registry_sha256")
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
    if plan.get("schema_version") not in {"agrinet.e35-cascade-manifest/v1", "agrinet.e35-cascade-manifest/v2"} or not isinstance(items, list):
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
        rag_called = False
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
                if action["type"] == "final":
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
                    if action["content"] == "INSUFFICIENT_EVIDENCE":
                        trajectory["private_rag_evidence_insufficient"] = bool(audit.get("rag_evidence_insufficient"))
                        validate_rag_terminal(trajectory)
                    path = root / "public" / str(work_item["work_id"]).replace(":", "_") / route / "trajectory.json"; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    stages.append({"route": route, "delivery_status": "delivered", "private_audit": audit["decision"], "request_id": request_id, "private_audit_request_id": audit_request_id, "parent_path": str(path)})
                    break
                name, arguments = action.get("name"), action.get("arguments")
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
                if name == "agrinet_rag_search": rag_called = True
                trace.append({"call": {"name": name, "arguments": arguments}, "response": result})
                _append_native_tool_exchange(messages, raw, action, result)
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
            # response and must never be admitted to delivery recovery.
            contract_error = _CONTRACT_ERRORS.get(str(exc))
            if contract_error is not None:
                stages.append({"route": route, "delivery_status": "delivered", "private_audit": "route_contract_reject", "request_id": f"rejected:{work_item['work_id']}:{route}", "parent_path": None, "error_type": type(exc).__name__, "contract_error": contract_error})
            else:
                stages.append({"route": route, "delivery_status": "unknown_delivery", "request_id": f"unresolved:{work_item['work_id']}:{route}", "parent_path": None, "error_type": type(exc).__name__})
        except Exception as exc:
            stages.append({"route": route, "delivery_status": "unknown_delivery", "request_id": f"unresolved:{work_item['work_id']}:{route}", "parent_path": None, "error_type": type(exc).__name__})
        # A private rejection is the sole legal reason to advance.  Do not
        # validate a deliberately incomplete prefix as though it were final.
        if stages[-1].get("delivery_status") in DELIVERY_FAILURES:
            return cascade_outcome(work_item=work_item, stages=stages)
        if stages[-1].get("private_audit") != "reject" or route == "rag":
            return cascade_outcome(work_item=work_item, stages=stages)
    return cascade_outcome(work_item=work_item, stages=stages)
