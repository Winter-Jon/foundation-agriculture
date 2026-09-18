"""Native full-tool sample state machine. No semantic correction or synthetic calls."""
from __future__ import annotations

from agrinet.rag.answer_normalization import answer_key
import copy
import json
import time
from pathlib import Path

from agrinet.rag.e344_recovery import SampleLedger as E35Ledger
from agrinet.rag.e35_transport import transport_image
from agrinet.vlm.full_tool import (SYSTEM_PROMPT, CLASSIFIER_PREDICT, RAG_SEARCH,
                                   tool_schemas, validate_arguments, validate_messages,
                                   contract_hashes, candidate_union)


def collect_sample(row, *, root: Path, teacher, classifier, retrieve, private_audit, reference_mode=False):
    """Adapters must return actual responses; ledger refuses unknown-delivery replay.

    The caller binds and verifies image, checkpoint and source before entry.
    All messages sent to the teacher originate here, never from private row fields.
    """
    from agrinet.rag.dual_teacher import SAMPLER_GUIDANCE, reference_attachments
    import hashlib
    from agrinet.rag.dual_teacher import VERSION, AUDITOR_GUIDANCE, audit_payload as public_audit_payload
    if reference_mode:
        from agrinet.rag.dual_teacher_reaudit import GUIDANCE as AUDITOR_GUIDANCE, revised_payload as public_audit_payload
    mode_hash = hashlib.sha256((VERSION + SAMPLER_GUIDANCE + AUDITOR_GUIDANCE).encode()).hexdigest() if reference_mode else None
    seen_references = {}
    reference_records = []
    root.mkdir(parents=True, exist_ok=True)
    completed = root / "trajectory.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if reference_mode and result.get("reference_contract_sha256") != mode_hash:
            raise ValueError("reference_contract_changed")
        if bool(result.get("reference_mode", False)) != reference_mode:
            raise ValueError("completed_sample_reference_mode_changed")
        if result.get("contract") != contract_hashes():
            raise ValueError("completed_sample_contract_changed")
        return result
    ledger = E35Ledger(root / "ledger", work_id=row["sample_id"], attempt_ordinal=0, intent_limit=10 if reference_mode else 9)
    image, _ = transport_image(Path(row["image_path"]), max_side=1024)
    question = "Identify the agricultural disease or pest shown in this image."
    options = row.get("public_options", [])
    if row["question_type"] == "option":
        if len(options) != 4 or len({x["name"] for x in options}) != 4:
            raise ValueError("invalid_public_options")
        question += " Compare all four options and answer with the canonical English name.\n"
        question += "\n".join(f"{x['label']}. {x['name']}" for x in options)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + (chr(10) + SAMPLER_GUIDANCE if reference_mode else "")},
                {"role": "user", "content": [{"type": "text", "text": question}, image]}]
    if reference_mode:
        messages.append({"role": "user", "content": "Observation stage only. In this turn describe visible morphology, affected organ and what is not visible in 2-4 sentences. Do not name a disease, choose an option, emit answer tags, or call a tool. A separate next turn will request classification and retrieval."})
    trace, requests, predicted, searches, repaired = [], [], False, 0, False
    started = time.monotonic()
    observed = not reference_mode
    for turn in range(9 if reference_mode else 8):
        choice = ({"type": "function", "function": {"name": CLASSIFIER_PREDICT}} if not predicted else
                  {"type": "function", "function": {"name": RAG_SEARCH}} if searches == 0 else
                  "none" if searches == 3 else "auto")
        if not observed: choice = "none"
        payload = {"model": "gpt-5.6-sol", "messages": copy.deepcopy(messages), "tools": tool_schemas(),
                   "tool_choice": choice, "parallel_tool_calls": False, "temperature": 0,
                   "max_tokens": 4096}
        request_id, raw = ledger.call(kind="generation", key=f"turn:{turn}", payload=payload,
                                      invoke=lambda: teacher(payload))
        requests.append({"request_id": request_id, "raw": raw})
        message = copy.deepcopy(raw["choices"][0]["message"])
        message = {k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "reasoning_content"}}
        messages.append(message)
        calls = message.get("tool_calls") or []
        errors = []
        if not observed and not calls and len(str(message.get("content") or "").strip()) >= 40 and "<answer>" not in str(message.get("content")):
            observed = True
            messages.append({"role": "user", "content": "Observation stage complete. Now call agrinet_classifier_predict, then perform 1-3 RAG searches. Do not give a final answer before reading their results."})
            continue
        if calls:
            if len(calls) != 1:
                errors = ["exactly_one_call_per_turn_required"]
            else:
                fn = calls[0].get("function", {})
                try:
                    args = json.loads(fn.get("arguments", ""))
                except (ValueError, TypeError):
                    args = None
                errors = validate_arguments(fn.get("name"), args)
                if not errors and ((fn["name"] == CLASSIFIER_PREDICT and predicted) or
                                   (fn["name"] == RAG_SEARCH and (not predicted or searches >= 3))):
                    errors = ["tool_order_or_count"]
            if not errors:
                cache = root / f"tool-{len(trace)}.json"
                binding = {"name": fn["name"], "arguments": args, "image_sha256": row["image_sha256"]}
                if cache.exists():
                    saved = json.loads(cache.read_text())
                    if saved["binding"] != binding:
                        raise ValueError("cached_tool_binding_changed")
                    response = saved["response"]
                else:
                    response = classifier(row) if fn["name"] == CLASSIFIER_PREDICT else retrieve(row, args)
                    temporary = cache.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"binding": binding, "response": response}))
                    temporary.replace(cache)
                messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                                 "content": json.dumps(response, ensure_ascii=False)})
                trace.append({"call": calls[0], "response": response})
                if reference_mode and fn["name"] == RAG_SEARCH:
                    attachment, records = reference_attachments(response, "datasets/AgriNet-1K/wiki/images", seen_references, row["image_sha256"], row["image_path"])
                    if attachment: messages.append(attachment)
                    reference_records.extend(records)
                predicted |= fn["name"] == CLASSIFIER_PREDICT
                searches += fn["name"] == RAG_SEARCH
                continue
        else:
            try:
                validation = validate_messages(messages)
            except ValueError as exc:
                errors = [str(exc)]
            else:
                result = {"messages": messages, "tool_trace": trace, "requests": requests,
                          "validation": validation, "contract": contract_hashes(),
                          "elapsed_seconds": time.monotonic() - started, "quality_repairs": int(repaired), "reference_records": reference_records, "reference_mode": reference_mode, "reference_contract_sha256": mode_hash}
                # Deterministic truth/candidate checks belong in the isolated reviewer.
                audit_payload = {"messages": messages, "tool_trace": trace,
                                 "contract": contract_hashes(), "private_truth": row["private"]}
                if reference_mode: audit_payload = public_audit_payload(row, messages)
                _, review = ledger.call(kind="private_audit", key="private_audit", payload=audit_payload,
                                         invoke=lambda: private_audit(row, result))
                result["private_audit"] = review
                result["candidate_union"] = candidate_union(trace)
                if review.get("raw"):
                    result["requests"].append({"request_id": _, "kind": "private_audit", "raw": review["raw"]})
                result["elapsed_seconds"] = time.monotonic() - started
                result["training_eligible"] = (not validation["insufficient_evidence"]
                    and review.get("semantic") == "correct" and review.get("quality") == "pass")
                allowed_names = ({option["name"] for option in options} if options else
                                 {candidate["name"] for candidate in result["candidate_union"]})
                if answer_key(validation["answer"]) not in {answer_key(name) for name in allowed_names}:
                    result["training_eligible"] = False
                result["disposition"] = ("qualified" if result["training_eligible"] else
                    "insufficient_evidence" if validation["insufficient_evidence"] else
                    "semantic_incorrect" if review.get("semantic") == "incorrect" else "quality_rejected")
                (root / "trajectory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                return result
        if repaired:
            break
        repaired = True
        # A malformed native call still needs matching error responses for provider validity.
        for call in calls:
            response = {"protocol_error": errors}
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(response)})
            trace.append({"call": call, "response": response})
        from agrinet.rag.protocol_repair import repair_instruction
        messages.append({"role": "user", "content": repair_instruction(errors)})
    result = {"messages": messages, "tool_trace": trace, "requests": requests,
              "training_eligible": False, "disposition": "protocol_failure",
              "elapsed_seconds": time.monotonic() - started, "contract": contract_hashes(),
              "reference_mode": reference_mode, "reference_contract_sha256": mode_hash, "reference_records": reference_records}
    (root / "trajectory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
