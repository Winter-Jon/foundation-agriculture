"""Collect E3.42 tool-free refusal trajectories with per-sample recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e35_budget import BudgetExhausted, E35TokenBudget, uncached_input_tokens
from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS
from agrinet.rag.e342_refusal_contract import public_catalog, render_refusal, validate_refusal
from agrinet.rag.e342_refusal_trajectories import EXPECTED_ROWS, PROTOCOL


class AuditUnresolved(ValueError):
    """Private refusal audit was delivered but failed its strict schema."""

ROUND_FILES = ("r0.json", "q1.json", "r1.json", "q1-r1.json", "r2.json", "q1-r2.json")
PRIVATE_AUDIT_POLICY = "legacy_subjective"


def _write(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"E3.42 immutable output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _latest(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for shard in sorted((root / "shards").glob("shard-*")):
        for name in ROUND_FILES:
            path = shard / "outcomes" / name
            if path.is_file():
                for outcome in json.loads(path.read_text()).get("outcomes") or []:
                    latest[outcome["sample_id"]] = outcome
    return latest


def _public_question(row: dict[str, Any]) -> str:
    question = str(row["question"])
    if row.get("question_type") == "option":
        question += "\n" + "\n".join(f"{item['label']}. {item['name']}" for item in row["public_options"])
    return question


def _parent_public_evidence(row: dict[str, Any]) -> dict[str, Any]:
    parent = row["e342_parent"]
    trajectory = json.loads(Path(parent["trajectory_path"]).read_text())
    evidence = json.loads(Path(parent["evidence_path"]).read_text())
    return {"parent_trajectory_answer": trajectory.get("answer"),
            "parent_rag_evidence": {"arguments": evidence.get("arguments"),
                                    "returned_standard_class_names": evidence.get("returned_standard_class_names"),
                                    "public_evidence": (evidence.get("raw_response") or {}).get("evidence")},
            "parent_trajectory_sha256": parent["trajectory_sha256"],
            "parent_evidence_sha256": parent["evidence_sha256"]}


def _refusal_payload(row: dict[str, Any], *, model: str, quality_repair: bool = False) -> dict[str, Any]:
    catalog = public_catalog(row)
    required_ids = sorted(catalog)
    system = ("Return JSON only with exactly observations, candidate_assessments, evidence_limitations, refusal_rationale, confidence, limitation. "
              "Do not call tools; no tool schema is available. Do not choose a class name, option letter, score, rank, or route. "
              "Use only image-visible traits and the supplied parent public evidence. "
              "observations must contain exactly three concise visible facts. candidate_assessments must contain exactly one object for each required candidate_id, with exactly candidate_id, visible_match, conflict_or_missing. "
              "Each candidate must state one visible match and one conflict or decisive missing trait. "
              "Explain why ambiguous, low-resolution, or merely possible features cannot safely close to any candidate. confidence must be low or medium.")
    if quality_repair:
        system += " This is the single complete format repair. Rebuild the full object from the same supplied public evidence; do not mention any prior response."
    public = {"question": _public_question(row), "candidate_catalog": catalog, "required_candidate_ids": required_ids,
              "parent_public_evidence": _parent_public_evidence(row)}
    return {"model": model, "temperature": 0, "max_tokens": 2200, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(public, ensure_ascii=False)}]}


def _private_audit_payload(row: dict[str, Any], trajectory: dict[str, Any], *, model: str) -> dict[str, Any]:
    parent = row["e342_parent"]
    content = {"parent_protocol": parent["parent_protocol"], "parent_terminal_disposition": parent["parent_outcome"]["disposition"],
               "parent_trajectory_sha256": parent["trajectory_sha256"], "parent_evidence_sha256": parent["evidence_sha256"],
               "public_refusal_trajectory": trajectory}
    system = ("You are an isolated refusal auditor. Return JSON only with exactly decision (accept or reject) and quality (pass or fail). "
              "Accept only if the parent is a future_reject, the public trajectory has no tool calls, uses only public parent evidence, assesses all required candidates, contains no private data, and ends exactly with INSUFFICIENT_EVIDENCE. "
              "Do not compare to or reveal a truth class, label map, checkpoint, fold, correct option, reason, correction, or route recommendation.")
    if PRIVATE_AUDIT_POLICY == "demonstrable_boundary_only":
        content.update({"public_question": _public_question(row), "candidate_catalog": public_catalog(row),
                        "parent_public_evidence": _parent_public_evidence(row)})
        system = ("You are an isolated refusal boundary auditor. Return JSON only with exactly violation_codes, a JSON array of strings. "
                  "Report a code only for a demonstrable violation in this supplied public packet: tool use, non-fixed final answer, private-data leakage, missing or duplicate required candidate ID, or direct internal contradiction. "
                  "Do not report a code because you prefer a different visual judgment, evidence is ambiguous, or you prefer a different rationale. "
                  "Do not reveal truth classes, labels, folds, checkpoints, correct options, or recommendations.")
    return {"model": model, "temperature": 0, "max_tokens": 256, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]}


def _parse_json_response(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        content = raw["choices"][0]["message"]["content"]
        value = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("E3.42 provider response is not a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("E3.42 provider response is not an object")
    return value


def _parse_private_audit(raw: dict[str, Any]) -> dict[str, str]:
    try:
        value = _parse_json_response(raw)
    except ValueError as exc:
        raise AuditUnresolved("E3.42 private refusal audit invalid") from exc
    if PRIVATE_AUDIT_POLICY == "demonstrable_boundary_only":
        codes = value.get("violation_codes")
        if set(value) != {"violation_codes"} or not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
            raise AuditUnresolved("E3.43 private boundary audit invalid")
        return {"decision": "accept", "quality": "pass", "provider_violation_codes": sorted(set(codes))}
    if set(value) != {"decision", "quality"} or value.get("decision") not in {"accept", "reject"} or value.get("quality") not in {"pass", "fail"}:
        raise AuditUnresolved("E3.42 private refusal audit invalid")
    return {"decision": value["decision"], "quality": value["quality"]}


def _demonstrable_boundary_violations(row: dict[str, Any], refusal: dict[str, Any], trajectory: dict[str, Any]) -> list[str]:
    """Return only violations that can be proven from the local public packet."""
    violations: list[str] = []
    try:
        validate_refusal(refusal, row, public_catalog(row))
    except ValueError:
        violations.append("public_contract_invalid")
    if trajectory.get("route") != "refusal" or trajectory.get("tool_trace") != []:
        violations.append("tool_or_route_boundary")
    if trajectory.get("answer", "").split("</think>")[-1] != "<answer>INSUFFICIENT_EVIDENCE</answer>":
        violations.append("fixed_answer_invalid")
    rendered = json.dumps({"refusal": refusal, "trajectory": trajectory}, ensure_ascii=False).casefold()
    if any(token in rendered for token in ("private truth", "correct option", "label map", "checkpoint", "held_out_fold")):
        violations.append("private_data_leakage")
    return violations


def _request_id(root: Path, item: dict[str, Any], kind: str) -> str:
    path = root / "ledgers" / item["work_id"].replace(":", "_") / "events.jsonl"
    if not path.is_file():
        return ""
    key = f"{item['work_id']}:{kind}"
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "intent" and event.get("key") == key:
            return str(event.get("request_id") or "")
    return ""


def _call(*, item: dict[str, Any], kind: str, payload: dict[str, Any], teacher, budget: E35TokenBudget,
          intents: GlobalMicuBudget, root: Path, reserve: int) -> tuple[str, dict[str, Any]]:
    key = f"{item['work_id']}:{kind}"
    budget.reserve(key, uncached_input_tokens=reserve, metadata={"kind": kind})
    intents.reserve(key)
    ledger = E35Ledger(root / "ledgers" / item["work_id"].replace(":", "_"), work_id=item["work_id"],
                       attempt_ordinal=item["attempt_ordinal"], intent_limit=8_000)
    payload_record = {"operation": kind, "wire_payload_sha256": digest_payload(payload), "tools": []}
    request_id, raw = ledger.call(kind="private_audit" if kind == "private_refusal_audit" else "generation",
                                  key=kind, payload=payload_record, invoke=lambda: teacher(payload))
    observed = uncached_input_tokens(raw)
    if observed is not None:
        budget.settle(key, uncached_input_tokens=observed)
    return request_id, raw


def digest_payload(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _persist(root: Path, item: dict[str, Any], name: str, value: Any) -> tuple[str, str]:
    path = root / "states" / item["work_id"].replace(":", "_") / f"{name}.json"
    _write(path, value)
    return str(path), digest(path)


def _one(row: dict[str, Any], item: dict[str, Any], *, model: str, teacher, budget: E35TokenBudget,
         intents: GlobalMicuBudget, root: Path) -> dict[str, Any]:
    base = {key: item.get(key) for key in ("work_id", "sample_id", "round", "attempt_ordinal", "quality_attempt_ordinal", "predecessor_request_id", "resume_operation")}
    state_dir = root / "states" / item["work_id"].replace(":", "_")
    parent_dir = Path(str(item.get("state_dir") or state_dir))
    current = "refusal"
    try:
        reuse_refusal = item.get("resume_operation") == "private_refusal_audit"
        if reuse_refusal:
            refusal = json.loads((parent_dir / "refusal.json").read_text())
            catalog = public_catalog(row)
            validate_refusal(refusal, row, catalog)
            refusal_path, refusal_sha = _persist(root, item, "refusal", refusal)
            refusal_request_id = str(item.get("refusal_request_id") or _request_id(root, item, "refusal"))
        else:
            quality_repair = item.get("resume_operation") == "refusal_quality_repair"
            refusal_request_id, raw = _call(item=item, kind="refusal", payload=_refusal_payload(row, model=model, quality_repair=quality_repair),
                                            teacher=teacher, budget=budget, intents=intents, root=root, reserve=4_000)
            refusal = validate_refusal(_parse_json_response(raw), row, public_catalog(row))
            refusal_path, refusal_sha = _persist(root, item, "refusal", refusal)
        trajectory = {"route": "refusal", "answer": render_refusal(refusal, public_catalog(row)), "tool_trace": [],
                      "parent_e341": row["e342_parent"], "messages": []}
        if trajectory["answer"].split("</think>")[-1] != "<answer>INSUFFICIENT_EVIDENCE</answer>":
            raise ValueError("E3.42 renderer answer invalid")
        trajectory_path, trajectory_sha = _persist(root, item, "trajectory", trajectory)
        current = "private_refusal_audit"
        audit_request_id, audit_raw = _call(item=item, kind="private_refusal_audit",
                                            payload=_private_audit_payload(row, trajectory, model=model), teacher=teacher,
                                            budget=budget, intents=intents, root=root, reserve=2_000)
        audit = _parse_private_audit(audit_raw)
        violations = _demonstrable_boundary_violations(row, refusal, trajectory) if PRIVATE_AUDIT_POLICY == "demonstrable_boundary_only" else []
        if PRIVATE_AUDIT_POLICY == "demonstrable_boundary_only":
            audit = {**audit, "demonstrable_violation_codes": violations}
        disposition = "refusal_trajectory_complete" if (audit.get("decision") == "accept" and audit.get("quality") == "pass" and not violations) else "refusal_rejected"
        return {**base, "delivery_status": "delivered", "request_id": audit_request_id, "refusal_request_id": refusal_request_id,
                "private_refusal_audit_request_id": audit_request_id, "disposition": disposition, "audit": audit,
                "quality": audit["quality"], "refusal_path": refusal_path, "refusal_sha256": refusal_sha,
                "trajectory_path": trajectory_path, "trajectory_sha256": trajectory_sha, "tool_calls": 0, "winner": disposition == "refusal_trajectory_complete"}
    except DeliveryUnresolved as exc:
        return {**base, "delivery_status": "unknown_delivery", "request_id": exc.request_id,
                "unresolved_operation": current, "state_dir": str(state_dir), "refusal_request_id": _request_id(root, item, "refusal"),
                "disposition": "delivery_unknown", "winner": False}
    except BudgetExhausted:
        return {**base, "delivery_status": "budget_shortfall", "request_id": None, "disposition": "budget_shortfall", "winner": False}
    except (ValueError, RuntimeError) as exc:
        return {**base, "delivery_status": "delivered", "request_id": _request_id(root, item, current), "failed_operation": current,
                "state_dir": str(state_dir), "refusal_request_id": _request_id(root, item, "refusal"),
                "disposition": "quality_reject", "contract_error": str(exc), "winner": False}


def _successor(prior: dict[str, Any], round_name: str, manifest_sha: str, *, quality: bool = False) -> dict[str, Any]:
    request_id = prior.get("request_id")
    if not request_id:
        raise ValueError("E3.42 successor lacks predecessor request ID")
    if quality:
        if prior.get("quality_attempt_ordinal") != 0 or prior.get("disposition") != "quality_reject":
            raise ValueError("E3.42 invalid Q1 predecessor")
        operation = prior.get("failed_operation") or "refusal"
        if operation == "private_refusal_audit":
            resume = "private_refusal_audit"
        else:
            resume = "refusal_quality_repair"
        attempt, quality_attempt = prior["attempt_ordinal"], 1
    else:
        if round_name not in {"R1", "R2"} or prior.get("delivery_status") != "unknown_delivery":
            raise ValueError("E3.42 invalid delivery-recovery predecessor")
        resume = prior.get("unresolved_operation") or "refusal"
        attempt, quality_attempt = prior["attempt_ordinal"] + 1, prior["quality_attempt_ordinal"]
    return {"work_id": f"{round_name}:{prior['sample_id']}:e342-{'quality' if quality else 'delivery'}",
            "sample_id": prior["sample_id"], "round": round_name, "attempt_ordinal": attempt,
            "quality_attempt_ordinal": quality_attempt, "resume_operation": resume,
            "predecessor_request_id": request_id, "predecessor_outcome_sha256": digest_payload({key: value for key, value in prior.items() if not key.startswith("_")}),
            "predecessor_manifest_sha256": manifest_sha, "state_dir": prior.get("state_dir"),
            "refusal_request_id": prior.get("refusal_request_id")}


def _outcome(path: Path, *, round_name: str, manifest_sha: str, shard: int, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"schema_version": "agrinet.e342-refusal-trajectories-outcomes/v1", "protocol": PROTOCOL,
             "round": round_name, "shard_index": shard, "manifest_sha256": manifest_sha, "outcomes": outcomes, **FLAGS}
    _write(path, value)
    return value


def _continuation(path: Path, *, round_name: str, root_sha: str, shard: int, items: list[dict[str, Any]]) -> str:
    value = {"schema_version": "agrinet.e342-refusal-trajectories-continuation/v1", "protocol": PROTOCOL,
             "round": round_name, "shard_index": shard, "root_manifest_sha256": root_sha, "work_items": items, **FLAGS}
    _write(path, value)
    return digest(path)


def _run(items: list[dict[str, Any]], by: dict[str, dict[str, Any]], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    if not items:
        return []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
        return list(pool.map(lambda item: _one(by[item["sample_id"]], item, **kwargs), items))


def _execute(*, root: Path, root_sha: str, shard: int, round_name: str, items: list[dict[str, Any]],
             by: dict[str, dict[str, Any]], kwargs: dict[str, Any], initial: bool = False) -> dict[str, Any]:
    shard_root = root / "shards" / f"shard-{shard:02d}"
    manifest_sha = root_sha if initial else _continuation(shard_root / "manifests" / f"{round_name.lower()}.json",
                                                          round_name=round_name, root_sha=root_sha, shard=shard, items=items)
    path = shard_root / "outcomes" / f"{round_name.lower()}.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("manifest_sha256") != manifest_sha:
            raise ValueError("E3.42 immutable outcome manifest binding changed")
        return value
    return _outcome(path, round_name=round_name, manifest_sha=manifest_sha, shard=shard, outcomes=_run(items, by, kwargs))


def _bound(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**outcome, "_manifest_sha256": document["manifest_sha256"]} for outcome in document.get("outcomes") or []]


def _run_shard(*, root: Path, root_sha: str, ref: dict[str, Any], by: dict[str, dict[str, Any]], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    shard = int(ref["shard_index"])
    plan = json.loads(Path(ref["path"]).read_text())
    r0 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="R0", items=plan["work_items"], by=by, kwargs=kwargs, initial=True)
    r0b = _bound(r0)
    q1 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="Q1", by=by, kwargs=kwargs,
                  items=[_successor(item, "Q1", item["_manifest_sha256"], quality=True) for item in r0b if item.get("disposition") == "quality_reject"])
    prior = [item for item in r0b + _bound(q1) if item.get("delivery_status") == "unknown_delivery"]
    r1 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="R1", by=by, kwargs=kwargs,
                  items=[_successor(item, "R1", item["_manifest_sha256"]) for item in prior])
    r1b = _bound(r1)
    q1r1 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="Q1-R1", by=by, kwargs=kwargs,
                    items=[_successor(item, "Q1-R1", item["_manifest_sha256"], quality=True) for item in r1b if item.get("disposition") == "quality_reject" and item.get("quality_attempt_ordinal") == 0])
    prior2 = [item for item in r1b + _bound(q1r1) if item.get("delivery_status") == "unknown_delivery"]
    r2 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="R2", by=by, kwargs=kwargs,
                  items=[_successor(item, "R2", item["_manifest_sha256"]) for item in prior2])
    r2b = _bound(r2)
    q1r2 = _execute(root=root, root_sha=root_sha, shard=shard, round_name="Q1-R2", by=by, kwargs=kwargs,
                    items=[_successor(item, "Q1-R2", item["_manifest_sha256"], quality=True) for item in r2b if item.get("disposition") == "quality_reject" and item.get("quality_attempt_ordinal") == 0])
    return [r0, q1, r1, q1r1, r2, q1r2]


def final_report(source_rows: list[dict[str, Any]], documents: list[dict[str, Any]], budget: E35TokenBudget) -> dict[str, Any]:
    from collections import Counter, defaultdict
    latest: dict[str, dict[str, Any]] = {}
    for document in documents:
        for outcome in document.get("outcomes") or []:
            latest[outcome["sample_id"]] = outcome
    expected = {row["sample_id"] for row in source_rows}
    if set(latest) != expected:
        raise ValueError("E3.42 final lineage incomplete")
    terminal = Counter(outcome.get("disposition") for outcome in latest.values())
    complete = sorted(sid for sid, outcome in latest.items() if outcome.get("disposition") == "refusal_trajectory_complete")
    residuals = [{"sample_id": sid, "disposition": outcome.get("disposition"), "delivery_status": outcome.get("delivery_status"),
                  "contract_error": outcome.get("contract_error"), "audit": outcome.get("audit"), **FLAGS}
                 for sid, outcome in sorted(latest.items()) if sid not in complete]
    dimensions = {"arm": lambda row: str((row.get("private") or {}).get("arm") or "unknown"),
                  "question_type": lambda row: str(row.get("question_type") or "unknown"),
                  "domain": lambda row: str(row.get("domain") or "unknown"),
                  "fold": lambda row: str((row.get("classifier") or {}).get("held_out_fold") or "unknown"),
                  "e327_overlap": lambda row: "overlap" if row.get("e327_overlap") else "non_overlap",
                  "truth_code": lambda row: str((row.get("private") or {}).get("truth_code") or "unknown")}
    strata = {}
    for name, key in dimensions.items():
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for row in source_rows:
            bucket = buckets[key(row)]
            outcome = latest[row["sample_id"]]
            bucket["rows"] += 1
            bucket[str(outcome.get("disposition"))] += 1
        strata[name] = {bucket: dict(counts) for bucket, counts in sorted(buckets.items())}
    unknown = [sid for sid, outcome in latest.items() if outcome.get("delivery_status") == "unknown_delivery"]
    terminal_unknown_limit = 8
    non_delivery_residuals = [outcome for outcome in latest.values()
                              if outcome.get("disposition") != "refusal_trajectory_complete"
                              and outcome.get("delivery_status") != "unknown_delivery"]
    candidate = not non_delivery_residuals and len(unknown) <= terminal_unknown_limit
    return {"schema_version": "agrinet.e342-refusal-trajectories-final-report/v1", "protocol": PROTOCOL,
            "rows": len(source_rows), "final_disposition": dict(terminal), "refusal_trajectory_complete_count": len(complete),
            "refusal_trajectory_complete_samples": complete, "refusal_residuals": residuals,
            "terminal_unknown_delivery": sorted(unknown), "terminal_unknown_delivery_count": len(unknown),
            "terminal_unknown_delivery_limit": terminal_unknown_limit, "stratification": strata, "budget": budget.report(),
            "campaign_gate_candidate": candidate, "refusal_gate_passed": False,
            "next_action": "artifact_audit_then_interview_user", **FLAGS}


def run_campaign(*, manifest: Path, source: Path, output_root: Path, teacher_model: str = "gpt-5.6-sol",
                 timeout: int = 180, authorize_live_collection: bool = False) -> dict[str, Any]:
    validate_inputs(manifest=manifest, source=source)
    if teacher_model != "gpt-5.6-sol" or timeout != 180 or not authorize_live_collection:
        raise ValueError("E3.42 live authorization invalid")
    report_path = output_root / "final-report.json"
    if report_path.exists():
        from agrinet.rag.e342_refusal_audit import audit, gate_decision
        audit_path = output_root / "artifact-audit.json"
        audit(source=source, campaign_root=output_root, output=audit_path)
        return {**json.loads(report_path.read_text()), **gate_decision(report=report_path, audit_report=audit_path, output=output_root / "final-gate-decision.json")}
    from agrinet.common.credentials import yunwu_environment
    environment = yunwu_environment(profile="micu_slb")
    base_url = str(environment.get("YUNWU_API_BASE_URL") or "")
    if not base_url.startswith("https://"):
        raise ValueError("E3.42 teacher endpoint must be HTTPS")
    headers = {"Authorization": f"Bearer {environment['YUNWU_API_KEY']}", "Content-Type": "application/json"}
    teacher = lambda payload: _isolated_micu_request(base_url.rstrip("/") + "/chat/completions", payload, headers, timeout)
    plan = json.loads(manifest.read_text())
    controls = plan["collection_controls"]
    budget = E35TokenBudget(output_root / "token_budget.jsonl", uncached_input_token_cap=controls["uncached_input_token_cap"])
    intents = GlobalMicuBudget(output_root / "global_micu_intents.jsonl", limit=controls["global_micu_intent_limit"])
    data = rows(source)
    by = {row["sample_id"]: row for row in data}
    kwargs = {"model": teacher_model, "teacher": teacher, "budget": budget, "intents": intents, "root": output_root}
    root_sha = digest(manifest)
    docs: list[dict[str, Any]] = []
    for ref in plan["shards"]:
        docs.extend(_run_shard(root=output_root, root_sha=root_sha, ref=ref, by=by, kwargs=kwargs))
    report = final_report(data, docs, budget)
    _write(report_path, report)
    from agrinet.rag.e342_refusal_audit import audit, gate_decision
    audit_path = output_root / "artifact-audit.json"
    audit(source=source, campaign_root=output_root, output=audit_path)
    return {**report, **gate_decision(report=report_path, audit_report=audit_path, output=output_root / "final-gate-decision.json")}


def validate_inputs(*, manifest: Path, source: Path) -> dict[str, Any]:
    plan = json.loads(manifest.read_text())
    data = rows(source)
    controls = plan.get("collection_controls") or {}
    expected_controls = {"uncached_input_token_cap": 1_000_000, "global_micu_intent_limit": 8_000,
                         "request_timeout_seconds": 180, "workers": 4,
                         "reservation_uncached_tokens": {"refusal": 4_000, "private_refusal_audit": 2_000},
                         "recovery_rounds": ["R0", "R1", "R2"], "quality_repair_max": 1,
                         "terminal_unknown_delivery_limit": 8}
    if plan.get("protocol") != PROTOCOL or plan.get("source_sha256") != digest(source):
        raise ValueError("E3.42 source binding invalid")
    if len(data) != EXPECTED_ROWS or plan.get("source_rows_expected") != EXPECTED_ROWS:
        raise ValueError("E3.42 source row count invalid")
    if controls != expected_controls or plan.get("shards_are_scheduling_only") is not True:
        raise ValueError("E3.42 collection controls invalid")
    forbidden = ["agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search", "planner", "ranker", "agrinet_reject"]
    if plan.get("forbidden_tools") != forbidden or any(plan.get(key) is not False for key in FLAGS):
        raise ValueError("E3.42 tool or training boundary invalid")
    expected_ids = [row["sample_id"] for row in data]
    seen: list[str] = []
    for ref in plan.get("shards") or []:
        shard_path = Path(str(ref.get("path") or ""))
        if not shard_path.is_file() or digest(shard_path) != ref.get("sha256"):
            raise ValueError("E3.42 shard binding invalid")
        shard = json.loads(shard_path.read_text())
        if shard.get("protocol") != PROTOCOL or digest(source) != shard.get("source_sha256"):
            raise ValueError("E3.42 shard/source binding invalid")
        seen.extend(shard.get("sample_ids") or [])
    if seen != expected_ids:
        raise ValueError("E3.42 shard coverage invalid")
    for row in data:
        parent = row.get("e342_parent") or {}
        trajectory = Path(str(parent.get("trajectory_path") or ""))
        evidence = Path(str(parent.get("evidence_path") or ""))
        if not trajectory.is_file() or not evidence.is_file() or digest(trajectory) != parent.get("trajectory_sha256") or digest(evidence) != parent.get("evidence_sha256"):
            raise ValueError("E3.42 parent lineage binding invalid")
        public_catalog(row)
    return {"ready": True, "rows": len(data), "provider_requests": 0, "protocol": PROTOCOL, **FLAGS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--authorize-live-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(validate_inputs(manifest=args.manifest, source=args.source), sort_keys=True))
        return 0
    value = run_campaign(manifest=args.manifest, source=args.source, output_root=args.output_root,
                         teacher_model=args.teacher_model, timeout=args.timeout,
                         authorize_live_collection=args.authorize_live_collection)
    print(json.dumps({"rows": value["rows"], "refusal_gate_passed": value["refusal_gate_passed"], **FLAGS}, sort_keys=True))
    return 0 if value["refusal_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
