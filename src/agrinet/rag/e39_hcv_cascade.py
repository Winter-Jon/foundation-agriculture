"""E3.9 public HCV contract layered on the immutable E3.5 source pool.

This module contains no provider calls.  It makes the collection-time Hermes
and HCV promises executable, while deliberately leaving historical E3.5
artifacts untouched.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agrinet.rag.e35_classifier_cascade import ARMS, DELIVERY_FAILURES, ROUTES, recovery_attempt, validate_source_rows
from agrinet.rag.hermes_protocol import is_final_answer, is_pre_tool_think

E39_PROTOCOL = "agrinet.e39-hcv-cascade/v1"
FIELDS = ("Visual observations:", "Candidate hypotheses:", "Candidate comparison:",
          "Evidence:", "Rejected alternatives:", "Uncertainty:")

BASE = ("You are an agricultural visual-diagnosis teacher. Use only the image, the public question, and actual public tool results. "
        "Produce useful HCV supervision: hypothesize candidates, contrast their discriminative traits, then verify only with evidence available on this route. "
        "Never infer private labels, classifier coverage, folds, or hidden metadata. "
        "Your final response must be exactly <think>...</think><answer>...</answer>. Inside <think>, put these headings in this exact order, each at the beginning of a line: "
        "Visual observations:, Candidate hypotheses:, Candidate comparison:, Evidence:, Rejected alternatives:, Uncertainty:. "
        "Give at least three directly visible observations over at least two visual dimensions. Compare candidates with concrete supporting or conflicting traits, not name semantics, rank, or score. "
        "Name at least two rejected alternatives with candidate-specific reasons. State low, medium, or high uncertainty and its evidence limitation. ")

E39_TEACHER_PROMPTS = {
    "direct": BASE + "No tools are available. For Open questions, propose two or three concrete canonical class-name hypotheses grounded in visible image traits. For Option questions, use exactly the four public choices as candidates and compare all of them. Evidence may contain only visible image facts. For Open answer with only the canonical public class name; for Option answer with `class name — letter`.",
    "classifier": BASE + "The full JSON Schemas for agrinet_classifier_predict and agrinet_classifier_expand are available. Your first action must be native agrinet_classifier_predict, preceded by a nonempty <think> planning turn. Treat its public card as the candidate source. You may call expand once only if Top-3 remains unresolved. Use scores only as weak clues. After actual results, give the required final format.",
    "rag": BASE + "The full JSON Schemas for agrinet_classifier_predict, agrinet_classifier_expand, and agrinet_rag_search are available. First call native agrinet_classifier_predict after a nonempty <think> planning turn. Then call agrinet_rag_search at least once and at most three times to expand or verify unresolved candidate differences. Each query must address a new discriminative uncertainty; never merge classifier and retrieval scores. Tool responses are the only source of retrieval evidence. After the actual evidence, give the required final format; INSUFFICIENT_EVIDENCE is allowed only when the retained RAG evidence cannot support any specific conclusion.",
}

def e39_prompts() -> dict[str, str]:
    return dict(E39_TEACHER_PROMPTS)

def select_e39_coverage(rows: list[dict[str, Any]], *, seed: str = "e39-hcv-audit-v1") -> dict[str, str]:
    """Choose balanced private route witnesses without examining truth."""
    if len(rows) != 32: raise ValueError("E3.9 coverage needs exactly 32 audit rows")
    # Spread the three required routes across different task cells in each arm.
    cells = {"direct": ("open", "disease"), "classifier": ("option", "disease"), "rag": ("open", "pest")}
    result: dict[str, str] = {}
    for arm in ARMS:
        for route, (kind, domain) in cells.items():
            candidates = [row for row in rows if row.get("arm") == arm and row.get("question_type") == kind and row.get("task_domain") == domain]
            if not candidates: raise ValueError(f"E3.9 lacks coverage cell {arm}/{route}")
            chosen = min(candidates, key=lambda row: hashlib.sha256(f"{seed}:{arm}:{route}:{row['sample_id']}".encode()).hexdigest())
            result[str(chosen["sample_id"])] = f"{arm}:{route}"
    return result

def _think(content: str) -> str:
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>.*?</answer>\s*", content, re.DOTALL | re.I)
    if not match:
        raise ValueError("E3.9 final must be legal Hermes")
    return match.group(1).strip()

def _names(row: dict[str, Any], route: str, trace: list[dict[str, Any]]) -> set[str]:
    names = {str(x.get("name")) for x in row.get("public_options") or [] if isinstance(x, dict)}
    if route == "direct":
        return names
    for event in trace:
        response = event.get("response") if isinstance(event, dict) else None
        rendered = json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else ""
        for item in row.get("classifier", {}).get("top5", []):
            if isinstance(item, dict) and str(item.get("name")) in rendered:
                names.add(str(item["name"]))
    return names

def validate_e39_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    """Fail closed on the public HCV and tool-order contract."""
    route, answer = trajectory.get("route"), trajectory.get("answer")
    if route not in ROUTES or not isinstance(answer, str) or not is_final_answer(answer):
        raise ValueError("E3.9 final must be legal Hermes")
    thought = _think(answer)
    positions = [thought.find(field) for field in FIELDS]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise ValueError("E3.9 HCV fields are missing or unordered")
    sections = {field: thought[positions[i] + len(field): positions[i + 1] if i + 1 < len(FIELDS) else None].strip() for i, field in enumerate(FIELDS)}
    visual_items = [line for line in sections[FIELDS[0]].splitlines() if line.strip()]
    # Teachers often produce a compact, semicolon-delimited M1-style visual
    # paragraph. Count independently separated observations without requiring
    # a cosmetic bullet-list rendering.
    if len(visual_items) < 3 and len([part for part in re.split(r"[;.](?:\s|$)", sections[FIELDS[0]]) if part.strip()]) < 3:
        raise ValueError("E3.9 requires three visual observations")
    if not re.search(r"\b(low|medium|high)\b", sections[FIELDS[-1]], re.I):
        raise ValueError("E3.9 uncertainty is incomplete")
    rejected_items = re.findall(r"(?:^|\n)\s*[-*0-9.]", sections[FIELDS[4]])
    if len(rejected_items) < 2 and len(re.findall(r"(?:;|\.|\n)\s*[^;:.]+:", sections[FIELDS[4]])) < 2:
        raise ValueError("E3.9 requires two rejected alternatives")
    trace = trajectory.get("tool_trace") or []
    calls = [item.get("call", {}).get("name") for item in trace if isinstance(item, dict)]
    if route == "direct":
        if calls: raise ValueError("E3.9 Direct cannot call tools")
        if row.get("question_type") == "open":
            # Candidate names are checked against the public registry by private audit; here require a bounded list.
            hypotheses = re.findall(r"(?:^|\n)\s*[-*]\s*.+", sections[FIELDS[1]])
            if not 2 <= len(hypotheses) <= 3: raise ValueError("E3.9 Direct Open requires two or three hypotheses")
    elif route == "classifier":
        if not calls or calls[0] != "agrinet_classifier_predict" or calls.count("agrinet_classifier_expand") > 1:
            raise ValueError("E3.9 Classifier tool order is invalid")
        if "agrinet_rag_search" in calls: raise ValueError("E3.9 Classifier cannot call RAG")
    else:
        rag_positions = [i for i, name in enumerate(calls) if name == "agrinet_rag_search"]
        if not calls or calls[0] != "agrinet_classifier_predict" or not 1 <= len(rag_positions) <= 3 or rag_positions[0] == 0:
            raise ValueError("E3.9 RAG tool order is invalid")
    if row.get("question_type") == "option":
        body = re.search(r"<answer>(.*?)</answer>", answer, re.S | re.I).group(1).strip()
        if body != "INSUFFICIENT_EVIDENCE" and not re.fullmatch(r".+\s+—\s+[A-D]", body):
            raise ValueError("E3.9 Option answer must be class name — letter")

def make_e39_audit_source(rows: list[dict[str, Any]], *, coverage: dict[str, str]) -> list[dict[str, Any]]:
    """Make an immutable E3.9 projection of the exact balanced 32-row source.

    Coverage values are private-only route designations keyed by sample ID.
    """
    if len(rows) != 32 or not validate_source_rows(rows, require_full_coverage=False)["ready"]:
        raise ValueError("E3.9 requires the exact valid 32-image audit source")
    required = {f"{arm}:{route}" for arm in ARMS for route in ROUTES}
    if set(coverage.values()) != required or len(coverage) != len(required):
        raise ValueError("E3.9 coverage requires one private seed per arm and route")
    source_ids = {str(row.get("sample_id")) for row in rows}
    if set(coverage) - source_ids:
        raise ValueError("E3.9 coverage refers to a foreign sample")
    output = []
    for row in rows:
        sample_id = str(row["sample_id"])
        private = dict(row.get("private") or {})
        designation = coverage.get(sample_id)
        if designation:
            private["e39_route_coverage"] = designation.rsplit(":", 1)[1]
        output.append({**row, "e39_protocol": E39_PROTOCOL, "teacher_system_prompts": e39_prompts(),
                       "private": private})
    return output

def write_e39_audit_source(*, base_source: Path, coverage_path: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise ValueError("E3.9 audit source destination is immutable")
    rows = [json.loads(line) for line in base_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, dict): raise ValueError("E3.9 coverage sidecar must be an object")
    materialized = make_e39_audit_source(rows, coverage={str(k): str(v) for k, v in coverage.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in materialized: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(materialized), "source_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
            "protocol": E39_PROTOCOL, "training_eligible": False, "training_authorized": False, "sft_may_start": False}

def write_e39_full_source(*, candidate_source: Path, audit_ids: set[str], output: Path) -> dict[str, Any]:
    """Freeze the exact non-audit E3.9 source with no full-campaign coverage forcing."""
    if output.exists(): raise ValueError("E3.9 full source destination is immutable")
    rows = [json.loads(line) for line in candidate_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if str(row.get("sample_id")) not in audit_ids]
    if len(rows) != 1070 or len(selected) != 1038 or len(audit_ids) != 32:
        raise ValueError("E3.9 full source must exclude exactly 32 audit rows from 1070 candidates")
    if not validate_source_rows(selected, require_full_coverage=False)["ready"]:
        raise ValueError("E3.9 full source violates identity/classifier isolation")
    materialized = [{**row, "e39_protocol": E39_PROTOCOL, "teacher_system_prompts": e39_prompts()} for row in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in materialized: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(materialized), "protocol": E39_PROTOCOL,
            "source_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}

def write_e39_initial_manifest(*, campaign_id: str, source_path: Path, rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if output.exists(): raise ValueError("E3.9 manifest destination is immutable")
    if len(rows) != 32: raise ValueError("E3.9 audit manifest requires 32 rows")
    payload = {"schema_version": "agrinet.e39-hcv-cascade-manifest/v1", "protocol": E39_PROTOCOL,
        "campaign_id": campaign_id, "round": "R0", "source": str(source_path),
        "source_sha256": hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest(), "source_rows_expected": 32,
        "audit_only": True, "work_items": [{"work_id": f"R0:{row['sample_id']}:e39-hcv", "round": "R0", "sample_id": row["sample_id"],
            "image_group_id": row.get("image_group_id", row["image_sha256"]), "attempt_ordinal": 0,
            "predecessor_request_id": None, "route_progression": list(ROUTES)} for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "collection_controls": {"uncached_input_token_cap": 300000, "transport_image_max_side": 1024, "max_public_turns_per_route": 8, "max_rag_searches": 3, "reservation_uncached_tokens": {"generation": {"direct": 5000, "classifier": 8000, "rag": 12000}, "private_audit": 18000}},
        "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload

def write_e39_replenishment_manifest(*, summary: dict[str, Any], next_round: str, output: Path) -> dict[str, Any]:
    """Freeze an E3.9 R1/R2 manifest for genuine unresolved delivery only."""
    if output.exists(): raise ValueError("E3.9 replenishment destination is immutable")
    expected_prior = {"R1": "R0", "R2": "R1"}
    if next_round not in expected_prior or summary.get("round") != expected_prior[next_round]:
        raise ValueError("E3.9 recovery must proceed R0 -> R1 -> R2")
    controls = summary.get("collection_controls")
    if not isinstance(controls, dict) or controls.get("max_rag_searches") != 3:
        raise ValueError("E3.9 recovery has no frozen HCV controls")
    work = []
    for row in summary.get("rows", []):
        if row.get("delivery_status") not in DELIVERY_FAILURES: continue
        recovery = recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal", -1)), status=str(row["delivery_status"]), predecessor_request_id=str(row.get("request_id") or ""))
        if recovery.get("new_attempt"):
            work.append({**{key: row[key] for key in ("sample_id", "image_group_id", "route_progression")}, "work_id": f"{next_round}:{row['sample_id']}:e39-hcv", "round": next_round, **recovery})
    payload = {"schema_version": "agrinet.e39-hcv-cascade-manifest/v1", "protocol": E39_PROTOCOL, "campaign_id": summary.get("campaign_id"), "round": next_round, "audit_only": True, "work_items": work, "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False, "collection_controls": controls, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    coverage = sub.add_parser("select-coverage"); coverage.add_argument("--base-source", type=Path, required=True); coverage.add_argument("--output", type=Path, required=True)
    source = sub.add_parser("materialize-audit-source"); source.add_argument("--base-source", type=Path, required=True); source.add_argument("--coverage", type=Path, required=True); source.add_argument("--output", type=Path, required=True)
    full_source = sub.add_parser("materialize-full-source"); full_source.add_argument("--candidate-source", type=Path, required=True); full_source.add_argument("--audit-source", type=Path, required=True); full_source.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan-audit-r0"); plan.add_argument("--source", type=Path, required=True); plan.add_argument("--campaign-id", required=True); plan.add_argument("--output", type=Path, required=True)
    replenish = sub.add_parser("plan-replenishment"); replenish.add_argument("--summary", type=Path, required=True); replenish.add_argument("--next-round", choices=("R1", "R2"), required=True); replenish.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("audit-final-report"); report.add_argument("--source", type=Path, required=True); report.add_argument("--r0-summary", type=Path, required=True); report.add_argument("--r1-summary", type=Path, required=True); report.add_argument("--r2-summary", type=Path, required=True); report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "select-coverage":
        if args.output.exists(): raise ValueError("E3.9 coverage destination is immutable")
        rows = [json.loads(line) for line in args.base_source.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = select_e39_coverage(rows); args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream: json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"coverage_rows": len(value), "output": str(args.output)})); return 0
    if args.operation == "materialize-audit-source":
        print(json.dumps(write_e39_audit_source(base_source=args.base_source, coverage_path=args.coverage, output=args.output))); return 0
    if args.operation == "materialize-full-source":
        audit_ids = {json.loads(line)["sample_id"] for line in args.audit_source.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(json.dumps(write_e39_full_source(candidate_source=args.candidate_source, audit_ids=audit_ids, output=args.output))); return 0
    if args.operation == "plan-replenishment":
        value = write_e39_replenishment_manifest(summary=json.loads(args.summary.read_text(encoding="utf-8")), next_round=args.next_round, output=args.output)
        print(json.dumps({"round": value["round"], "rows": len(value["work_items"]), "output": str(args.output)})); return 0
    if args.operation == "audit-final-report":
        if args.output.exists(): raise ValueError("E3.9 audit report destination is immutable")
        rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = e39_layered_audit_report(source_rows=rows, summaries=[json.loads(path.read_text(encoding="utf-8")) for path in (args.r0_summary, args.r1_summary, args.r2_summary)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream: json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"protocol_gate_passed": value["protocol_gate_passed"], "output": str(args.output)})); return 0 if value["protocol_gate_passed"] else 2
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = write_e39_initial_manifest(campaign_id=args.campaign_id, source_path=args.source, rows=rows, output=args.output)
    print(json.dumps({"rows": len(manifest["work_items"]), "output": str(args.output)})); return 0

def e39_audit_report(*, source_rows: list[dict[str, Any]], terminals: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("sample_id")): row for row in source_rows}
    latest = {str(row.get("sample_id")): row for row in terminals}
    if len(by_id) != 32 or set(latest) != set(by_id): raise ValueError("E3.9 audit report requires closed 32-row terminals")
    seen = {arm: set() for arm in ARMS}
    for sample_id, terminal in latest.items():
        row = by_id[sample_id]; target = (row.get("private") or {}).get("e39_route_coverage")
        if terminal.get("winner") and terminal.get("final_route") == target:
            seen[row["arm"]].add(target)
    passed = all(seen[arm] == set(ROUTES) for arm in ARMS)
    return {"schema_version": "agrinet.e39-audit-final-report/v1", "protocol": E39_PROTOCOL, "rows": 32,
            "coverage_winners": {arm: sorted(seen[arm]) for arm in ARMS}, "protocol_gate_passed": passed,
            "full_campaign_expansion": "requires_separate_manifest" if passed else "not_authorized",
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}

def e39_layered_audit_report(*, source_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Overlay legal R0/R1/R2 recovery without losing terminal shortfalls."""
    if len(source_rows) != 32 or [item.get("round") for item in summaries] != ["R0", "R1", "R2"]:
        raise ValueError("E3.9 final audit needs its 32-row R0/R1/R2 lineage")
    by_id = {str(row.get("sample_id")): row for row in source_rows}
    if len(by_id) != 32: raise ValueError("E3.9 audit source has duplicate sample IDs")
    latest = {str(row.get("sample_id")): row for row in summaries[0].get("rows", [])}
    if set(latest) != set(by_id): raise ValueError("E3.9 R0 summary does not close source scope")
    for summary in summaries[1:]:
        expected = {sample_id for sample_id, row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed = {str(row.get("sample_id")) for row in summary.get("rows", [])}
        if observed != expected: raise ValueError("E3.9 recovery summary scope is not exact")
        for row in summary.get("rows", []): latest[str(row["sample_id"])] = row
    base = e39_audit_report(source_rows=source_rows, terminals=list(latest.values()))
    from collections import Counter
    terminal = Counter()
    routes = Counter()
    for row in latest.values():
        if row.get("delivery_status") in DELIVERY_FAILURES:
            terminal["delivery_shortfall"] += 1
        elif row.get("winner"):
            terminal["accepted"] += 1
        else:
            terminal[str(row.get("quality_status") or "quality_rejected")] += 1
        routes[str(row.get("final_route") or "none")] += 1
    return {**base, "rounds": ["R0", "R1", "R2"], "terminal_counts": dict(terminal), "final_routes": dict(routes),
            "delivery_shortfall_sample_ids": sorted(sample_id for sample_id, row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES),
            "full_campaign_expansion": "not_authorized" if not base["protocol_gate_passed"] else "requires_separate_manifest"}

def write_e39_full_manifest(*, audit_report: dict[str, Any], candidate_rows: list[dict[str, Any]], audit_ids: set[str], campaign_id: str, source_path: Path, output: Path) -> dict[str, Any]:
    if audit_report.get("protocol_gate_passed") is not True: raise ValueError("E3.9 audit gate has not passed")
    rows = [row for row in candidate_rows if str(row.get("sample_id")) not in audit_ids]
    if len(rows) != 1038 or len(candidate_rows) != 1070: raise ValueError("E3.9 full campaign must contain exact non-audit 1038 rows")
    bound = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(bound) != 1038 or {str(row.get("sample_id")) for row in bound} != {str(row.get("sample_id")) for row in rows} or any(row.get("e39_protocol") != E39_PROTOCOL for row in bound):
        raise ValueError("E3.9 full manifest must bind the materialized 1038-row E3.9 source")
    if output.exists(): raise ValueError("E3.9 full manifest destination is immutable")
    payload = {"schema_version": "agrinet.e39-hcv-cascade-manifest/v1", "protocol": E39_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source_path), "source_sha256": hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest(), "source_rows_expected": 1038, "audit_only": False,
        "work_items": [{"work_id": f"R0:{row['sample_id']}:e39-hcv", "round": "R0", "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]), "attempt_ordinal": 0, "predecessor_request_id": None, "route_progression": list(ROUTES)} for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False, "collection_controls": {"uncached_input_token_cap": 8000000, "transport_image_max_side": 1024, "max_public_turns_per_route": 8, "max_rag_searches": 3, "reservation_uncached_tokens": {"generation": {"direct": 5000, "classifier": 8000, "rag": 12000}, "private_audit": 18000}},
        "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload

if __name__ == "__main__":
    raise SystemExit(main())
