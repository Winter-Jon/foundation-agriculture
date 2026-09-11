#!/usr/bin/env python3
"""Append-only operational memory and offline evidence for the E2/E3 eligibility loop."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a compact JSON source object or a JSONL candidate pool."""
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    rows = read_json(path).get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"source lacks rows: {path}")
    return [row for row in rows if isinstance(row, dict)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"immutable destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def initialize(*, root: Path, e2_contract: Path, e3_contract: Path) -> dict[str, Any]:
    locks = {
        "schema_version": "agrinet.e2-e3-training-eligibility-contract-locks/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contracts": [
            {"name": "e2-exploration", "path": str(e2_contract), "sha256": sha256(e2_contract)},
            {"name": "e3-adjacent-classfold", "path": str(e3_contract), "sha256": sha256(e3_contract)},
        ],
        "invariants": {
            "contracts_are_read_only": True, "strict_canary_history_is_read_only": True,
            "automatic_replay_allowed": False, "sft_may_start": False,
            "training_rows_required": 32, "rows_per_fixed_cell_required": 4,
        },
    }
    write_immutable(root / "contract-locks.json", locks)
    return locks


def validate_patch(*, root: Path, patch: Path) -> dict[str, Any]:
    locks, value = read_json(root / "contract-locks.json"), read_json(patch)
    if value.get("schema_version") != "agrinet.e2-e3-training-eligibility-contract-patch/v1":
        raise ValueError("patch schema is invalid")
    if not isinstance(value.get("patch_id"), str) or not value["patch_id"]:
        raise ValueError("patch_id is required")
    allowed = {item["sha256"] for item in locks["contracts"]}
    if value.get("base_contract_sha256") not in allowed:
        raise ValueError("patch does not bind a locked contract")
    prohibited = set(value.get("prohibited_changes") or [])
    required = {"historical_contract", "strict_canary_history", "old_ledgers_or_request_ids", "sft_authorization"}
    if not required.issubset(prohibited):
        raise ValueError("patch must explicitly preserve all immutable boundaries")
    patch_resolved = patch.resolve()
    prior = [item for item in sorted((root / "patches").glob("*.json"))
             if item.resolve() != patch_resolved] if (root / "patches").is_dir() else []
    if any(read_json(item).get("patch_id") == value["patch_id"] for item in prior):
        raise ValueError("patch_id already exists")
    parent = value.get("parent_patch_sha256")
    if parent is not None and parent not in {sha256(item) for item in prior}:
        raise ValueError("patch parent is not an existing immutable patch")
    return {"patch_id": value["patch_id"], "base_contract_sha256": value["base_contract_sha256"],
            "parent_patch_sha256": parent, "valid": True}


def rewrite_analysis(*, rewrite_root: Path) -> dict[str, Any]:
    files = sorted(rewrite_root.glob("rounds/r*/public/*/rewrite.json"))
    errors: Counter[str] = Counter(); routes: Counter[str] = Counter(); by_route: dict[str, Counter[str]] = {}
    for path in files:
        row = read_json(path); route = str(row.get("route") or "unknown")
        routes[route] += 1; bucket = by_route.setdefault(route, Counter())
        for error in row.get("errors") or []:
            errors[str(error)] += 1; bucket[str(error)] += 1
    return {
        "schema_version": "agrinet.e2-e3-training-eligibility-rewrite-analysis/v1",
        "rewrite_files": len(files), "error_counts": dict(sorted(errors.items())),
        "route_counts": dict(sorted(routes.items())),
        "errors_by_route": {route: dict(sorted(counts.items())) for route, counts in sorted(by_route.items())},
        "conclusion": "prompt_validator_contract_mismatch" if files and errors else "insufficient_evidence",
        "provider_retry_authorized": False, "training_eligible": False,
    }


def e3_status(*, run_root: Path) -> dict[str, Any]:
    folds = []
    for fold in range(3):
        status_path = run_root / f"vision-openagri-v3-known-vitl-e3-fold{fold}-v1" / "20260910T235014-5fc04488-a01" / "status.json"
        status = read_json(status_path) if status_path.is_file() else {"status": "missing"}
        folds.append({"fold": fold, "status_path": str(status_path), "status": status.get("status"),
                      "pid": status.get("pid"), "exit_code": status.get("exit_code")})
    terminal = all(item["status"] == "complete" and item["exit_code"] == 0 for item in folds)
    return {"schema_version": "agrinet.e2-e3-training-eligibility-e3-status/v1", "folds": folds,
            "all_formal_training_verified": terminal, "next_phase": "e3-evaluate" if terminal else "monitor-e3",
            "teacher_collection_authorized": False, "sft_may_start": False}


def e3_aggregate(*, run_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Aggregate terminal E3 fold evidence, refusing partial training as proof."""
    status = e3_status(run_root=run_root)
    if not status["all_formal_training_verified"]:
        raise ValueError("E3 aggregation requires three successful terminal formal runs")
    assignment_path, folds_path = artifact_root / "class_assignments.jsonl", artifact_root / "folds.json"
    if not assignment_path.is_file() or not folds_path.is_file():
        raise ValueError("E3 aggregate lacks frozen adjacency evidence")
    assignment_rows, fold_summary = read_jsonl(assignment_path), read_json(folds_path)
    if (fold_summary.get("bridge_contract") or {}).get("all_held_out_classes_have_training_neighbour") is not True:
        raise ValueError("E3 frozen graph does not guarantee training-side RAG bridges")
    assignments = {str(row.get("canonical_class_code") or ""): row for row in assignment_rows if isinstance(row, dict)}
    if len(assignments) != 107:
        raise ValueError("E3 frozen class assignment coverage is incomplete")
    folds = []
    for fold in range(3):
        base = artifact_root / f"fold-{fold}"
        history_path, checkpoint, labels = base / "classifier/classifier_dev_metrics.json", base / "classifier/model_best.pth.tar", base / "label_map.json"
        test_metrics = base / "classifier/metrics_test_known.json"
        if not history_path.is_file() or not checkpoint.is_file() or not labels.is_file() or not test_metrics.is_file():
            raise ValueError(f"E3 fold {fold} lacks final metric/checkpoint/label/test evaluation evidence")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or len(history) != 50:
            raise ValueError(f"E3 fold {fold} does not have 50 epoch metrics")
        best = max(history, key=lambda item: float(item.get("macro_f1", -1)))
        test = read_json(test_metrics)
        if not isinstance(test.get("macro_f1"), (int, float)):
            raise ValueError(f"E3 fold {fold} test-known metrics are invalid")
        train = read_jsonl(base / "manifests/train.jsonl")
        heldout = read_jsonl(base / "manifests/holdout_train_candidate.jsonl")
        label_values = json.loads(labels.read_text(encoding="utf-8"))
        if not isinstance(label_values, list):
            raise ValueError(f"E3 fold {fold} label map is invalid")
        label_codes = {str(item.get("canonical_class_code") or "") for item in label_values if isinstance(item, dict)}
        heldout_codes = {str(item.get("canonical_class_code") or "") for item in heldout}
        train_codes = {str(item.get("canonical_class_code") or "") for item in train}
        train_images = {str(item.get("image_sha256") or "") for item in train}
        heldout_images = {str(item.get("image_sha256") or "") for item in heldout}
        if not heldout_codes or heldout_codes & label_codes or heldout_codes & train_codes or heldout_images & train_images:
            raise ValueError(f"E3 fold {fold} violates held-out class/image isolation")
        witnesses = {code: assignments.get(code, {}).get("training_neighbour_witnesses") for code in heldout_codes}
        if any(not isinstance(value, list) or not value for value in witnesses.values()):
            raise ValueError(f"E3 fold {fold} lacks frozen training-side RAG witness")
        folds.append({"fold": fold, "epochs": len(history), "best_epoch": best.get("epoch"),
                      "best_macro_f1": best.get("macro_f1"), "best_disease_macro_f1": best.get("disease_macro_f1"),
                      "best_pest_macro_f1": best.get("pest_macro_f1"), "checkpoint": str(checkpoint),
                      "checkpoint_sha256": sha256(checkpoint), "label_map_sha256": sha256(labels),
                      "test_known_macro_f1": test["macro_f1"], "test_known_balanced_accuracy": test.get("balanced_accuracy"),
                      "training_label_count": len(label_codes), "held_out_class_count": len(heldout_codes),
                      "held_out_image_count": len(heldout), "held_out_isolation_verified": True,
                      "rag_training_neighbour_witnesses_verified": len(witnesses)})
    mean = {key: sum(float(row[key]) for row in folds) / len(folds) for key in ("best_macro_f1", "best_disease_macro_f1", "best_pest_macro_f1")}
    return {"schema_version": "agrinet.e2-e3-training-eligibility-e3-aggregate/v1", "folds": folds, "mean_best_metrics": mean,
            "generalization_evidence_complete": True, "teacher_collection_authorized": False,
            "training_eligible": False, "sft_may_start": False}


def training_qualification(*, source: Path, conversion_gate: Path) -> dict[str, Any]:
    """Apply the frozen 32-row/eight-cell qualification bar after conversion."""
    source_rows = read_json(source).get("rows")
    gate = read_json(conversion_gate)
    if not isinstance(source_rows, list) or not isinstance(gate.get("accepted"), list):
        raise ValueError("source or conversion gate lacks rows")
    by_group = {str(row.get("image_group_id")): row for row in source_rows if isinstance(row, dict)}
    accepted = gate["accepted"]
    groups = [str(row.get("image_group_id")) for row in accepted if isinstance(row, dict)]
    errors = []
    if len(accepted) != 32 or len(groups) != 32 or len(set(groups)) != 32:
        errors.append("requires_exactly_32_unique_converted_rows")
    missing = [group for group in groups if group not in by_group]
    if missing:
        errors.append("conversion_contains_unknown_image_group")
    cells = Counter()
    for group in groups:
        row = by_group.get(group)
        if row:
            cell = "-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))
            cells[cell] += 1
    required_cells = {f"{question}-{language}-{domain}" for question in ("open", "option") for language in ("en", "zh") for domain in ("disease", "pest")}
    if set(cells) != required_cells or any(cells[cell] != 4 for cell in required_cells):
        errors.append("requires_four_rows_in_each_fixed_cell")
    return {
        "schema_version": "agrinet.e2-e3-training-eligibility-qualification/v1",
        "source": str(source), "source_sha256": sha256(source),
        "conversion_gate": str(conversion_gate), "conversion_gate_sha256": sha256(conversion_gate),
        "accepted_rows": len(accepted), "cell_counts": dict(sorted(cells.items())),
        "errors": errors, "training_eligible": not errors, "sft_may_start": False,
    }


def pilot_preflight(*, e3_aggregate_path: Path | None, patch: Path, source: Path,
                    risk_exception_patch: Path | None = None) -> dict[str, Any]:
    """Validate a future 8-cell x 1 risk-exception pilot before any provider intent."""
    aggregate = read_json(e3_aggregate_path) if e3_aggregate_path is not None else {}
    patch_value = read_json(patch)
    e3_complete = aggregate.get("generalization_evidence_complete") is True
    exception = read_json(risk_exception_patch) if risk_exception_patch is not None else None
    if not e3_complete:
        if not isinstance(exception, dict) or exception.get("patch_id") != "0002-e3-running-pilot-risk-exception":
            raise ValueError("E2 pilot requires completed E3 evidence or the explicit E3-running risk exception")
        if exception.get("parent_patch_sha256") != sha256(patch):
            raise ValueError("E3-running risk exception must bind the approved rewrite patch")
    if patch_value.get("patch_id") != "0001-rewrite-structure-v2":
        raise ValueError("E2 pilot requires the approved rewrite v2 patch")
    rows = read_rows(source)
    if len(rows) != 8:
        raise ValueError("E2 pilot requires exactly eight new source rows")
    groups = [str(row.get("image_group_id") or "") for row in rows if isinstance(row, dict)]
    cells = Counter("-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain")) for row in rows if isinstance(row, dict))
    required = {f"{question}-{language}-{domain}" for question in ("open", "option") for language in ("en", "zh") for domain in ("disease", "pest")}
    if len(groups) != 8 or len(set(groups)) != 8 or not all(groups) or set(cells) != required or any(cells[cell] != 1 for cell in required):
        raise ValueError("E2 pilot source must provide one unique image in each fixed cell")
    return {
        "schema_version": "agrinet.e2-e3-training-eligibility-pilot-preflight/v1",
        "e3_aggregate": str(e3_aggregate_path) if e3_aggregate_path else None,
        "e3_aggregate_sha256": sha256(e3_aggregate_path) if e3_aggregate_path else None,
        "e3_generalization_evidence_complete": e3_complete,
        "risk_exception_patch": str(risk_exception_patch) if risk_exception_patch else None,
        "risk_exception_patch_sha256": sha256(risk_exception_patch) if risk_exception_patch else None,
        "contract_patch": str(patch), "contract_patch_sha256": sha256(patch),
        "source": str(source), "source_sha256": sha256(source), "cell_counts": dict(sorted(cells.items())),
        "rounds": ["R0", "R1", "R2"], "new_lineage_required": True,
        "automatic_replay_allowed": False, "strict_canary_overridden": False,
        "provider_request_authorized": not e3_complete and exception is not None,
        "training_eligible": False, "sft_may_start": False,
    }


def freeze_pilot_source(*, pool: Path, prior_sources: list[Path]) -> dict[str, Any]:
    """Select one deterministic, historically isolated candidate in each pilot cell."""
    candidates = read_rows(pool)
    historical: dict[str, set[str]] = {key: set() for key in ("image_group_id", "source_group_id", "near_duplicate_group_id")}
    for source in prior_sources:
        rows = read_rows(source)
        for row in rows:
            if isinstance(row, dict):
                for key in historical:
                    value = str(row.get(key) or "")
                    if value:
                        historical[key].add(value)
    required = {f"{question}-{language}-{domain}" for question in ("open", "option") for language in ("en", "zh") for domain in ("disease", "pest")}
    selected = []
    for cell in sorted(required):
        options = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            row_cell = "-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))
            if row_cell != cell:
                continue
            if any(not str(row.get(key) or "") or str(row.get(key)) in historical[key] for key in historical):
                continue
            options.append(row)
        if not options:
            raise ValueError(f"pilot pool has no isolated candidate for {cell}")
        selected.append(sorted(options, key=lambda row: str(row.get("sample_id") or row.get("image_group_id")))[0])
    for key in historical:
        values = [str(row[key]) for row in selected]
        if len(values) != len(set(values)):
            raise ValueError(f"pilot source duplicates {key}")
    return {"schema_version": "agrinet.e2-e3-training-eligibility-pilot-source/v1", "rows": selected,
            "prior_sources": [str(path) for path in prior_sources],
            "prior_source_sha256": {str(path): sha256(path) for path in prior_sources},
            "automatic_replay_allowed": False, "training_eligible": False, "sft_may_start": False}


def freeze_campaign_source(*, pool: Path, prior_sources: list[Path], selection: str = "stable") -> dict[str, Any]:
    """Freeze the formal 32-row source with four isolated groups per cell.

    This is intentionally separate from the eight-row pilot freezer: callers
    cannot accidentally treat an evidence-only pilot as a training candidate
    campaign, and the same image/source/near-duplicate exclusions apply.
    """
    if selection not in {"stable", "lower_pattern_first", "higher_confidence_first", "p6_first"}:
        raise ValueError("campaign source selection is unsupported")
    candidates = read_rows(pool)
    identity_keys = ("image_group_id", "source_group_id", "near_duplicate_group_id")
    blocked: dict[str, set[str]] = {key: set() for key in identity_keys}
    for source in prior_sources:
        for row in read_rows(source):
            for key in identity_keys:
                value = str(row.get(key) or "")
                if value:
                    blocked[key].add(value)
    required = [(question, language, domain) for question in ("open", "option")
                for language in ("en", "zh") for domain in ("disease", "pest")]
    selected: list[dict[str, Any]] = []
    used: dict[str, set[str]] = {key: set() for key in identity_keys}
    for question, language, domain in required:
        eligible = [row for row in candidates
                    if (str(row.get("question_type")), str(row.get("language")), str(row.get("task_domain"))) == (question, language, domain)
                    and all(str(row.get(key) or "")
                            and str(row.get(key)) not in blocked[key]
                            and str(row.get(key)) not in used[key] for key in identity_keys)]
        def order(row: dict[str, Any]) -> tuple[float, str]:
            pattern = str((row.get("private") or {}).get("candidate_pattern") or "P99")
            try:
                rank = int(pattern.removeprefix("P"))
            except ValueError:
                rank = 99
            sample = str(row.get("sample_id") or row.get("image_group_id"))
            if selection == "lower_pattern_first":
                return (float(rank), sample)
            if selection == "p6_first":
                # P6 is intentionally selected first only for a separately
                # authorized class-holdout arm; the remaining patterns retain
                # deterministic lower-pattern fallback ordering.
                return (-1.0 if pattern == "P6" else float(rank), sample)
            if selection == "higher_confidence_first":
                top5 = ((row.get("prediction") or {}).get("top5") or [])
                score = top5[0].get("score") if top5 and isinstance(top5[0], dict) else None
                confidence = float(score) if isinstance(score, (int, float)) else -1.0
                return (-confidence, sample)
            return (0.0, sample)
        choices = sorted(eligible, key=order)[:4]
        if len(choices) != 4:
            raise ValueError(f"campaign pool lacks four isolated candidates for {question}-{language}-{domain}")
        selected.extend(choices)
        for row in choices:
            for key in identity_keys:
                used[key].add(str(row[key]))
    cells = Counter("-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))
                    for row in selected)
    if len(selected) != 32 or len({str(row["image_group_id"]) for row in selected}) != 32 or any(value != 4 for value in cells.values()):
        raise ValueError("formal campaign source must contain 32 isolated rows and four per fixed cell")
    return {"schema_version": "agrinet.e2-e3-training-eligibility-campaign-source/v1", "rows": selected,
            "prior_sources": [str(path) for path in prior_sources],
            "prior_source_sha256": {str(path): sha256(path) for path in prior_sources},
            "selection": selection,
            "cell_counts": dict(sorted(cells.items())), "automatic_replay_allowed": False,
            "training_eligible": False, "sft_may_start": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize"); init.add_argument("--root", type=Path, required=True); init.add_argument("--e2-contract", type=Path, required=True); init.add_argument("--e3-contract", type=Path, required=True)
    patch = sub.add_parser("validate-patch"); patch.add_argument("--root", type=Path, required=True); patch.add_argument("--patch", type=Path, required=True)
    analysis = sub.add_parser("analyze-rewrites"); analysis.add_argument("--rewrite-root", type=Path, required=True); analysis.add_argument("--output", type=Path, required=True)
    status = sub.add_parser("e3-status"); status.add_argument("--run-root", type=Path, required=True); status.add_argument("--output", type=Path, required=True)
    aggregate = sub.add_parser("e3-aggregate"); aggregate.add_argument("--run-root", type=Path, required=True); aggregate.add_argument("--artifact-root", type=Path, required=True); aggregate.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser("training-qualification"); qualify.add_argument("--source", type=Path, required=True); qualify.add_argument("--conversion-gate", type=Path, required=True); qualify.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot-preflight"); pilot.add_argument("--e3-aggregate", type=Path); pilot.add_argument("--patch", type=Path, required=True); pilot.add_argument("--risk-exception-patch", type=Path); pilot.add_argument("--source", type=Path, required=True); pilot.add_argument("--output", type=Path, required=True)
    pilot_source = sub.add_parser("freeze-pilot-source"); pilot_source.add_argument("--pool", type=Path, required=True); pilot_source.add_argument("--prior-source", type=Path, action="append", required=True); pilot_source.add_argument("--output", type=Path, required=True)
    campaign_source = sub.add_parser("freeze-campaign-source"); campaign_source.add_argument("--pool", type=Path, required=True); campaign_source.add_argument("--prior-source", type=Path, action="append", required=True); campaign_source.add_argument("--selection", choices=("stable", "lower_pattern_first", "higher_confidence_first", "p6_first"), default="stable"); campaign_source.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize": result = initialize(root=args.root, e2_contract=args.e2_contract, e3_contract=args.e3_contract)
    elif args.command == "validate-patch": result = validate_patch(root=args.root, patch=args.patch)
    elif args.command == "analyze-rewrites":
        result = rewrite_analysis(rewrite_root=args.rewrite_root); write_immutable(args.output, result)
    elif args.command == "e3-status":
        result = e3_status(run_root=args.run_root); write_immutable(args.output, result)
    elif args.command == "e3-aggregate":
        result = e3_aggregate(run_root=args.run_root, artifact_root=args.artifact_root); write_immutable(args.output, result)
    elif args.command == "training-qualification":
        result = training_qualification(source=args.source, conversion_gate=args.conversion_gate); write_immutable(args.output, result)
    elif args.command == "pilot-preflight":
        result = pilot_preflight(e3_aggregate_path=args.e3_aggregate, patch=args.patch, source=args.source, risk_exception_patch=args.risk_exception_patch); write_immutable(args.output, result)
    elif args.command == "freeze-pilot-source":
        result = freeze_pilot_source(pool=args.pool, prior_sources=args.prior_source); write_immutable(args.output, result)
    else:
        result = freeze_campaign_source(pool=args.pool, prior_sources=args.prior_source, selection=args.selection); write_immutable(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
