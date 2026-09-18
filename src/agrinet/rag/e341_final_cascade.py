"""Build the immutable E3.38/E3.41 sample-level cascade conservation report."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agrinet.rag.e322_presample import digest, rows
from agrinet.rag.e328_classifier_full import FLAGS

ROUND_FILES = ("r0.json", "q1.json", "r1.json", "q1-r1.json", "r2.json", "q1-r2.json")
SAFE = {"semantic_correct", "future_reject"}


def _write(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"E3.41 final cascade report is immutable: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _latest(campaign: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for shard in sorted((campaign / "shards").glob("shard-*")):
        for name in ROUND_FILES:
            path = shard / "outcomes" / name
            if path.is_file():
                for outcome in json.loads(path.read_text()).get("outcomes") or []:
                    latest[outcome["sample_id"]] = outcome
    return latest


def _stratify(source: list[dict[str, Any]], latest: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    dimensions = {
        "arm": lambda row: str((row.get("private") or {}).get("arm") or "unknown"),
        "question_type": lambda row: str(row.get("question_type") or "unknown"),
        "domain": lambda row: str(row.get("domain") or "unknown"),
        "truth_code": lambda row: str((row.get("private") or {}).get("truth_code") or "unknown"),
        "fold": lambda row: str((row.get("classifier") or {}).get("held_out_fold") or "unknown"),
        "e327_overlap": lambda row: "overlap" if row.get("e327_overlap") else "non_overlap",
    }
    for name, key in dimensions.items():
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        for row in source:
            outcome = latest[row["sample_id"]]
            bucket = buckets[key(row)]
            bucket["rows"] += 1
            bucket[str(outcome.get("disposition"))] += 1
            bucket["safe_terminal"] += int(outcome.get("disposition") in SAFE)
        result[name] = {value: dict(counts) for value, counts in sorted(buckets.items())}
    return result


def build(*, e338_report: Path, e341_source: Path, campaign: Path, audit: Path, gate: Path, output: Path) -> dict[str, Any]:
    classifier = json.loads(e338_report.read_text())
    source = rows(e341_source)
    latest = _latest(campaign)
    audit_value = json.loads(audit.read_text())
    gate_value = json.loads(gate.read_text())
    if len(source) != 261 or set(latest) != {row["sample_id"] for row in source}:
        raise ValueError("E3.41 cascade coverage invalid")
    if not audit_value.get("artifact_audit_passed") or not gate_value.get("partial_rag_input_gate_passed"):
        raise ValueError("E3.41 audited safe subset gate is not passed")
    safe = {sid: value for sid, value in latest.items() if value.get("disposition") in SAFE}
    residual = {sid: value for sid, value in latest.items() if sid not in safe}
    if any(value.get("delivery_status") != "delivered" for value in latest.values()):
        raise ValueError("E3.41 terminal delivery incomplete")
    direct_winners, direct_residuals = 530, 15
    classifier_disposition = Counter(classifier.get("final_disposition") or {})
    rag_disposition = Counter(value.get("disposition") for value in latest.values())
    conservation = {
        "all_images": 1070,
        "direct_winners_frozen": direct_winners,
        "classifier_queue": int(classifier.get("rows") or 0),
        "direct_residuals_frozen": direct_residuals,
        "all_images_holds": direct_winners + int(classifier.get("rows") or 0) + direct_residuals == 1070,
        "classifier": {"rows": int(classifier.get("rows") or 0), **dict(classifier_disposition), "future_rag": int(classifier.get("future_rag_count") or 0), "residuals": len(classifier.get("classifier_terminal_shortfalls") or [])},
        "rag": {"input": len(source), **dict(rag_disposition), "safe_terminal": len(safe), "residuals": len(residual), "safe_plus_residual_holds": len(safe) + len(residual) == len(source)},
    }
    value = {
        "schema_version": "agrinet.e341-final-cascade-report/v1",
        "classifier_protocol": classifier.get("protocol"),
        "rag_protocol": gate_value.get("protocol"),
        "immutable_inputs": {
            "e338_final_report": {"path": str(e338_report), "sha256": digest(e338_report)},
            "e341_source": {"path": str(e341_source), "sha256": digest(e341_source)},
            "e341_audit": {"path": str(audit), "sha256": digest(audit)},
            "e341_gate": {"path": str(gate), "sha256": digest(gate)},
        },
        "conservation": conservation,
        "classifier_source_distribution": classifier.get("source_distribution"),
        "classifier_truth_classes": classifier.get("truth_classes"),
        "classifier_folds": classifier.get("folds"),
        "classifier_checkpoint_usage": classifier.get("checkpoint_usage"),
        "classifier_residuals": classifier.get("classifier_terminal_shortfalls"),
        "rag_safe_terminal_samples": sorted(safe),
        "rag_residuals": [
            {"sample_id": sid, "disposition": value.get("disposition"), "contract_error": value.get("contract_error"), "audit_contract_error": value.get("audit_contract_error"), **FLAGS}
            for sid, value in sorted(residual.items())
        ],
        "rag_stratification": _stratify(source, latest),
        "rag_top3_metrics": json.loads((campaign / "final-report.json").read_text()).get("top3_metrics"),
        "rag_overlap_groups": json.loads((campaign / "final-report.json").read_text()).get("overlap_groups"),
        "provider_reconciliation": {
            "classifier": classifier.get("budget"),
            "rag_global_intents": audit_value.get("global_intents"),
            "rag_ledger_intents": audit_value.get("ledger_intents"),
            "rag_reject_calls": audit_value.get("reject_calls"),
            "rag_candidate_slot_loss_samples": audit_value.get("candidate_slot_loss_samples"),
        },
        "gates": {
            "classifier_full_hard_gate": False,
            "classifier_partial_safe_input_gate": True,
            "rag_full_hard_gate": bool(gate_value.get("rag_gate_passed")),
            "rag_partial_safe_terminal_gate": bool(gate_value.get("partial_rag_input_gate_passed")),
            "no_reject": audit_value.get("reject_calls") == 0,
            "no_training": all(value is False for value in FLAGS.values()),
        },
        "next_action": "interview_user; do_not_run_reject_conversion_sft_or_training",
        **FLAGS,
    }
    _write(output, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e338-report", "e341-source", "campaign", "audit", "gate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = build(e338_report=args.e338_report, e341_source=args.e341_source, campaign=args.campaign, audit=args.audit, gate=args.gate, output=args.output)
    print(json.dumps({"all_images_holds": value["conservation"]["all_images_holds"], "rag_safe_terminal": value["conservation"]["rag"]["safe_terminal"], "rag_residuals": value["conservation"]["rag"]["residuals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
