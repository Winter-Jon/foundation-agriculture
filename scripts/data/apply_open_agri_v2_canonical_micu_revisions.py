#!/usr/bin/env python3
"""Apply user-authorized Micu revise taxonomy recommendations.

Manual-review recommendations remain unchanged. Applying revisions reopens the
formal approval gate, so an already approved taxonomy requires explicit intent.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agrinet.research.open_agri_v2_canonical.micu_revision_apply import (
    REVISION_REVIEWER,
    apply_revisions,
)
from agrinet.research.open_agri_v2_canonical.registry import build_registry_rows, registry_digest
from agrinet.research.open_agri_v2_canonical.taxonomy_audit import (
    canonical_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

DEFAULT_DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
SOURCE_CATALOG = ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--recommendations", type=Path)
    parser.add_argument(
        "--reopen-approval",
        action="store_true",
        help=(
            "allow rewriting an approved taxonomy; this resets its formal "
            "approval gate to pending_manual_review"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def render_review(rows: Iterable[dict[str, Any]], domain: str) -> str:
    title = "病害" if domain == "disease" else "虫害"
    lines = [
        f"# OpenAgri canonical-v1 当前规范方案：{title}", "",
        "本文件呈现 current canonical proposal；历史标签请看同级 ../legacy/ 文件。",
        "请在 review_decisions.jsonl 中记录决定；manual_review 仅保留审计备注，未采纳改名。", "",
        "| Canonical code | Source codes | English | Chinese | Scientific name | Life stage | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["domain"] != domain:
            continue
        lines.append(
            f"| {row['canonical_code']} | {', '.join(row['source_codes'])} | "
            f"{row['canonical_english_name']} | {row['canonical_chinese_name']} | "
            f"{row['scientific_name'] or '—'} | {row['life_stage']} | [ ] approve / [ ] revise / [ ] reject |"
        )
    lines.extend(["", "## Review checklist", "", "- canonical English and Chinese display names", "- scientific name and lifecycle wording", "- merge decision and lower-code retention", "- rationale against the separate legacy evidence", ""])
    return "\n".join(lines) + "\n"


def update_catalog(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    prior = {str(item["canonical_code"]): item for item in read_jsonl(path)}
    updated = []
    for row in rows:
        item = dict(prior[str(row["canonical_code"])])
        item.update(row)
        updated.append(item)
    return updated


def update_current(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "agrinet.canonical-current-proposal/v1",
        "taxonomy_version": row["taxonomy_version"], "domain": row["domain"], **row["current"],
    } for row in rows]


def update_decisions(path: Path, rows: list[dict[str, Any]], applied: list[dict[str, Any]], timestamp: str) -> list[dict[str, Any]]:
    by_code = {str(row["canonical_code"]): row for row in rows}
    changes = {str(item["canonical_code"]): item["changes"] for item in applied}
    result = []
    for item in read_jsonl(path):
        code = str(item["canonical_code"])
        row = by_code[code]
        replacement = {**item, "current_proposal": {
            "english": row["canonical_english_name"], "chinese": row["canonical_chinese_name"],
            "scientific_name": row["scientific_name"], "life_stage": row["life_stage"],
        }}
        if code in changes:
            replacement.update({
                "decision": "revise", "reviewer": REVISION_REVIEWER, "reviewed_at": timestamp,
                "notes": f"Applied explicit Micu revise fields: {canonical_json(changes[code])}",
            })
        result.append(replacement)
    return result


def update_class_split(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    labels = {str(row["canonical_code"]): row for row in rows}
    result = []
    for item in read_jsonl(path):
        row = labels[str(item["canonical_class_code"])]
        result.append({
            **item, "canonical_english_name": row["canonical_english_name"],
            "canonical_chinese_name": row["canonical_chinese_name"], "life_stage": row["life_stage"],
        })
    return result


def cumulative_field_changes(rows: list[dict[str, Any]], audit: Path) -> list[dict[str, Any]]:
    """Compare final values with the frozen pre-adoption Micu request inputs."""
    prior = {
        str(item["canonical_code"]): item["input"]["class_record"]
        for item in read_jsonl(audit / "requests.jsonl")
    }
    current = {str(item["canonical_code"]): item for item in rows}
    result = []
    for code, before in sorted(prior.items()):
        row = current[code]
        changed = {}
        for report_name, prior_name, current_name in (
            ("english", "canonical_english_name", "canonical_english_name"),
            ("chinese", "canonical_chinese_name", "canonical_chinese_name"),
            ("scientific_name", "scientific_name", "scientific_name"),
            ("life_stage", "life_stage", "life_stage"),
        ):
            if before[prior_name] != row[current_name]:
                changed[report_name] = {"before": before[prior_name], "after": row[current_name]}
        if changed:
            result.append({"canonical_code": code, "changes": changed})
    return result


def cumulative_aliases(rows: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report every Micu alias now represented in the final registry."""
    by_code = {str(row["canonical_code"]): row for row in rows}
    result = []
    for recommendation in recommendations:
        code = str(recommendation["canonical_code"])
        for proposed in recommendation.get("aliases") or []:
            matching = [
                alias for alias in by_code[code]["aliases"]
                if alias["value"] == proposed["value"]
                and alias["language"] == proposed["language"]
                and alias["alias_type"] == proposed["alias_type"]
            ]
            if matching:
                result.append({
                    "canonical_code": code, "value": proposed["value"], "language": proposed["language"],
                    "alias_type": proposed["alias_type"], "scope": matching[0]["scope"],
                    "model_recommended_scope": proposed["recommended_scope"],
                })
    return result


def main() -> None:
    options = args()
    dataset = options.dataset_root.resolve()
    taxonomy = dataset / "taxonomy"
    audit = taxonomy / "review/micu_alias_audit_v1"
    recommendations = options.recommendations or audit / "recommendations.jsonl"
    approval_path = taxonomy / "approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("status") == "approved" and not options.reopen_approval:
        raise SystemExit(
            "Refusing to modify an approved taxonomy. Pass --reopen-approval "
            "only when an explicit new review cycle is authorized."
        )
    # Reconstruct from the frozen source catalogue on every apply. This keeps
    # all legacy answer aliases needed by historical SFT preview artifacts,
    # rather than depending on a previous partial adoption state.
    old_rows = build_registry_rows(read_jsonl(SOURCE_CATALOG))
    recommendation_rows = read_jsonl(recommendations)
    rows, applied, skipped, appended_aliases = apply_revisions(old_rows, recommendation_rows)
    manual_codes = sorted(item["canonical_code"] for item in recommendation_rows if item.get("verdict") == "manual_review")
    timestamp = now()
    final_changes = cumulative_field_changes(rows, audit)
    final_aliases = cumulative_aliases(rows, recommendation_rows)
    report = {
        "schema_version": "agrinet.canonical-label-micu-revision-adoption/v1",
        "applied_at": timestamp, "reviewer": REVISION_REVIEWER,
        "source_recommendations": str(recommendations), "applied_rows": len(final_changes),
        "applied_field_changes": sum(len(item["changes"]) for item in final_changes),
        "appended_aliases": len(final_aliases),
        "appended_alias_details": final_aliases,
        "this_execution": {"field_changes": sum(len(item["changes"]) for item in applied), "aliases_added": len(appended_aliases)},
        "skipped_display_revisions": skipped,
        "manual_review_unchanged": manual_codes,
        "registry_sha256_before": registry_digest(old_rows), "registry_sha256_after": registry_digest(rows),
        "approval_gate": "unchanged_pending_manual_review", "applied": final_changes,
    }
    if options.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    write_jsonl_atomic(taxonomy / "canonical_label_registry.jsonl", rows)
    write_jsonl_atomic(taxonomy / "current/canonical_proposals.jsonl", update_current(rows))
    write_jsonl_atomic(taxonomy / "review/review_decisions.jsonl", update_decisions(taxonomy / "review/review_decisions.jsonl", rows, applied, timestamp))
    write_jsonl_atomic(dataset / "catalog/canonical_catalog.jsonl", update_catalog(rows, dataset / "catalog/canonical_catalog.jsonl"))
    write_jsonl_atomic(dataset / "manifests/class_split.jsonl", update_class_split(rows, dataset / "manifests/class_split.jsonl"))
    for domain in ("disease", "pest"):
        (taxonomy / f"review/current/{domain}.md").write_text(render_review(rows, domain), encoding="utf-8")
    approval["registry_sha256"] = report["registry_sha256_after"]
    approval["status"] = "pending_manual_review"
    approval["approved_rows"] = 0
    write_json_atomic(approval_path, approval)
    summary_path = dataset / "manifests/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["registry_sha256"] = report["registry_sha256_after"]
    write_json_atomic(summary_path, summary)
    write_json_atomic(audit / "revision_adoption.json", report)
    write_jsonl_atomic(audit / "appended_aliases.jsonl", final_aliases)
    print(canonical_json({key: report[key] for key in ("applied_rows", "applied_field_changes", "appended_aliases", "manual_review_unchanged", "registry_sha256_after")}))


if __name__ == "__main__":
    main()
