"""Public E3.5 candidate export with a separate private leakage audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agrinet.rag.e35_classifier_cascade import PRIVATE_TOKENS, campaign_report, hermes_candidate


def export_candidates(*, source_rows: list[dict[str, Any]], accepted: list[dict[str, Any]],
                      output_dir: Path) -> dict[str, Any]:
    """Export only one rewrite-audited winner per image, never private sidecars.

    ``accepted`` is intentionally supplied by an isolated private controller:
    it may include audit fields, but the public export projection below cannot.
    """
    root = Path(output_dir)
    paths = {name: root / name for name in ("data.jsonl", "lineage.jsonl", "private-leakage-audit.json", "campaign-report.json")}
    if any(path.exists() for path in paths.values()):
        raise ValueError("E3.5 export destinations are immutable")
    rows = {str(row.get("sample_id")): row for row in source_rows}
    if len(rows) != len(source_rows): raise ValueError("source has duplicate sample IDs")
    candidates, lineage, outcomes, audit_rows = [], [], [], []
    used_images: set[str] = set()
    for item in accepted:
        sample_id = str(item.get("sample_id") or "")
        row = rows.get(sample_id)
        if row is None or item.get("winner") is not True:
            raise ValueError("candidate export needs a frozen accepted winner")
        if row["image_sha256"] in used_images: raise ValueError("E3.5 export has more than one winner per image")
        candidate = hermes_candidate(row=row, trajectory=dict(item.get("trajectory") or {}),
                                     rewrite=dict(item.get("rewrite") or {}),
                                     rewrite_audit=str(item.get("rewrite_audit") or ""))
        rendered = json.dumps(candidate, ensure_ascii=False).casefold()
        leaks = sorted(token for token in PRIVATE_TOKENS if token in rendered)
        audit_rows.append({"sample_id": sample_id, "status": "reject" if leaks else "accept", "leaks": leaks})
        if leaks: raise ValueError("candidate export leaks private fields")
        used_images.add(row["image_sha256"]); candidates.append(candidate)
        lineage.append({"sample_id": sample_id, "image_sha256": row["image_sha256"],
                        "arm": row["arm"], "route": candidate["route"],
                        "parent_request_id": item.get("request_id"),
                        "rewrite_request_id": item.get("rewrite_request_id"),
                        "training_eligible": False, "training_authorized": False, "sft_may_start": False})
        outcomes.append({"arm": row["arm"], "canonical_class_code": row["canonical_class_code"],
                         "winner": True, "delivery_status": "delivered", "final_route": candidate["route"],
                         "quality_status": "accepted", "rag_result": item.get("rag_result")})
    report = campaign_report(outcomes, source_rows=source_rows)
    root.mkdir(parents=True, exist_ok=True)
    for name, values in (("data.jsonl", candidates), ("lineage.jsonl", lineage)):
        with paths[name].open("x", encoding="utf-8") as stream:
            for value in values: stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    with paths["private-leakage-audit.json"].open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "agrinet.e35-private-leakage-audit/v1", "rows": audit_rows,
                   "training_eligible": False, "training_authorized": False, "sft_may_start": False}, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    with paths["campaign-report.json"].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return {"rows": len(candidates), "paths": {name: str(path) for name, path in paths.items()},
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}
