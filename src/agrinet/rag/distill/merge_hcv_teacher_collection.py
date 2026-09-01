#!/usr/bin/env python3
"""Merge audited HCV collections into an exact, fail-closed freeze candidate."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from agrinet.data.rebuild_sft import CELLS
from agrinet.rag.distill.audit_hcv_teacher_collection import audit_row
from agrinet.rag.distill.catalog_and_isolation import read_jsonl, write_jsonl

def cell(row: dict[str, Any]) -> str:
    m = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return "/".join(str(m.get(k) or "") for k in ("question_type", "language", "task_domain"))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--accepted", type=Path, nargs="+", required=True)
    p.add_argument("--private-audit", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    ns = p.parse_args()
    if len(ns.accepted) != len(ns.private_audit): raise SystemExit("accepted/private-audit inputs must be paired")
    private = {}
    for path in ns.private_audit:
        for row in read_jsonl(path):
            sid = str(row.get("sample_id") or "")
            if not sid or sid in private: raise SystemExit(f"duplicate private audit ID: {sid}")
            private[sid] = row
    candidates, rejected, seen_ids, seen_hashes = [], [], set(), set()
    for path in ns.accepted:
        for row in read_jsonl(path):
            sid = str(row.get("sample_id") or "")
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            image_hash = str(meta.get("image_sha256") or "")
            errors = ["missing_private_audit"] if sid not in private else audit_row(row, private[sid])
            if sid in seen_ids: errors.append("duplicate_sample_id")
            if image_hash and image_hash in seen_hashes: errors.append("duplicate_image_hash")
            if errors:
                rejected.append({"sample_id": sid, "cell": cell(row), "errors": errors}); continue
            seen_ids.add(sid); seen_hashes.add(image_hash); candidates.append(row)
    selected, counts = [], Counter()
    for c in CELLS:
        key = "/".join(c); rows = [r for r in candidates if cell(r) == key]
        selected.extend(rows[:4]); counts[key] = min(4, len(rows))
    ids = {str(r.get("sample_id") or "") for r in selected}
    hashes = {str((r.get("metadata") or {}).get("image_sha256") or "") for r in selected}
    report = {"schema_version": "agrinet.hcv-teacher-collection-merge/v1", "input_accepted_rows": sum(len(read_jsonl(p)) for p in ns.accepted), "valid_candidate_rows": len(candidates), "selected_rows": len(selected), "valid_by_cell": dict(sorted(counts.items())), "rejected_candidates": rejected, "invariants": {"exact_32_rows": len(selected) == 32, "eight_cells_four_each": dict(counts) == {"/".join(c): 4 for c in CELLS}, "unique_sample_ids": len(ids) == len(selected), "unique_image_hashes": len(hashes) == len(selected) and "" not in hashes, "private_audit_complete": all(str(r.get("sample_id") or "") in private for r in selected)}}
    report["freeze_authorized"] = all(report["invariants"].values())
    ns.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(ns.output_dir / "accepted_candidates.jsonl", selected)
    write_jsonl(ns.output_dir / "private_audit.jsonl", [private[str(r["sample_id"])] for r in selected])
    (ns.output_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False)); return 0 if report["freeze_authorized"] else 1

if __name__ == "__main__": raise SystemExit(main())
