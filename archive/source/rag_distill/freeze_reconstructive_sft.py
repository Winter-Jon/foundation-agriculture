#!/usr/bin/env python3
"""Freeze a reconstructive Direct + Blind RAG corpus after hard validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest, query_image, validate_direct_rows, validate_hermes_1to1_rows, validate_rag_rows
from agrinet.data.sft_recovery import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, help="Formal 618 manifest required for the 1:1 freeze.")
    parser.add_argument("--lineage", type=Path, help="JSON lineage for legacy Direct audit and RAG reuse/new collection.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--stage", choices=("intermediate", "full", "hermes_1to1"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    direct, rag, diagnostic = read_jsonl(args.direct), read_jsonl(args.rag), read_jsonl(args.diagnostic_manifest)
    formal = read_jsonl(args.formal_manifest) if args.formal_manifest else []
    if args.stage == "hermes_1to1" and not args.formal_manifest:
        raise ValueError("--formal-manifest is required for the hermes_1to1 freeze")
    diagnostic_hashes = {str(row.get("image_sha256") or image_digest(query_image(row), args.repo_root)) for row in [*diagnostic, *formal]}
    for row in [*direct, *rag]:
        metadata = row.setdefault("metadata", {})
        metadata.setdefault("image_sha256", image_digest(query_image(row), args.repo_root))
    if args.stage == "hermes_1to1":
        combined = validate_hermes_1to1_rows(direct, rag, forbidden_hashes=diagnostic_hashes)
        direct_report, rag_report = {"valid": combined["valid"]}, {"valid": combined["valid"]}
        report = {"stage": args.stage, "one_to_one": combined, "diagnostic_manifest_sha256": canonical_json_hash(diagnostic), "formal_manifest_sha256": canonical_json_hash(formal), "valid": combined["valid"]}
    else:
        direct_report = validate_direct_rows(direct, stage=args.stage, image_hashes=diagnostic_hashes)
        direct_hashes = {str(row["metadata"]["image_sha256"]) for row in direct}
        rag_report = validate_rag_rows(rag, stage=args.stage, image_hashes=diagnostic_hashes | direct_hashes)
        report = {"stage": args.stage, "direct": direct_report, "rag": rag_report, "diagnostic_manifest_sha256": canonical_json_hash(diagnostic), "valid": direct_report["valid"] and rag_report["valid"]}
    if not report["valid"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    rows = [*direct, *rag]
    data_hash = canonical_json_hash(rows)
    data_path = args.destination / "data.jsonl"
    if args.destination.exists():
        if not data_path.is_file() or canonical_json_hash(read_jsonl(data_path)) != data_hash:
            raise RuntimeError(f"immutable artifact conflict: {args.destination}")
    else:
        args.destination.mkdir(parents=True)
        write_jsonl(data_path, rows)
    lineage = json.loads(args.lineage.read_text(encoding="utf-8")) if args.lineage else {}
    manifest = {"schema_version": "agrinet.reconstructive-sft-freeze/v2", "artifact_id": args.artifact_id, "stage": args.stage, "data_sha256": data_hash, "immutable": True, "initialization_model": "models/Qwen3-VL-4B-Instruct", "diagnostic_manifest_sha256": report["diagnostic_manifest_sha256"], "formal_manifest_sha256": report.get("formal_manifest_sha256"), "lineage": lineage, "validation": report}
    (args.destination / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (args.destination / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.destination), "data_sha256": data_hash, "valid": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
