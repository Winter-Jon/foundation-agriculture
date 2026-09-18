"""Derive a route-balanced immutable SFT artifact from unified v4 data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from agrinet.data.io import DataError, write_jsonl_atomic

PARENT_ID = "agrinet-e343-three-route-sft-v4-unified-auto-route"
ARTIFACT_ID = "agrinet-e343-three-route-sft-v5-balanced"
ROUTES = ("direct", "classifier", "rag")
PER_ROUTE = 165


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selection_key(item: dict[str, Any]) -> tuple[str, str]:
    sample_id = str(item.get("sample_id") or "")
    if not sample_id:
        raise DataError("lineage item has no sample_id")
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), sample_id


def build(*, parent: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise DataError(f"refuse existing immutable destination: {output}")
    manifest = yaml.safe_load((parent / "artifact.yaml").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != PARENT_ID or manifest.get("data_sha256") != sha256(parent / "data.jsonl"):
        raise DataError("parent artifact identity or data hash mismatch")
    rows = read_jsonl(parent / "data.jsonl")
    lineage = read_jsonl(parent / "lineage.jsonl")
    by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if len(by_id) != len(rows) or len(lineage) != len(rows):
        raise DataError("parent rows or lineage are not uniquely aligned")
    grouped: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTES}
    for item in lineage:
        route = str(item.get("route") or "")
        sample_id = str(item.get("sample_id") or "")
        if route not in grouped or sample_id not in by_id:
            raise DataError(f"invalid lineage row: route={route} sample_id={sample_id}")
        grouped[route].append(item)
    selected = [item for route in ROUTES for item in sorted(grouped[route], key=selection_key)[:PER_ROUTE]]
    counts = Counter(str(item["route"]) for item in selected)
    expected = {route: PER_ROUTE for route in ROUTES}
    if dict(counts) != expected or len(selected) != PER_ROUTE * len(ROUTES):
        raise DataError(f"balanced selection failed: {dict(counts)}")
    selected_ids = {str(item["sample_id"]) for item in selected}
    if len(selected_ids) != len(selected):
        raise DataError("selection contains duplicate sample IDs")
    output.mkdir(parents=True)
    selected_rows = [row for row in rows if str(row["sample_id"]) in selected_ids]
    if len(selected_rows) != len(selected):
        raise DataError("selected row count mismatch")
    write_jsonl_atomic(output / "data.jsonl", selected_rows)
    write_jsonl_atomic(output / "lineage.jsonl", selected)
    for filename in ("exclusions.jsonl", "evaluation-isolation.json"):
        (output / filename).write_bytes((parent / filename).read_bytes())
    stats = {"rows": len(selected), "routes": {**expected, "refusal": 0}, "refusal_rows": 0,
             "selection": "per-route deterministic sha256(sample_id) without replacement",
             "parent_rows": len(rows), "training_eligible": True, "training_authorized": True, "sft_may_start": True}
    admission = {"artifact_id": ARTIFACT_ID, "parent_artifact_id": PARENT_ID,
                 "decision": "Route-balanced retry requested after Direct-route collapse in EXP-20260917-001.",
                 "conversion_policy": "Select exactly 165 rows per route; preserve each selected row unchanged.",
                 **stats}
    for filename, value in (("statistics.json", stats), ("source-admission.json", admission)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact = {"schema_version": "agrinet.sft.frozen/v3-route-balanced", "artifact_id": ARTIFACT_ID,
                "artifact_type": "datasets", "immutable": True, "parent_artifact_id": PARENT_ID,
                "parent_manifest_sha256": sha256(parent / "artifact.yaml"), "statistics": stats,
                "data_sha256": sha256(output / "data.jsonl"), "lineage_sha256": sha256(output / "lineage.jsonl"),
                "exclusions_sha256": sha256(output / "exclusions.jsonl"), **{key: manifest[key] for key in ("protocol_version", "system_prompt_sha256", "tool_schema_sha256")},
                "training_eligible": True, "training_authorized": True, "sft_may_start": True}
    (output / "artifact.yaml").write_text(yaml.safe_dump(artifact, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "README.md").write_text(f"# {ARTIFACT_ID}\n\n495 English rows: Direct, Classifier, and RAG each contribute 165 unchanged rows.\n", encoding="utf-8")
    return {"artifact_dir": str(output), **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(parent=args.parent, output=args.output), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
