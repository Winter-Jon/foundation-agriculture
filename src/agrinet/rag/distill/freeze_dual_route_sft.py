from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from agrinet.data.sft_recovery import read_jsonl, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a reviewed dual-route RAG SFT corpus.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--single-image", action="store_true", help="Keep only the query image to match one <image> placeholder.")
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    data_path = args.destination / "data.jsonl"
    if args.single_image:
        frozen_rows = [{**row, "images": list(row.get("images") or [])[:1]} for row in source_rows]
        serialized = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in frozen_rows)
        import hashlib
        source_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    else:
        frozen_rows = source_rows
        serialized = None
        source_hash = sha256_file(args.source)
    if args.destination.exists():
        if not data_path.is_file() or sha256_file(data_path) != source_hash:
            raise RuntimeError(f"immutable artifact conflict: {args.destination}")
    else:
        args.destination.mkdir(parents=True)
        if serialized is None:
            shutil.copyfile(args.source, data_path)
        else:
            data_path.write_text(serialized, encoding="utf-8")

    rows = read_jsonl(data_path)
    routes: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    targets: set[str] = set()
    missing_images: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in sample_ids:
            duplicate_sample_ids.append(sample_id)
        sample_ids.add(sample_id)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        routes[str(metadata.get("generation_route") or "unknown")] += 1
        languages[str(metadata.get("language") or "unknown")] += 1
        domains[str(metadata.get("task_domain") or "unknown")] += 1
        strategies[str(metadata.get("strategy_id") or "unknown")] += 1
        modes[str(metadata.get("trajectory_mode") or "unknown")] += 1
        target_id = str(metadata.get("target_id") or "")
        if target_id:
            targets.add(target_id)
        for image in row.get("images") or []:
            path = Path(str(image))
            resolved = path if path.is_absolute() else args.repo_root / path
            if not resolved.is_file():
                missing_images.append({"sample_id": sample_id, "image": str(image)})

    stats = {
        "rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "unique_targets": len(targets),
        "languages": dict(sorted(languages.items())),
        "domains": dict(sorted(domains.items())),
        "generation_routes": dict(sorted(routes.items())),
        "strategies": dict(sorted(strategies.items())),
        "trajectory_modes": dict(sorted(modes.items())),
        "tool_rows": sum(bool(row.get("tools")) for row in rows),
        "missing_images": len(missing_images),
        "duplicate_sample_ids": len(duplicate_sample_ids),
    }
    manifest = {
        "schema_version": "agrinet.sft.frozen/v2",
        "artifact_id": args.artifact_id,
        "artifact_type": "datasets",
        "source_path": str(args.source),
        "data_sha256": source_hash,
        "immutable": True,
        "preserves_dual_route_provenance": True,
        "single_image": args.single_image,
        "statistics": stats,
    }
    (args.destination / "artifact.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (args.destination / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {
        "valid": not missing_images and not duplicate_sample_ids and len(rows) == len(sample_ids),
        "data_sha256": source_hash,
        "missing_images": missing_images,
        "duplicate_sample_ids": duplicate_sample_ids,
    }
    (args.destination / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.destination / "README.md").write_text(
        f"# {args.artifact_id}\n\nReviewed bilingual Oracle/Blind RAG tool-call SFT corpus.\n\n"
        f"Frozen without message rewriting from `{args.source}`. Do not edit `data.jsonl` in place.\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.destination), "sha256": source_hash, "statistics": stats, "valid": validation["valid"]}, ensure_ascii=False, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
