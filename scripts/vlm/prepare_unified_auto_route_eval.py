#!/usr/bin/env python3
"""Prepare public English dev/test manifests and classifier-only inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "<image> Task: identify the image with one canonical disease or pest name."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--add-smoke-only", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.add_smoke_only:
        if not output.is_dir():
            raise SystemExit(f"smoke addition requires existing evaluation artifact: {output}")
        destinations = (output / "smoke_en.jsonl", output / "smoke_classifier.jsonl")
        if any(path.exists() for path in destinations):
            raise SystemExit("refusing to overwrite existing smoke manifests")
        public_rows, classifier_rows = read_jsonl(output / "dev_en.jsonl"), read_jsonl(output / "dev_classifier.jsonl")
        strata = {}
        for public, classifier in zip(public_rows, classifier_rows):
            strata.setdefault((public["class_role"], public["task_domain"]), []).append((public, classifier))
        selected = [strata[key][index] for index in range(8) for key in sorted(strata) if index < len(strata[key])]
        write_jsonl(destinations[0], [item[0] for item in selected])
        write_jsonl(destinations[1], [item[1] for item in selected])
        print(json.dumps({"rows": len(selected), "strata": sorted(strata)}, ensure_ascii=False))
        return
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable evaluation artifact: {output}")
    output.mkdir(parents=True)
    dev_source = ROOT / "datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/dev.jsonl"
    test_source = ROOT / "outputs/artifacts/datasets/open-agri-v3-evaluation-v1/test_en.jsonl"
    sources = {"dev": read_jsonl(dev_source), "test": read_jsonl(test_source)}
    expected = {"dev": 812, "test": 1019}
    manifest = {"schema_version": "agrinet.unified-auto-route-eval/v1", "public_only": True, "splits": {}}
    for split, source_rows in sources.items():
        if len(source_rows) != expected[split]:
            raise ValueError(f"{split}: expected {expected[split]} rows")
        public_rows, classifier_rows = [], []
        for row in source_rows:
            image_path = (row.get("images") or [row.get("image_path")])[0]
            domain = (row.get("metadata") or {}).get("domain") or row.get("task_domain")
            public_rows.append({"id": row["id"], "image_path": image_path,
                                "image_sha256": row["image_sha256"], "language": "en",
                                "task_domain": domain, "question_type": "open", "question": QUESTION,
                                "class_role": row["class_role"], "evaluation_bucket": row["evaluation_bucket"]})
            classifier_rows.append({"image_id": row["id"],
                                    "image_path": str((ROOT / image_path).resolve()),
                                    "image_sha256": row["image_sha256"], "label": 0})
        if len({row["id"] for row in public_rows}) != len(public_rows):
            raise ValueError(f"{split}: duplicate IDs")
        public_path, classifier_path = output / f"{split}_en.jsonl", output / f"{split}_classifier.jsonl"
        write_jsonl(public_path, public_rows); write_jsonl(classifier_path, classifier_rows)
        manifest["splits"][split] = {"rows": len(public_rows), "public_manifest": display_path(public_path),
                                           "public_sha256": sha256(public_path),
                                           "classifier_manifest": display_path(classifier_path),
                                           "classifier_sha256": sha256(classifier_path)}
        if split == "dev":
            # Deterministic interleaving avoids the class-sorted dev prefix.
            strata = {}
            for public, classifier in zip(public_rows, classifier_rows):
                key = (public["class_role"], public["task_domain"])
                strata.setdefault(key, []).append((public, classifier))
            smoke_public, smoke_classifier = [], []
            for index in range(8):
                for key in sorted(strata):
                    if index < len(strata[key]):
                        public, classifier = strata[key][index]
                        smoke_public.append(public); smoke_classifier.append(classifier)
            write_jsonl(output / "smoke_en.jsonl", smoke_public)
            write_jsonl(output / "smoke_classifier.jsonl", smoke_classifier)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
