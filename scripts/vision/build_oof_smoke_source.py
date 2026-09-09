#!/usr/bin/env python3
"""Build the private/public 32-image v2 smoke source from merged OOF rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CELLS = tuple((q, lang, domain) for q in ("open", "option") for lang in ("en", "zh") for domain in ("disease", "pest"))
REGISTRY_SHA = "4fa6426203c64631315a2d754f3d53d6c3a7639e26c1c8617b2cbf92caf00cb8"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded_hashes(paths: list[Path]) -> set[str]:
    result = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            value = str(row.get("image_sha256") or row.get("sha256") or "")
            if value:
                result.add(value)
    return result


def historical_group_exclusions(*, blocked_hashes: set[str], images_manifest: Path) -> tuple[set[str], set[str]]:
    """Expand contacted identities to source/pHash groups asserted by v3 metadata.

    Historical ledgers are SHA-only.  A group peer must not re-enter a fresh
    smoke set merely because its exact SHA was not listed in that older ledger.
    """
    sources: set[str] = set(); near_duplicates: set[str] = set()
    for row in read_jsonl(images_manifest):
        if str(row.get("image_sha256") or "") not in blocked_hashes:
            continue
        source = str(Path(str(row.get("source_path") or "")).resolve())
        sources.add("source:" + hashlib.sha256(source.encode()).hexdigest())
        near_duplicates.add("phash:" + hashlib.sha256(str(row.get("phash") or "").encode()).hexdigest())
    return sources, near_duplicates


def order(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def pattern(row: dict) -> str:
    truth = row["canonical_class_code"]
    top = row["prediction"]["top5"]
    ranks = [i for i, item in enumerate(top, 1) if item["code"] == truth]
    if not ranks:
        return "P5"
    rank = ranks[0]
    if rank == 1 and float(top[0]["score"]) >= 0.8:
        return "P1"
    if rank == 1:
        return "P2"
    if rank <= 3:
        return "P3"
    return "P4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--class-split", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-exclusions", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--seed", default="micu-classifier-hcv-v2-oof-smoke-v1")
    args = parser.parse_args()
    classes = {str(row["canonical_class_code"]): row for row in read_jsonl(args.class_split)}
    roles = {code: row["class_role"] for code, row in classes.items()}
    blocked = excluded_hashes(args.exclude)
    blocked_sources, blocked_near_duplicates = historical_group_exclusions(
        blocked_hashes=blocked, images_manifest=args.images_manifest)
    pool = []
    for row in read_jsonl(args.merged):
        code = str(row.get("canonical_class_code") or "")
        if (roles.get(code) != "known" or row.get("image_sha256") in blocked
                or row.get("source_group_id") in blocked_sources
                or row.get("near_duplicate_group_id") in blocked_near_duplicates):
            continue
        if row.get("prediction", {}).get("registry_sha256") != REGISTRY_SHA:
            raise ValueError("OOF prediction registry mismatch")
        pool.append(row)
    used: set[str] = set(); used_groups: set[str] = set(); selected: list[dict] = []; counts: Counter[str] = Counter()
    for q, lang, domain in CELLS:
        candidates = sorted((r for r in pool if r["domain"] == domain and r["image_sha256"] not in used and r["near_duplicate_group_id"] not in used_groups), key=lambda r: order(args.seed, f"{q}:{lang}:{r['image_sha256']}"))
        for row in candidates:
            if counts[(q, lang, domain)] >= 4:
                break
            digest = row["image_sha256"]; used.add(digest); used_groups.add(row["near_duplicate_group_id"]); counts[(q, lang, domain)] += 1
            code = row["canonical_class_code"]; truth = classes[code]
            public_name = truth["canonical_english_name"] if lang == "en" else truth["canonical_chinese_name"]
            question = ("Identify the disease or pest shown in the image." if lang == "en" else "请识别图像中的病害或害虫。")
            public = {"A": truth}
            if q == "option":
                distractors = [r for r in classes.values() if r["class_role"] == "known" and r["domain"] == domain and r["canonical_class_code"] != code]
                options = sorted([truth] + sorted(distractors, key=lambda r: order(args.seed, f"{digest}:{r['canonical_class_code']}"))[:3], key=lambda r: order(args.seed, f"option:{digest}:{r['canonical_class_code']}"))
                public_options = [{"label": chr(65 + i), "code": r["canonical_class_code"], "name": r["canonical_english_name"], "name_zh": r["canonical_chinese_name"]} for i, r in enumerate(options)]
                correct = next(item["label"] for item in public_options if item["code"] == code)
                question += " Choose one option: " + "; ".join(f"{x['label']}: {x['name'] if lang == 'en' else x['name_zh']}" for x in public_options)
            else:
                public_options, correct = [], None
            sample = {"sample_id": "oof-smoke-" + digest[:20], "image_group_id": "image:" + digest, "source_group_id": row["source_group_id"], "near_duplicate_group_id": row["near_duplicate_group_id"], "image_sha256": digest, "image_path": row["image_path"], "question": question, "question_type": q, "language": lang, "task_domain": domain, "dataset_version": "open_agri_v3", "split": "train_candidate", "prediction": row["prediction"], "public_options": public_options, "private": {"truth_code": code, "class_role": "known", "target_pattern": pattern(row), "sampling_bucket": "hard" if pattern(row) in {"P3", "P4", "P5"} else "ordinary", "status": "fresh", "intervention": False, "simulated_unknown": False, "correct_option": correct, "truth_name": public_name, "mae_saw_related_unlabeled": "unknown"}}
            selected.append(sample)
    shortages = {f"{q}-{lang}-{domain}": 4 - counts[(q, lang, domain)] for q, lang, domain in CELLS if counts[(q, lang, domain)] < 4}
    if shortages or len(selected) != 32:
        raise ValueError(f"smoke source shortage: {shortages}, rows={len(selected)}")
    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected), encoding="utf-8")
    records = []
    for item in read_jsonl(args.images_manifest):
        digest = str(item.get("image_sha256") or "")
        if digest not in blocked:
            continue
        source = str(item.get("source_path") or item.get("image_path") or "")
        phash = str(item.get("phash") or "")
        records.append({
            "image_sha256": digest,
            "image_group_id": "image:" + digest,
            "source_group_id": "source:" + hashlib.sha256(source.encode()).hexdigest(),
            "near_duplicate_group_id": "phash:" + hashlib.sha256(phash.encode()).hexdigest(),
            "status": "contacted",
        })
    # A complete ledger may contain historical hashes not present in v3; keep
    # those as image-only records only if no v3 identity can be reconstructed.
    known = {row["image_sha256"] for row in records}
    records.extend({"image_sha256": digest, "image_group_id": "image:" + digest,
                    "source_group_id": "legacy-source:" + digest,
                    "near_duplicate_group_id": "legacy-near:" + digest,
                    "status": "contacted"}
                   for digest in sorted(blocked - known))
    args.output_exclusions.parent.mkdir(parents=True, exist_ok=True)
    args.output_exclusions.write_text(json.dumps({"schema_version": "agrinet.hcv-classifier-exclusions/v1", "complete": True, "provenance": {"source": "historical isolation ledgers plus formal evaluation identities", "inputs": [str(p) for p in args.exclude], "group_expansion": "source_group_id_and_exact_phash_group"}, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected), "per_cell": {"-".join(k): v for k, v in counts.items()}, "patterns": dict(Counter(r["private"]["target_pattern"] for r in selected)), "blocked_hashes": len(blocked), "blocked_source_groups": len(blocked_sources), "blocked_near_duplicate_groups": len(blocked_near_duplicates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
