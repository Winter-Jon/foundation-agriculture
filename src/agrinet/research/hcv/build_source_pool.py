#!/usr/bin/env python3
"""Build an image-isolated, truth-audited HCV preflight source pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[4]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import image_digest
from agrinet.research.shared.catalog import isolation_hashes
from agrinet.research.shared.catalog import ROOT, read_jsonl, write_jsonl
from agrinet.research.hcv.retrieval_preflight import explicit_isolation_hashes

SEED = "hcv-source-pool-v1-20260823"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=256)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--image-root", type=Path, default=ROOT / "datasets/AgriNet-1K/all")
    parser.add_argument("--exclude-manifests", type=Path, nargs="*", default=())
    return parser.parse_args()


def canonical_classes() -> list[dict[str, str]]:
    payload = json.loads((ROOT / "datasets/AgriNet-1K/wiki/base.json").read_text(encoding="utf-8"))
    unique: dict[str, dict[str, str]] = {}
    for key, item in (payload.get("description") or {}).items():
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or key.rsplit("::", 1)[-1]).strip()
        english = str(item.get("english_name") or "").strip()
        chinese = str(item.get("chinese_name") or "").strip()
        domain = "disease" if code.startswith("N04") else "pest" if code.startswith("N05") else ""
        if code and english and chinese and domain:
            unique[code] = {"code": code, "name": english, "chinese_name": chinese, "task_domain": domain}
    return [unique[key] for key in sorted(unique)]


def public_labels(classes: list[dict[str, str]], target: dict[str, str]) -> list[dict[str, str]]:
    others = [item for item in classes if item["task_domain"] == target["task_domain"] and item["code"] != target["code"]]
    ordered = sorted(others, key=lambda item: hashlib.sha256(f"{SEED}:label:{target['code']}:{item['code']}".encode()).hexdigest())
    labels = sorted([target, *ordered[:3]], key=lambda item: hashlib.sha256(f"{SEED}:option:{target['code']}:{item['code']}".encode()).hexdigest())
    return [{"code": item["code"], "name": item["name"], "chinese_name": item["chinese_name"], "task_domain": item["task_domain"]} for item in labels]


def forbidden_hashes(exclude_manifests: list[Path]) -> set[str]:
    forbidden = set(isolation_hashes(ROOT))
    explicit, _ = explicit_isolation_hashes(ROOT)
    forbidden.update(explicit)
    for manifest in exclude_manifests:
        for row in read_jsonl(manifest):
            value = str(row.get("image_sha256") or "")
            if value:
                forbidden.add(value)
    return forbidden


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.per_domain < 1 or args.max_per_class < 1:
        raise ValueError("per-domain and max-per-class must be positive")
    classes = canonical_classes()
    by_domain = {domain: [item for item in classes if item["task_domain"] == domain] for domain in ("disease", "pest")}
    forbidden = forbidden_hashes(list(args.exclude_manifests))
    selected: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    skipped: defaultdict[str, int] = defaultdict(int)
    scanned = 0
    for domain in ("disease", "pest"):
        candidates: list[tuple[str, dict[str, str], str]] = []
        for target in by_domain[domain]:
            paths = sorted((args.image_root / target["code"]).glob(f"{target['code']}_*.jpg"))
            paths = sorted(paths, key=lambda path: hashlib.sha256(f"{SEED}:{path}".encode()).hexdigest())
            for path in paths[: args.max_per_class]:
                scanned += 1
                try:
                    digest = image_digest(str(path), ROOT)
                except (FileNotFoundError, OSError):
                    skipped["missing_image"] += 1
                    continue
                if digest in forbidden:
                    skipped["image_isolation"] += 1
                    continue
                candidates.append((str(path.relative_to(ROOT)), target, digest))
        candidates.sort(key=lambda item: hashlib.sha256(f"{SEED}:{domain}:{item[0]}".encode()).hexdigest())
        for image, target, digest in candidates:
            if counts[domain] >= args.per_domain:
                break
            selected.append({
                "sample_id": f"hcv-source-{domain}-{len(selected)+1:05d}",
                "query_image": image, "images": [image], "image_sha256": digest,
                "task_domain": domain, "final_label": target["code"],
                "final_label_zh": target["chinese_name"],
                "candidate_labels": public_labels(by_domain[domain], target),
                "metadata": {"source_pool": "AgriNet-1K/all", "pool_seed": SEED, "task_domain": domain},
            })
            counts[domain] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)
    report = {
        "schema_version": "agrinet.hcv-source-pool/v1", "seed": SEED,
        "output": str(args.output), "requested_per_domain": args.per_domain,
        "rows": len(selected), "by_domain": dict(counts),
        "unique_image_hashes": len({row["image_sha256"] for row in selected}) == len(selected),
        "forbidden_hashes": len(forbidden), "scanned": scanned, "skipped": dict(skipped),
        "teacher_authorized": False, "sft_authorized": False,
    }
    report_path = args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False) + chr(10), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["rows"] == args.per_domain * 2 and report["unique_image_hashes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
