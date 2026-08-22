#!/usr/bin/env python3
"""Create immutable, fresh, class-balanced 2x attempt targets for Hermes v2."""
from __future__ import annotations

import argparse, json, random
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import CELLS, canonical_json_hash, image_digest, load_hermes_1to1_specification
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from tools.rag_distill.build_hermes_1to1_collection_shard import contacted_hashes

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.json"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"
SEED = 20260818


def labels(classes: dict[str, dict[str, str]], code: str, domain: str, letter: str) -> list[dict[str, str]]:
    alternatives = [classes[item] for item in sorted(classes) if classes[item]["task_domain"] == domain and item != code][:3]
    alternatives.insert("ABCD".index(letter), classes[code])
    return [{"code": item["code"], "name": item["english_name"], "chinese_name": item["chinese_name"], "task_domain": item["task_domain"]} for item in alternatives]


def fresh(code: str, forbidden: set[str]) -> list[tuple[str, str]]:
    result = []
    for path in sorted((IMAGE_ROOT / code).glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
        if digest not in forbidden:
            result.append((image, digest))
    return result


def build_targets(*, classes: dict[str, dict[str, str]], forbidden: set[str], attempts: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = load_hermes_1to1_specification()
    if attempts != 2 * int(spec["direct_per_cell"]):
        raise ValueError("large plan requires exactly 2x the final per-cell target")
    # Hash a class directory once.  The same catalog is used by twelve cells,
    # so rescanning it per cell turns a bounded preflight into needless I/O.
    eligible = {code: fresh(code, forbidden) for code in classes}
    direct: list[dict[str, Any]] = []; rag: list[dict[str, Any]] = []; used: set[str] = set(); shortages: dict[str, int] = {}
    # 35 classes x four independent images gives two expected final accepts/class
    # at the current approximately 50% acceptance rate.
    for question_type in ("open", "option"):
        for domain in ("disease", "pest"):
            # The other route is globally image-isolated.  Prefer classes with
            # enough reserve for both routes rather than consuming a sparse
            # four-image class in Direct and starving Blind RAG later.
            codes = [code for code in classes if classes[code]["task_domain"] == domain and len(eligible[code]) >= 8]
            random.Random(f"{SEED}:direct:{question_type}:{domain}").shuffle(codes)
            codes = sorted(codes, key=lambda code: -len(eligible[code]))[:int(spec["direct_min_classes"])]
            selected = []
            for round_index in range(4):
                for code in codes:
                    if len(selected) >= attempts: break
                    candidate = next(((image, digest) for image, digest in eligible[code] if digest not in used), None)
                    if candidate is None: continue
                    selected.append((code, *candidate)); used.add(candidate[1])
                if len(selected) >= attempts: break
            shortages[f"direct/{question_type}/{domain}"] = attempts - len(selected)
            for index, (code, image, digest) in enumerate(selected, 1):
                for language in ("en", "zh"):
                    target = {"target_id": f"hermes-v2-large-direct-{question_type}-{language}-{domain}-{index:03d}", "query_image": image, "image_sha256": digest, "canonical_class": code, "canonical_name": classes[code]["chinese_name"] if language == "zh" else classes[code]["english_name"], "task_domain": domain, "question_type": question_type, "language": language, "paired_image_id": digest, "generation_route": "direct_visual_comparison", "label_visible_to_teacher": True, "uncertainty": spec["uncertainty"], "teacher_requirements": {"visible_candidates_min": spec["direct_min_candidates"], "neighbor_exclusions_min": spec["direct_min_negative_exclusions"], "preserve_long_comparison": True}}
                    if question_type == "option":
                        target["correct_option"] = "ABCD"[(index - 1) % 4]; target["candidate_labels"] = labels(classes, code, domain, target["correct_option"])
                    direct.append(target)
    for question_type, language, domain in CELLS:
        codes = [code for code in classes if classes[code]["task_domain"] == domain and len(eligible[code]) >= 8]
        random.Random(f"{SEED}:rag:{question_type}:{language}:{domain}").shuffle(codes)
        # Earlier cells consume from the same globally isolated image pool.
        # Prefer the largest remaining class reserve, retaining seeded order as
        # the deterministic tie-breaker, so sparse classes cannot starve later cells.
        codes = sorted(codes, key=lambda code: -sum(digest not in used for _, digest in eligible[code]))[:int(spec["rag_min_classes"])]
        selected = []
        for round_index in range(4):
            for code in codes:
                if len(selected) >= attempts: break
                candidate = next(((image, digest) for image, digest in eligible[code] if digest not in used), None)
                if candidate is None: continue
                selected.append((code, *candidate)); used.add(candidate[1])
            if len(selected) >= attempts: break
        shortages[f"rag/{question_type}/{language}/{domain}"] = attempts - len(selected)
        for index, (code, image, digest) in enumerate(selected, 1):
            target = {"target_id": f"hermes-v2-large-rag-{question_type}-{language}-{domain}-{index:03d}", "query_image": image, "image_sha256": digest, "canonical_class": code, "canonical_name": classes[code]["chinese_name"] if language == "zh" else classes[code]["english_name"], "task_domain": domain, "question_type": question_type, "language": language, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "uncertainty": spec["uncertainty"], "teacher_requirements": {"pre_tool_visual_candidates_min": spec["rag_min_candidates"], "evidence_exclusions_min": spec["rag_min_evidence_exclusions"], "public_tool_evidence_only": True}}
            if question_type == "option":
                target["correct_option"] = "ABCD"[(index - 1) % 4]; target["candidate_labels"] = labels(classes, code, domain, target["correct_option"])
            rag.append(target)
    return direct, rag, {"shortages": shortages, "used_hashes": len(used)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--forbidden-hashes", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--attempts-per-cell", type=int, default=140)
    args = parser.parse_args()
    if args.destination.exists(): raise RuntimeError(f"destination exists: {args.destination}")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entries = catalog.get("classes") if isinstance(catalog, dict) else None
    if not isinstance(entries, list): raise ValueError(f"canonical catalog lacks classes: {CATALOG}")
    classes = {str(row["code"]): {"code": str(row["code"]), "english_name": str(row["english_name"]), "chinese_name": str(row["chinese_name"]), "task_domain": "disease" if str(row["code"]).startswith("N04") else "pest" if str(row["code"]).startswith("N05") else ""} for row in entries if row.get("code") and row.get("english_name") and row.get("chinese_name")}
    forbidden = set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8"))) | contacted_hashes(args.collection_root)
    direct, rag, extra = build_targets(classes=classes, forbidden=forbidden, attempts=args.attempts_per_cell)
    report = {"schema_version": "agrinet.hermes-large-plan/v1", "seed": SEED, "attempts_per_cell": args.attempts_per_cell, "forbidden_hashes": len(forbidden), "direct_rows": len(direct), "rag_rows": len(rag), "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag), **extra, "training_authorized": False}
    args.destination.mkdir(parents=True)
    write_jsonl(args.destination / "direct_targets.jsonl", direct); write_jsonl(args.destination / "rag_targets.jsonl", rag)
    for route, rows in (("direct", direct), ("rag", rag)):
        for question_type, language, domain in CELLS:
            cell_rows = [row for row in rows if (row["question_type"], row["language"], row["task_domain"]) == (question_type, language, domain)]
            write_jsonl(args.destination / f"{route}-{question_type}-{language}-{domain}.jsonl", cell_rows)
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False)); return 0 if not any(extra["shortages"].values()) else 1


if __name__ == "__main__": raise SystemExit(main())
