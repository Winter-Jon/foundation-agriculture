#!/usr/bin/env python3
"""Build fresh, class-targeted large-009 supplements from a freeze preflight.

The selector, rather than a heuristic pool count, is the source of shortages.
Every output image is excluded from held-out sets and every previous collection
attempt.  Direct targets retain the required same-image en/zh pairing.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from tools.rag_distill.build_hermes_1to1_collection_shard import contacted_hashes
from tools.rag_distill.build_hermes_1to1_large_plan import labels

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
PREFLIGHT = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-freeze-preflight-v2"
ISOLATION = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/isolation/forbidden_image_sha256.json"
DEST = ROOT / "outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-large-009/supplement"
CATALOG = ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.json"
IMAGE_ROOT = ROOT / "datasets/AgriNet-1K/all"

# Oversupply only fresh, currently unseen classes, so the final two-per-class
# cap keeps the freeze balanced while teacher rejection cannot starve coverage.
DIRECT_CLASS_COUNT = 12
DIRECT_ATTEMPTS_PER_CLASS = 2
RAG = {
    ("open", "en", "disease"): (14, 2),
    ("open", "en", "pest"): (20, 2),
    ("open", "zh", "pest"): (25, 2),
}


def cell(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    return tuple(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))  # type: ignore[return-value]


def catalog() -> dict[str, dict[str, str]]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = payload.get("classes") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"invalid class catalog: {CATALOG}")
    result = {}
    for row in rows:
        code = str(row.get("code") or "")
        if not code or not row.get("english_name") or not row.get("chinese_name"):
            continue
        result[code] = {
            "code": code, "english_name": str(row["english_name"]),
            "chinese_name": str(row["chinese_name"]),
            "task_domain": "disease" if code.startswith("N04") else "pest" if code.startswith("N05") else "",
        }
    return result


def fresh_images(code: str, *, forbidden: set[str], used: set[str]) -> list[tuple[str, str]]:
    result = []
    directory = IMAGE_ROOT / code
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = str(path.relative_to(ROOT)); digest = image_digest(image, ROOT)
        if digest not in forbidden and digest not in used:
            result.append((image, digest))
    return result


def select_unseen(*, existing: Counter[str], domain: str, count: int, attempts: int, classes: dict[str, dict[str, str]], forbidden: set[str], used: set[str]) -> list[tuple[str, str, str]]:
    choices = []
    for code, entry in sorted(classes.items()):
        if entry["task_domain"] != domain or existing[code] != 0:
            continue
        available = fresh_images(code, forbidden=forbidden, used=used)
        if len(available) >= attempts:
            choices.append((code, available))
    # Prefer ample reserve, then canonical code: deterministic and leaves no
    # ambiguity about why a particular class entered the supplement.
    choices.sort(key=lambda item: (-len(item[1]), item[0]))
    if len(choices) < count:
        raise ValueError(f"insufficient fresh unseen {domain} classes: {len(choices)} < {count}")
    selected = []
    for code, available in choices[:count]:
        for image, digest in available[:attempts]:
            selected.append((code, image, digest)); used.add(digest)
    return selected


def direct_target(*, code: str, image: str, digest: str, language: str, entry: dict[str, str], index: int) -> dict[str, Any]:
    return {
        "target_id": f"hermes-v2-large009-direct-open-{language}-pest-{index:03d}",
        "query_image": image, "image_sha256": digest, "canonical_class": code,
        "canonical_name": entry["chinese_name"] if language == "zh" else entry["english_name"],
        "task_domain": "pest", "question_type": "open", "language": language,
        "paired_image_id": digest, "generation_route": "direct_visual_comparison",
        "label_visible_to_teacher": True,
        "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
        "teacher_requirements": {"visible_candidates_min": 4, "neighbor_exclusions_min": 3, "preserve_long_comparison": True},
    }


def rag_target(*, code: str, image: str, digest: str, question: str, language: str, domain: str, entry: dict[str, str], index: int) -> dict[str, Any]:
    return {
        "target_id": f"hermes-v2-large009-rag-{question}-{language}-{domain}-{index:03d}",
        "query_image": image, "image_sha256": digest, "canonical_class": code,
        "canonical_name": entry["chinese_name"] if language == "zh" else entry["english_name"],
        "task_domain": domain, "question_type": question, "language": language,
        "generation_route": "blind_evidence", "label_visible_to_teacher": False,
        "uncertainty": "Uncertainty/不确定性: The image evidence is insufficient for a confidence estimate.",
        "teacher_requirements": {"pre_tool_visual_candidates_min": 3, "evidence_exclusions_min": 2, "public_tool_evidence_only": True},
    }


def main() -> int:
    if DEST.exists():
        raise RuntimeError(f"destination exists: {DEST}")
    direct_selected = read_jsonl(PREFLIGHT / "direct_selected.jsonl")
    rag_selected = read_jsonl(PREFLIGHT / "rag_selected.jsonl")
    entries = catalog()
    forbidden = set(json.loads(ISOLATION.read_text(encoding="utf-8"))) | contacted_hashes(COLLECTION)
    used: set[str] = set()

    direct_counts_en = Counter(str((row.get("metadata") or {}).get("canonical_class") or "") for row in direct_selected if cell(row) == ("open", "en", "pest"))
    direct_counts_zh = Counter(str((row.get("metadata") or {}).get("canonical_class") or "") for row in direct_selected if cell(row) == ("open", "zh", "pest"))
    direct_existing = Counter({code: max(direct_counts_en[code], direct_counts_zh[code]) for code in set(direct_counts_en) | set(direct_counts_zh)})
    direct_choices = select_unseen(existing=direct_existing, domain="pest", count=DIRECT_CLASS_COUNT, attempts=DIRECT_ATTEMPTS_PER_CLASS, classes=entries, forbidden=forbidden, used=used)
    direct = []
    for index, (code, image, digest) in enumerate(direct_choices, 1):
        direct.extend([direct_target(code=code, image=image, digest=digest, language=language, entry=entries[code], index=index) for language in ("en", "zh")])

    rag = []
    rag_summary = {}
    for target_cell, (class_count, attempts) in RAG.items():
        question, language, domain = target_cell
        counts = Counter(str((row.get("metadata") or {}).get("canonical_class") or "") for row in rag_selected if cell(row) == target_cell)
        choices = select_unseen(existing=counts, domain=domain, count=class_count, attempts=attempts, classes=entries, forbidden=forbidden, used=used)
        for index, (code, image, digest) in enumerate(choices, 1):
            rag.append(rag_target(code=code, image=image, digest=digest, question=question, language=language, domain=domain, entry=entries[code], index=index))
        rag_summary["/".join(target_cell)] = {"unseen_classes": class_count, "attempts": len(choices)}

    if {row["image_sha256"] for row in direct} & {row["image_sha256"] for row in rag}:
        raise ValueError("direct/rag target overlap")
    DEST.mkdir(parents=True)
    for route, rows in (("direct", direct), ("rag", rag)):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = f"{route}-{row['question_type']}-{row['language']}-{row['task_domain']}"
            groups.setdefault(key, []).append(row)
        for key, group in groups.items():
            write_jsonl(DEST / f"{key}.jsonl", group)
    report = {
        "schema_version": "agrinet.hermes-large009-targeted-supplement/v1",
        "source_preflight": str(PREFLIGHT.relative_to(ROOT)), "forbidden_hashes": len(forbidden),
        "direct_rows": len(direct), "rag_rows": len(rag), "unique_images": len(used),
        "direct_open_pest_unseen_classes": DIRECT_CLASS_COUNT, "rag": rag_summary,
        "direct_sha256": canonical_json_hash(direct), "rag_sha256": canonical_json_hash(rag),
        "training_authorized": False,
    }
    (DEST / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
