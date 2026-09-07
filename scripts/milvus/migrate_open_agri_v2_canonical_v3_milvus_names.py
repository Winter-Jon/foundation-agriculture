#!/usr/bin/env python3
"""Copy the local Milvus index and replace only canonical display-name fields.

The migration deliberately preserves all retrieval-bearing and public-evidence
fields: entry IDs, source codes, descriptions, aliases, reference images,
vectors, and similar-class order.  The target is a parallel collection, leaving
the legacy collection untouched and reversible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agrinet.rag.milvus_tools.ingest_agrinet_wiki import create_class_collection, create_image_collection
from agrinet.research.open_agri_v2_canonical.registry import Registry, load_registry

DATASET = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v3"
LITE_DB = REPO_ROOT / "outputs/milvus/agrinet_wiki_lite.db"
SIMILARITY_REPORT = REPO_ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.json"
SOURCE_CLASS_COLLECTION = "agrinet_wiki_siglip2_classes"
SOURCE_IMAGE_COLLECTION = "agrinet_wiki_siglip2_images"
TARGET_CLASS_COLLECTION = "open_agri_v3_classes"
TARGET_IMAGE_COLLECTION = "open_agri_v3_images"

NAME_FIELDS = {
    "english_name", "chinese_name", "similar_english_classes", "similar_chinese_classes",
}


def collection_fields(client: MilvusClient, collection: str) -> set[str]:
    description = client.describe_collection(collection)
    schema = description.get("schema") or description
    return {str(field["name"]) for field in schema.get("fields") or []}


def all_rows(client: MilvusClient, collection: str, primary_key: str) -> list[dict[str, Any]]:
    rows = client.query(collection_name=collection, filter=f"{primary_key} != ''", output_fields=["*"], limit=1000)
    expected = int(client.get_collection_stats(collection).get("row_count", 0))
    if len(rows) != expected:
        raise ValueError(f"{collection}: expected {expected} rows, query returned {len(rows)}; pagination is required")
    return [dict(row) for row in rows]


def report_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("classes") or []
    if not isinstance(entries, list):
        raise ValueError("similarity report must contain a classes list")
    output: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("entry_id"):
            continue
        similar = entry.get("similar") or []
        if not isinstance(similar, list):
            raise ValueError(f"{entry['entry_id']}: similarity row is malformed")
        output[str(entry["entry_id"])] = [dict(item) for item in similar if isinstance(item, dict)]
    return output


def canonical_code(registry: Registry, source_code: object) -> str | None:
    return registry.source_to_canonical.get(str(source_code or ""))


def replace_similar_names(
    row: dict[str, Any],
    registry: Registry,
    candidates_by_entry: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Replace names by report-recorded source codes without changing links/order."""
    output = dict(row)
    entry_id = str(row["entry_id"])
    candidates = candidates_by_entry.get(entry_id)
    if candidates is None:
        raise ValueError(f"{entry_id}: no similarity provenance row")
    current_en = list(row.get("similar_english_classes") or [])
    current_zh = list(row.get("similar_chinese_classes") or [])
    if len(current_en) != len(candidates) or len(current_zh) != len(candidates):
        raise ValueError(f"{entry_id}: stored similar-class lengths differ from provenance")
    expected_en = [str(item.get("english_name") or "") for item in candidates]
    expected_zh = [str(item.get("chinese_name") or "") for item in candidates]
    if current_en != expected_en or current_zh != expected_zh:
        raise ValueError(f"{entry_id}: stored similar classes differ from provenance; refusing to reorder or guess")
    mapped_codes = [canonical_code(registry, item.get("code")) for item in candidates]
    output["similar_english_classes"] = [
        registry.display_name(code, "en") if code else value
        for code, value in zip(mapped_codes, current_en, strict=True)
    ]
    output["similar_chinese_classes"] = [
        registry.display_name(code, "zh") if code else value
        for code, value in zip(mapped_codes, current_zh, strict=True)
    ]
    return output


def rename_class_row(row: dict[str, Any], registry: Registry, candidates_by_entry: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output = replace_similar_names(row, registry, candidates_by_entry)
    code = canonical_code(registry, row.get("code"))
    if code:
        output["english_name"] = registry.display_name(code, "en")
        output["chinese_name"] = registry.display_name(code, "zh")
    return output


def rename_image_row(row: dict[str, Any], registry: Registry) -> dict[str, Any]:
    output = dict(row)
    code = canonical_code(registry, row.get("code"))
    if code:
        output["english_name"] = registry.display_name(code, "en")
        output["chinese_name"] = registry.display_name(code, "zh")
    return output


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {key for key in set(before) | set(after) if before.get(key) != after.get(key)}


def verify_copy(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], registry: Registry, primary_key: str, *, is_class: bool,
) -> dict[str, int]:
    source_by_id = {str(row[primary_key]): row for row in source_rows}
    target_by_id = {str(row[primary_key]): row for row in target_rows}
    if set(source_by_id) != set(target_by_id):
        raise ValueError("source and target primary keys differ")
    mapped = 0
    for key, source in source_by_id.items():
        target = target_by_id[key]
        allowed = NAME_FIELDS if is_class else {"english_name", "chinese_name"}
        if not changed_fields(source, target) <= allowed:
            raise ValueError(f"{key}: migration changed non-name fields {sorted(changed_fields(source, target) - allowed)}")
        code = canonical_code(registry, source.get("code"))
        if code:
            mapped += 1
            if target["english_name"] != registry.display_name(code, "en") or target["chinese_name"] != registry.display_name(code, "zh"):
                raise ValueError(f"{key}: standard display-name conversion failed")
    return {"rows": len(target_rows), "canonical_name_mapped_rows": mapped, "unmapped_rows_retained": len(target_rows) - mapped}


def verify_similar_displays(
    target_rows: list[dict[str, Any]], registry: Registry, candidates_by_entry: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Verify every retained relation slot renders its source-code canonical name."""
    mapped_slots = 0
    unmapped_slots = 0
    display_errors = 0
    for row in target_rows:
        candidates = candidates_by_entry[str(row["entry_id"])]
        english = list(row.get("similar_english_classes") or [])
        chinese = list(row.get("similar_chinese_classes") or [])
        for candidate, value_en, value_zh in zip(candidates, english, chinese, strict=True):
            code = canonical_code(registry, candidate.get("code"))
            if not code:
                unmapped_slots += 1
                continue
            mapped_slots += 1
            if value_en != registry.display_name(code, "en") or value_zh != registry.display_name(code, "zh"):
                display_errors += 1
    if display_errors:
        raise ValueError(f"{display_errors} similar-class slots do not use canonical display names")
    return {
        "mapped_relation_slots": mapped_slots,
        "unmapped_relation_slots_retained": unmapped_slots,
        "canonical_display_errors": display_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--lite-db", type=Path, default=LITE_DB)
    parser.add_argument("--similarity-report", type=Path, default=SIMILARITY_REPORT)
    parser.add_argument("--source-class-collection", default=SOURCE_CLASS_COLLECTION)
    parser.add_argument("--source-image-collection", default=SOURCE_IMAGE_COLLECTION)
    parser.add_argument("--target-class-collection", default=TARGET_CLASS_COLLECTION)
    parser.add_argument("--target-image-collection", default=TARGET_IMAGE_COLLECTION)
    parser.add_argument("--verify-existing", action="store_true", help="Verify existing target collections without writing them.")
    args = parser.parse_args()

    registry = load_registry(args.dataset_root / "taxonomy/canonical_label_registry.jsonl", args.dataset_root / "taxonomy/approval.json")
    client = MilvusClient(uri=str(args.lite_db))
    target_exists = client.has_collection(args.target_class_collection) or client.has_collection(args.target_image_collection)
    if target_exists and not (client.has_collection(args.target_class_collection) and client.has_collection(args.target_image_collection)):
        raise ValueError("only one target collection exists; refusing partial migration state")
    if target_exists and not args.verify_existing:
        raise ValueError("target open_agri_v3 collection already exists; migration is intentionally non-overwriting")
    source_classes = all_rows(client, args.source_class_collection, "entry_id")
    source_images = all_rows(client, args.source_image_collection, "image_id")
    class_fields = collection_fields(client, args.source_class_collection)
    image_fields = collection_fields(client, args.source_image_collection)
    if "text_vector" not in class_fields or "image_vector" not in image_fields:
        raise ValueError("source collections lack the expected SigLIP vector fields")
    dim = len(source_classes[0]["text_vector"])
    if dim != len(source_images[0]["image_vector"]):
        raise ValueError("source text and image vector dimensions differ")
    candidates = report_candidates(args.similarity_report)
    if not target_exists:
        target_classes = [rename_class_row(row, registry, candidates) for row in source_classes]
        target_images = [rename_image_row(row, registry) for row in source_images]
        create_class_collection(client, args.target_class_collection, dim, drop_existing=False)
        create_image_collection(client, args.target_image_collection, dim, drop_existing=False)
        client.insert(collection_name=args.target_class_collection, data=target_classes)
        client.insert(collection_name=args.target_image_collection, data=target_images)
        client.flush(args.target_class_collection)
        client.flush(args.target_image_collection)
    target_classes = all_rows(client, args.target_class_collection, "entry_id")
    target_images = all_rows(client, args.target_image_collection, "image_id")
    verification = {
        "class_collection": verify_copy(source_classes, target_classes, registry, "entry_id", is_class=True),
        "image_collection": verify_copy(source_images, target_images, registry, "image_id", is_class=False),
        "similar_class_displays": verify_similar_displays(target_classes, registry, candidates),
    }
    result = {
        "schema_version": "agrinet.open-agri-v3.milvus-name-migration/v1",
        "taxonomy_version": "open_agri_v3",
        "registry_sha256": registry.digest,
        "database": str(args.lite_db.resolve()),
        "source": {"class_collection": args.source_class_collection, "image_collection": args.source_image_collection},
        "target": {"class_collection": args.target_class_collection, "image_collection": args.target_image_collection},
        "preserved_fields": ["entry IDs", "source codes", "descriptions", "aliases", "reference images", "text/image/sparse vectors", "similarity order"],
        "changed_fields": sorted(NAME_FIELDS),
        "verification": verification,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
