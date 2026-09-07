#!/usr/bin/env python3
"""Build a parallel, approval-gated canonical-v1 Milvus collection.

This wrapper intentionally emits an input JSONL for the existing embedder only
after validating canonical-v1 approval. It never names or mutates the v2
collections. Run the printed command explicitly for the expensive embedding
operation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
sys.path.insert(0, str(REPO_ROOT / "src"))
from agrinet.research.open_agri_v2_canonical.registry import CANONICAL_VERSION, load_registry

CLASS_COLLECTION = "open_agri_v2_canonical_v1_classes"
IMAGE_COLLECTION = "open_agri_v2_canonical_v1_images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/artifacts/open-agri-v2-canonical-v1-milvus")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--execute", action="store_true", help="Reserved: require a dedicated embedding run; this wrapper only prepares the release spec.")
    parser.add_argument("--review-preview", action="store_true", help="Render a non-executable draft spec while manual taxonomy review is pending.")
    args = parser.parse_args()
    registry = load_registry(ROOT / "taxonomy/canonical_label_registry.jsonl", ROOT / "taxonomy/approval.json", require_approval=not args.review_preview)
    if args.execute:
        raise ValueError("Use the reviewed canonical collection spec with the dedicated embedding runner; this gate does not silently launch a long re-embedding job.")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = ROOT / "catalog/canonical_catalog.jsonl"
    rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 211:
        raise ValueError("canonical catalog must contain 211 approved rows")
    spec = {
        "schema_version": "agrinet.open-agri-v2-canonical-v1-milvus-spec/v1",
        "taxonomy_version": CANONICAL_VERSION, "registry_sha256": registry.digest,
        "class_collection": CLASS_COLLECTION, "image_collection": IMAGE_COLLECTION,
        "collection_policy": "parallel_only; existing Milvus collections are immutable",
        "release_status": "review_preview_only" if args.review_preview else "approved",
        "catalog": str(catalog), "images": str(ROOT / "manifests/images.jsonl"),
        "embedding_inputs": ["canonical English/Chinese names", "approved answer aliases", "public knowledge", "life stage", "source-code lineage"],
        "image_metadata": ["source_class_code", "canonical_class_code", "image_sha256", "image_split"],
        "retrieval_output_schema": "agrinet.open-agri-v2-canonical-v1.candidate-card/v1",
        "command_template": None if args.review_preview else f".venv/bin/python scripts/milvus/ingest_agrinet_wiki.py --collection {CLASS_COLLECTION.rsplit('_classes', 1)[0]} --class-collection {CLASS_COLLECTION} --image-collection {IMAGE_COLLECTION} --milvus-uri {args.milvus_uri} --model-name {args.model_name}",
    }
    (args.output / "collection_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(spec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
