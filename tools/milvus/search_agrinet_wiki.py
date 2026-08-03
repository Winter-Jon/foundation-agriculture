#!/usr/bin/env python3
"""Search AgriNet wiki Milvus collection with SigLIP2 query embeddings."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoModel, AutoProcessor

from ingest_agrinet_wiki import derived_collection_names, encode_sparse_text, model_device, normalize


DEFAULT_OUTPUT_FIELDS = [
    "entry_id",
    "code",
    "english_name",
    "chinese_name",
    "source_dataset",
    "local_reference_images",
    "alias_en",
    "alias_cn",
]

IMAGE_OUTPUT_FIELDS = [
    "image_id",
    "entry_id",
    "code",
    "english_name",
    "chinese_name",
    "source_dataset",
    "local_reference_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", default=[], help="Text query to encode with the SigLIP text encoder. Repeat for multi-text queries.")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Image query to encode with the SigLIP image encoder. Repeat for multi-image queries.")
    parser.add_argument("--collection", default="agrinet_wiki_siglip2")
    parser.add_argument("--class-collection", default="", help="Override class collection name. Default: <collection>_classes.")
    parser.add_argument("--image-collection", default="", help="Override image collection name. Default: <collection>_images.")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--mode", choices=("standalone", "lite"), default="lite")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--lite-db", default="outputs/milvus/agrinet_wiki_lite.db")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--filter", default="", help="Optional Milvus scalar filter expression.")
    parser.add_argument("--ranker", choices=("weighted", "rrf"), default="weighted")
    parser.add_argument("--text-weight", type=float, default=0.35)
    parser.add_argument("--image-weight", type=float, default=0.45)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--separate", action="store_true", help="Return separate per-vector searches instead of hybrid search.")
    parser.add_argument("--output-json", action="store_true", help="Emit machine-readable JSON instead of text lines.")
    args = parser.parse_args()
    if not args.text and not args.image:
        parser.error("Provide --text, --image, or both.")
    for image in args.image:
        if not image.exists():
            parser.error(f"Image does not exist: {image}")
    return args


def connect(args: argparse.Namespace) -> MilvusClient:
    if args.mode == "lite":
        return MilvusClient(uri=args.lite_db)
    return MilvusClient(uri=args.milvus_uri)


def resolve_collection_names(client: MilvusClient, base_collection: str, class_collection: str = "", image_collection: str = "") -> tuple[str, str]:
    classes, images = derived_collection_names(base_collection, class_collection, image_collection)
    if client.has_collection(classes) and client.has_collection(images):
        return classes, images
    if client.has_collection(base_collection):
        return base_collection, base_collection
    raise RuntimeError(f"Collections do not exist: expected {classes}/{images} or legacy {base_collection}")


def load_siglip(model_name: str, requested_device: str) -> tuple[AutoProcessor, AutoModel, torch.device]:
    device = model_device(requested_device)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def encode_text_query(processor: AutoProcessor, model: AutoModel, device: torch.device, text: str) -> list[float]:
    return encode_text_queries(processor, model, device, [text])


def encode_text_queries(processor: AutoProcessor, model: AutoModel, device: torch.device, texts: list[str]) -> list[float]:
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise ValueError("At least one non-empty text query is required")
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.get_text_features(**{key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}})
        features = getattr(outputs, "pooler_output", outputs)
        if features is None:
            raise RuntimeError("Could not obtain text embedding from SigLIP model")
    vectors = normalize(features)
    return normalize(vectors.mean(dim=0, keepdim=True))[0].cpu().tolist()


def encode_pil_image_query(processor: AutoProcessor, model: AutoModel, device: torch.device, image: Image.Image) -> list[float]:
    return encode_pil_image_queries(processor, model, device, [image])


def encode_pil_image_queries(processor: AutoProcessor, model: AutoModel, device: torch.device, images: list[Image.Image]) -> list[float]:
    if not images:
        raise ValueError("At least one image query is required")
    images = [image.convert("RGB") for image in images]
    inputs = processor(text=[""] * len(images), images=images, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        features = getattr(outputs, "image_embeds", None)
        if features is None:
            raise RuntimeError("Could not obtain image embedding from SigLIP model")
    vectors = normalize(features)
    return normalize(vectors.mean(dim=0, keepdim=True))[0].cpu().tolist()


def encode_image_query(processor: AutoProcessor, model: AutoModel, device: torch.device, image_path: Path) -> list[float]:
    return encode_image_queries(processor, model, device, [image_path])


def encode_image_queries(processor: AutoProcessor, model: AutoModel, device: torch.device, image_paths: list[Path]) -> list[float]:
    images = [Image.open(image_path) for image_path in image_paths]
    return encode_pil_image_queries(processor, model, device, images)


def output_fields_for(field: str) -> list[str]:
    return IMAGE_OUTPUT_FIELDS if field == "image_vector" else DEFAULT_OUTPUT_FIELDS


def run_search(client: MilvusClient, collection: str, field: str, vector: list[float] | dict[int, float], top_k: int, expr: str) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "data": [vector],
        "anns_field": field,
        "limit": top_k,
        "output_fields": output_fields_for(field),
    }
    if expr:
        kwargs["filter"] = expr
    hits = client.search(**kwargs)
    return hits[0] if hits else []


def lexical_name_score(query: str, entity: dict[str, Any]) -> float:
    fields = [
        entity.get("english_name") or "",
        entity.get("chinese_name") or "",
        " ".join(entity.get("alias_en") or []),
        " ".join(entity.get("alias_cn") or []),
    ]
    text = " ".join(fields).lower().replace("_", " ").replace("-", " ")
    query_text = query.lower().replace("_", " ").replace("-", " ")
    query_tokens = re.findall(r"[a-z0-9]+", query_text)
    name_tokens = set(re.findall(r"[a-z0-9]+", text))
    if not query_tokens:
        return 0.0
    score = sum(1.0 for token in query_tokens if token in name_tokens)
    for token in query_tokens:
        if token == "healthy" and "normal" in name_tokens:
            score += 0.95
        if token == "normal" and "healthy" in name_tokens:
            score += 0.95
    if "healthy" in query_tokens or "normal" in query_tokens:
        if {"healthy", "normal"} & name_tokens:
            score += 3.0
        disease_terms = {"scorch", "spot", "mildew", "disease", "rot", "blight", "rust", "virus", "wilt", "scab", "powdery", "hole"}
        if disease_terms & name_tokens:
            score -= 2.0
    ordered = " ".join(query_tokens)
    if ordered and ordered in text:
        score += 4.0
    for ngram_size in (2, 3):
        for index in range(0, max(0, len(query_tokens) - ngram_size + 1)):
            if " ".join(query_tokens[index : index + ngram_size]) in text:
                score += float(ngram_size)
    return score / math.sqrt(max(1, len(name_tokens)))


def name_query_score(query: str, entity: dict[str, Any], distance: float) -> float:
    return distance + lexical_name_score(query, entity)


def collect_class_entities(client: MilvusClient, collection: str, entry_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not entry_ids:
        return {}
    quoted = ", ".join(json.dumps(entry_id) for entry_id in entry_ids)
    rows = client.query(collection_name=collection, filter=f"entry_id in [{quoted}]", output_fields=DEFAULT_OUTPUT_FIELDS)
    return {str(row.get("entry_id")): row for row in rows}


def fuse_hits(
    client: MilvusClient,
    class_collection: str,
    raw_hits_by_field: dict[str, list[dict[str, Any]]],
    query_text: str,
    top_k: int,
    ranker_name: str,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    image_entry_ids = []
    for hit in raw_hits_by_field.get("image_vector", []):
        entity = hit.get("entity", {})
        entry_id = str(entity.get("entry_id") or "")
        if entry_id:
            image_entry_ids.append(entry_id)
    class_entities = collect_class_entities(client, class_collection, image_entry_ids)

    fused: dict[str, dict[str, Any]] = {}
    for field, hits in raw_hits_by_field.items():
        field_weight = weights.get(field, 0.0)
        for rank_index, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            entry_id = str(entity.get("entry_id") or hit.get("id") or "")
            if not entry_id:
                continue
            class_entity = class_entities.get(entry_id, entity) if field == "image_vector" else entity
            item = fused.setdefault(
                entry_id,
                {
                    "id": entry_id,
                    "entity": class_entity,
                    "distance": 0.0,
                    "component_scores": {},
                    "image_hits": [],
                },
            )
            distance = float(hit.get("distance") or 0.0)
            if field == "sparse_vector" and query_text:
                distance = name_query_score(query_text, class_entity, distance)
            weighted = (1.0 / (60.0 + rank_index)) if ranker_name == "rrf" else field_weight * distance
            item["distance"] += weighted
            item["component_scores"][field] = max(float(item["component_scores"].get(field, float("-inf"))), weighted)
            if field == "image_vector":
                image_path = entity.get("local_reference_image")
                if image_path:
                    item["image_hits"].append({"path": image_path, "score": distance, "image_id": entity.get("image_id")})

    ranked = sorted(fused.values(), key=lambda value: (float(value["distance"]), str(value["id"])), reverse=True)
    return ranked[:top_k]


def run_hybrid_search(
    client: MilvusClient,
    class_collection: str,
    image_collection: str,
    vectors: dict[str, list[float]],
    query_text: str,
    top_k: int,
    expr: str,
    ranker_name: str,
    text_weight: float,
    image_weight: float,
    sparse_weight: float,
) -> list[dict[str, Any]]:
    if not vectors:
        return []
    if ranker_name == "rrf":
        sparse_weight = 0.0
    overfetch = max(top_k * 8, 32)
    raw_hits_by_field: dict[str, list[dict[str, Any]]] = {}
    if "text_vector" in vectors and text_weight > 0:
        raw_hits_by_field["text_vector"] = run_search(client, class_collection, "text_vector", vectors["text_vector"], overfetch, expr)
    if "sparse_vector" in vectors and sparse_weight > 0:
        raw_hits_by_field["sparse_vector"] = run_search(client, class_collection, "sparse_vector", vectors["sparse_vector"], overfetch, expr)
    if "image_vector" in vectors and image_weight > 0:
        image_expr = expr if image_collection == class_collection else ""
        raw_hits_by_field["image_vector"] = run_search(client, image_collection, "image_vector", vectors["image_vector"], overfetch, image_expr)
    active_count = sum(1 for hits in raw_hits_by_field.values() if hits)
    if active_count == 1 and ranker_name == "weighted" and "image_vector" not in raw_hits_by_field:
        field, hits = next(iter(raw_hits_by_field.items()))
        if field == "sparse_vector" and query_text:
            for hit in hits:
                hit["distance"] = name_query_score(query_text, hit.get("entity", {}), float(hit.get("distance") or 0.0))
            hits = sorted(hits, key=lambda hit: (float(hit.get("distance") or 0.0), str(hit.get("id") or "")), reverse=True)
        return hits[:top_k]
    weights = {"text_vector": text_weight, "image_vector": image_weight, "sparse_vector": sparse_weight}
    return fuse_hits(client, class_collection, raw_hits_by_field, query_text, top_k, ranker_name, weights)


def simplify_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        try:
            return [json_safe(item) for item in value]
        except TypeError:
            return str(value)

    simplified = []
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        local_reference_images = entity.get("local_reference_images")
        if local_reference_images is not None and not isinstance(local_reference_images, list):
            local_reference_images = list(local_reference_images)
        image_hits = hit.get("image_hits") or []
        if image_hits:
            ranked_paths = [item["path"] for item in sorted(image_hits, key=lambda value: float(value.get("score") or 0.0), reverse=True) if item.get("path")]
            existing = local_reference_images or []
            local_reference_images = list(dict.fromkeys([*ranked_paths, *existing]))
        distance = float(hit.get("distance"))
        simplified.append(
            {
                "rank": rank,
                "id": json_safe(hit.get("id")),
                "distance": distance,
                "english_name": json_safe(entity.get("english_name")),
                "chinese_name": json_safe(entity.get("chinese_name")),
                "alias_en": json_safe(entity.get("alias_en") or []),
                "alias_cn": json_safe(entity.get("alias_cn") or []),
                "source_dataset": json_safe(entity.get("source_dataset")),
                "local_reference_images": json_safe(local_reference_images),
            }
        )
    return simplified


def print_hits(label: str, hits: list[dict[str, Any]], *, simplified: bool = False) -> None:
    print(label, flush=True)
    rows = hits if simplified else simplify_hits(hits)
    for hit in rows:
        print(
            f"  rank={hit['rank']} id={hit['id']} distance={hit['distance']} "
            f"english_name={hit['english_name']} chinese_name={hit['chinese_name']}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    client = connect(args)
    class_collection, image_collection = resolve_collection_names(client, args.collection, args.class_collection, args.image_collection)
    client.load_collection(class_collection)
    if image_collection != class_collection:
        client.load_collection(image_collection)

    processor, model, device = load_siglip(args.model_name, args.device)
    vectors: dict[str, list[float]] = {}
    results: dict[str, list[dict[str, Any]]] = {}

    if args.text:
        vectors["text_vector"] = encode_text_queries(processor, model, device, args.text)
        vectors["sparse_vector"] = encode_sparse_text("\n".join(args.text))

    if args.image:
        vectors["image_vector"] = encode_image_queries(processor, model, device, args.image)

    if args.separate:
        for field, vector in vectors.items():
            collection = image_collection if field == "image_vector" else class_collection
            hits = run_search(client, collection, field, vector, args.top_k, args.filter if collection == class_collection else "")
            results[field] = simplify_hits(hits)
    else:
        hits = run_hybrid_search(
            client,
            class_collection,
            image_collection,
            vectors,
            "\n".join(args.text),
            args.top_k,
            args.filter,
            args.ranker,
            args.text_weight,
            args.image_weight,
            args.sparse_weight,
        )
        results["hybrid"] = simplify_hits(hits)

    if args.output_json:
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    else:
        for field, hits in results.items():
            print_hits(field, hits, simplified=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
