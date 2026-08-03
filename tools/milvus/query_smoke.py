#!/usr/bin/env python3
"""Run smoke queries against the AgriNet wiki Milvus collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoModel, AutoProcessor

from ingest_agrinet_wiki import encode_sparse_text, images_from_entries, load_entries, model_device, normalize
from search_agrinet_wiki import resolve_collection_names, run_hybrid_search, simplify_hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", default="datasets/AgriNet-1K/wiki/base.json")
    parser.add_argument("--image-dir", default="datasets/AgriNet-1K/wiki/images")
    parser.add_argument("--collection", default="agrinet_wiki_siglip2")
    parser.add_argument("--class-collection", default="", help="Override class collection name. Default: <collection>_classes.")
    parser.add_argument("--image-collection", default="", help="Override image collection name. Default: <collection>_images.")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--mode", choices=("standalone", "lite"), default="standalone")
    parser.add_argument("--lite-db", default="outputs/milvus/agrinet_wiki_lite.db")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def connect(args: argparse.Namespace) -> MilvusClient:
    if args.mode == "lite":
        return MilvusClient(uri=args.lite_db)
    return MilvusClient(uri=args.milvus_uri)


def encode_text(model: AutoModel, processor: AutoProcessor, device, text: str) -> list[float]:
    inputs = processor(text=[text], padding="max_length", return_tensors="pt").to(device)
    with __import__("torch").inference_mode():
        text_features = model.get_text_features(**{k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}})
        text_features = getattr(text_features, "pooler_output", text_features)
        if text_features is None:
            raise RuntimeError("Could not obtain text embeddings")
    return normalize(text_features)[0].cpu().tolist()


def encode_image(model: AutoModel, processor: AutoProcessor, device, image_path: Path) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[""], images=[image], padding="max_length", return_tensors="pt").to(device)
    with __import__("torch").inference_mode():
        outputs = model(**inputs)
        image_features = getattr(outputs, "image_embeds", None)
        if image_features is None and hasattr(model, "get_image_features"):
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = getattr(image_features, "pooler_output", image_features)
        if image_features is None:
            raise RuntimeError("Could not obtain image embeddings")
    return normalize(image_features)[0].cpu().tolist()


def print_hits(label: str, hits: list[list[dict]]) -> None:
    print(label, flush=True)
    for hit in hits[0]:
        entity = hit.get("entity", {})
        print(
            f"  rank_hit id={hit.get('id')} distance={hit.get('distance')} "
            f"code={entity.get('code')} english_name={entity.get('english_name')} "
            f"refs={len(entity.get('local_reference_images') or [])}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    entries, stats = load_entries(Path(args.knowledge_base), Path(args.image_dir))
    images = images_from_entries(entries)
    if not entries:
        raise RuntimeError("No local entries for query smoke")

    client = connect(args)
    class_collection, image_collection = resolve_collection_names(client, args.collection, args.class_collection, args.image_collection)
    client.load_collection(class_collection)
    if image_collection != class_collection:
        client.load_collection(image_collection)
    class_stats = client.get_collection_stats(class_collection)
    image_stats = client.get_collection_stats(image_collection)
    class_row_count = int(class_stats.get("row_count", 0))
    image_row_count = int(image_stats.get("row_count", 0))
    print(f"data_stats={stats}", flush=True)
    print(f"class_collection={class_collection} class_stats={class_stats}", flush=True)
    print(f"image_collection={image_collection} image_stats={image_stats}", flush=True)
    if class_row_count != stats["total"]:
        raise RuntimeError(f"Expected class row_count={stats['total']}, got {class_row_count}")
    if image_collection != class_collection and image_row_count != len(images):
        raise RuntimeError(f"Expected image row_count={len(images)}, got {image_row_count}")

    sample = entries[0]
    sample_image = images[0]
    print(f"sample_entry={sample.entry_id} image={sample_image.image_path}", flush=True)
    device = model_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    text_vector = encode_text(model, processor, device, sample.embedding_text)
    image_vector = encode_image(model, processor, device, sample_image.image_path)

    output_fields = [
        "entry_id",
        "code",
        "english_name",
        "chinese_name",
        "content_1",
        "similar_english_classes",
        "reference_images",
        "local_reference_images",
        "alias_en",
        "alias_cn",
        "payload",
    ]
    text_hits = client.search(
        collection_name=class_collection,
        data=[text_vector],
        anns_field="text_vector",
        limit=args.top_k,
        output_fields=output_fields,
    )
    image_hits = client.search(
        collection_name=image_collection,
        data=[image_vector],
        anns_field="image_vector",
        limit=args.top_k,
        output_fields=["image_id", "entry_id", "code", "english_name", "local_reference_image"],
    )
    print_hits("text_query_hits", text_hits)
    print_hits("image_query_hits", image_hits)

    if not image_hits or not image_hits[0] or image_hits[0][0].get("entity", {}).get("entry_id") != sample.entry_id:
        raise RuntimeError("Image self-query did not return the sample entry as top-1")
    if not text_hits or not text_hits[0]:
        raise RuntimeError("Text query returned no hits")

    entity = text_hits[0][0].get("entity", {})
    print(
        "typed_field_sample="
        f"content_1_len={len(entity.get('content_1') or '')} "
        f"similar_en={entity.get('similar_english_classes')} "
        f"reference_images={entity.get('reference_images')} "
        f"payload_keys={sorted((entity.get('payload') or {}).keys())}",
        flush=True,
    )

    name_query = "healthy cherry leaf"
    name_hits = run_hybrid_search(
        client,
        class_collection,
        image_collection,
        {"sparse_vector": encode_sparse_text(name_query)},
        name_query,
        3,
        "",
        "weighted",
        0.0,
        0.0,
        1.0,
    )
    simplified_name_hits = simplify_hits(name_hits)
    print(f"name_regression_query={name_query}", flush=True)
    for hit in simplified_name_hits:
        print(f"  rank={hit['rank']} score={hit['distance']} english_name={hit['english_name']} chinese_name={hit['chinese_name']}", flush=True)
    top_name = (simplified_name_hits[0].get("english_name") or "").lower() if simplified_name_hits else ""
    if "cherry" not in top_name or not ({"healthy", "normal"} & set(top_name.split())):
        raise RuntimeError(f"Name regression failed: expected healthy/normal cherry top-1, got {top_name!r}")
    print("query_smoke_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
