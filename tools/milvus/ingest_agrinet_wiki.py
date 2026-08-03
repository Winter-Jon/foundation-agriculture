#!/usr/bin/env python3
"""Embed AgriNet-1K wiki classes with SigLIP2 and ingest them into Milvus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from pymilvus import DataType, MilvusClient
from transformers import AutoModel, AutoProcessor


CONTENT_FIELDS = tuple(f"content_{index}" for index in range(1, 9))
ARRAY_CAPACITY = 16
ARRAY_ITEM_LENGTH = 1024
TEXT_MAX_LENGTH = 65535
SPARSE_DIM = 1_048_576


@dataclass(frozen=True)
class WikiEntry:
    entry_id: str
    code: str
    english_name: str
    chinese_name: str
    source_dataset: str
    contents: dict[str, str]
    similar_chinese_classes: list[str]
    similar_english_classes: list[str]
    reference_images: list[str]
    local_reference_images: list[str]
    alias_en: list[str]
    alias_cn: list[str]
    payload: dict[str, Any]
    embedding_text: str
    sparse_text: str
    sparse_vector: dict[int, float]


@dataclass(frozen=True)
class WikiImage:
    image_id: str
    entry_id: str
    code: str
    english_name: str
    chinese_name: str
    source_dataset: str
    reference_image: str
    local_reference_image: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", default="datasets/AgriNet-1K/wiki/base.json")
    parser.add_argument("--image-dir", default="datasets/AgriNet-1K/wiki/images")
    parser.add_argument("--collection", default="agrinet_wiki_siglip2", help="Collection base name. Two collections are created: <base>_classes and <base>_images.")
    parser.add_argument("--class-collection", default="", help="Override class collection name. Default: <collection>_classes.")
    parser.add_argument("--image-collection", default="", help="Override image collection name. Default: <collection>_images.")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--mode", choices=("standalone", "lite"), default="standalone")
    parser.add_argument("--lite-db", default="outputs/milvus/agrinet_wiki_lite.db")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke limit. 0 means all classes.")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        return [as_str(value)]
    return [as_str(item) for item in value if as_str(item)]


def clamp_text(value: str, field_name: str, entry_id: str) -> str:
    if len(value) <= TEXT_MAX_LENGTH:
        return value
    print(f"warning=truncated_text entry_id={entry_id} field={field_name} original_len={len(value)}", flush=True)
    return value[:TEXT_MAX_LENGTH]


def resolve_reference_images(reference_images: list[str], image_dir: Path) -> list[str]:
    local_paths: list[str] = []
    for image in reference_images:
        image_path = image_dir / Path(image).name
        if image_path.exists():
            local_paths.append(str(image_path))
    return local_paths


def build_embedding_text(entry_id: str, item: dict[str, Any], contents: dict[str, str]) -> str:
    parts = [
        as_str(item.get("english_name")),
        as_str(item.get("chinese_name")),
        " ".join(as_str_list(item.get("alias_en"))),
        " ".join(as_str_list(item.get("alias_cn"))),
        " ".join(as_str_list(item.get("similar_english_classes"))),
        " ".join(as_str_list(item.get("similar_chinese_classes"))),
    ]
    parts.extend(contents[field] for field in CONTENT_FIELDS if contents[field])
    text = "\n".join(part for part in parts if part.strip())
    return clamp_text(text, "embedding_text", entry_id)


def build_sparse_text(item: dict[str, Any]) -> str:
    names = [
        as_str(item.get("english_name")),
        as_str(item.get("chinese_name")),
        " ".join(as_str_list(item.get("alias_en"))),
        " ".join(as_str_list(item.get("alias_cn"))),
    ]
    expanded = []
    for name in names:
        if not name.strip():
            continue
        expanded.append(name)
        normalized = re.sub(r"[_\-]+", " ", name.lower())
        if "healthy" in normalized:
            expanded.append(re.sub(r"\bhealthy\b", "normal", normalized))
        if "normal" in normalized:
            expanded.append(re.sub(r"\bnormal\b", "healthy", normalized))
            if "leaf" in normalized:
                expanded.append(re.sub(r"\bnormal\s+leaf\b", "healthy leaf", normalized))
        if "healthy" in normalized and "leaf" not in normalized:
            expanded.append(f"{normalized} leaf")
    return "\n".join(part for part in expanded if part.strip())


def sparse_tokens(text: str) -> list[str]:
    normalized = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", normalized)
    phrase_tokens: list[str] = []
    for ngram_size in (2, 3):
        phrase_tokens.extend(" ".join(latin_tokens[index : index + ngram_size]) for index in range(0, max(0, len(latin_tokens) - ngram_size + 1)))
    for index, token_a in enumerate(latin_tokens):
        for token_b in latin_tokens[index + 1 :]:
            phrase_tokens.append(" ".join(sorted((token_a, token_b))))
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", normalized)
    return latin_tokens + phrase_tokens + cjk_tokens


def token_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % SPARSE_DIM


def encode_sparse_text(text: str) -> dict[int, float]:
    counts: dict[int, int] = {}
    for token in sparse_tokens(text):
        index = token_index(token)
        counts[index] = counts.get(index, 0) + 1
    if not counts:
        return {0: 0.0}
    vector = {index: 1.0 + math.log(count) for index, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {index: value / norm for index, value in vector.items()}


def load_entries(kb_path: Path, image_dir: Path) -> tuple[list[WikiEntry], dict[str, int]]:
    with kb_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries: list[WikiEntry] = []
    missing_images: list[str] = []
    descriptions = payload.get("description") or {}

    for entry_id, item in descriptions.items():
        desc = item.get("description") or {}
        contents = {field: clamp_text(as_str(desc.get(field)), field, entry_id) for field in CONTENT_FIELDS}
        reference_images = as_str_list(item.get("reference_images"))
        local_reference_images = resolve_reference_images(reference_images, image_dir)
        if not local_reference_images:
            missing_images.append(entry_id)
            continue

        sparse_text = build_sparse_text(item)
        entry = WikiEntry(
            entry_id=entry_id,
            code=as_str(item.get("code")),
            english_name=as_str(item.get("english_name")),
            chinese_name=as_str(item.get("chinese_name")),
            source_dataset=as_str(item.get("source_dataset")),
            contents=contents,
            similar_chinese_classes=as_str_list(item.get("similar_chinese_classes")),
            similar_english_classes=as_str_list(item.get("similar_english_classes")),
            reference_images=reference_images,
            local_reference_images=local_reference_images,
            alias_en=as_str_list(item.get("alias_en")),
            alias_cn=as_str_list(item.get("alias_cn")),
            payload=item,
            embedding_text=build_embedding_text(entry_id, item, contents),
            sparse_text=sparse_text,
            sparse_vector=encode_sparse_text(sparse_text),
        )
        entries.append(entry)

    if missing_images:
        preview = ", ".join(missing_images[:20])
        raise RuntimeError(f"Missing local reference image for {len(missing_images)} classes: {preview}")

    stats = {
        "total": len(descriptions),
        "eligible": len(entries),
        "missing_local_image": len(missing_images),
        "with_content": sum(1 for entry in entries if any(entry.contents.values())),
    }
    return entries, stats


def images_from_entries(entries: list[WikiEntry]) -> list[WikiImage]:
    images: list[WikiImage] = []
    for entry in entries:
        for index, local_path in enumerate(entry.local_reference_images):
            reference_image = entry.reference_images[index] if index < len(entry.reference_images) else local_path
            images.append(
                WikiImage(
                    image_id=f"{entry.entry_id}::image::{index:04d}",
                    entry_id=entry.entry_id,
                    code=entry.code,
                    english_name=entry.english_name,
                    chinese_name=entry.chinese_name,
                    source_dataset=entry.source_dataset,
                    reference_image=reference_image,
                    local_reference_image=local_path,
                    image_path=Path(local_path),
                )
            )
    return images


def batches(items: list[WikiEntry], batch_size: int) -> Iterable[list[WikiEntry]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, p=2, dim=-1)


def model_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(requested)


def add_string_array(schema: Any, field_name: str) -> None:
    schema.add_field(
        field_name,
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=ARRAY_CAPACITY,
        max_length=ARRAY_ITEM_LENGTH,
    )


def derived_collection_names(base_collection: str, class_collection: str = "", image_collection: str = "") -> tuple[str, str]:
    return class_collection or f"{base_collection}_classes", image_collection or f"{base_collection}_images"


def create_class_collection(client: MilvusClient, collection: str, dim: int, drop_existing: bool) -> None:
    if client.has_collection(collection):
        if not drop_existing:
            raise RuntimeError(f"Collection already exists: {collection}; pass --drop-existing to rebuild it")
        client.drop_collection(collection)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("entry_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("code", DataType.VARCHAR, max_length=32)
    schema.add_field("english_name", DataType.VARCHAR, max_length=256)
    schema.add_field("chinese_name", DataType.VARCHAR, max_length=128)
    schema.add_field("source_dataset", DataType.VARCHAR, max_length=128)
    schema.add_field("model_name", DataType.VARCHAR, max_length=256)
    schema.add_field("has_reference_image", DataType.BOOL)
    for field in CONTENT_FIELDS:
        schema.add_field(field, DataType.VARCHAR, max_length=TEXT_MAX_LENGTH)
    schema.add_field("embedding_text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH)
    schema.add_field("sparse_text", DataType.VARCHAR, max_length=4096)
    for field in (
        "similar_chinese_classes",
        "similar_english_classes",
        "reference_images",
        "local_reference_images",
        "alias_en",
        "alias_cn",
    ):
        add_string_array(schema, field)
    schema.add_field("payload", DataType.JSON)
    schema.add_field("text_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index("text_vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)


def create_image_collection(client: MilvusClient, collection: str, dim: int, drop_existing: bool) -> None:
    if client.has_collection(collection):
        if not drop_existing:
            raise RuntimeError(f"Collection already exists: {collection}; pass --drop-existing to rebuild it")
        client.drop_collection(collection)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("image_id", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("entry_id", DataType.VARCHAR, max_length=128)
    schema.add_field("code", DataType.VARCHAR, max_length=32)
    schema.add_field("english_name", DataType.VARCHAR, max_length=256)
    schema.add_field("chinese_name", DataType.VARCHAR, max_length=128)
    schema.add_field("source_dataset", DataType.VARCHAR, max_length=128)
    schema.add_field("model_name", DataType.VARCHAR, max_length=256)
    schema.add_field("reference_image", DataType.VARCHAR, max_length=1024)
    schema.add_field("local_reference_image", DataType.VARCHAR, max_length=1024)
    schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index("image_vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)


def connect(args: argparse.Namespace) -> MilvusClient:
    if args.mode == "lite":
        lite_path = Path(args.lite_db)
        lite_path.parent.mkdir(parents=True, exist_ok=True)
        return MilvusClient(uri=str(lite_path))
    return MilvusClient(uri=args.milvus_uri)


def encode_text_batch(model: AutoModel, processor: AutoProcessor, device: torch.device, batch: list[WikiEntry]) -> list[list[float]]:
    texts = [item.embedding_text for item in batch]
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)

    with torch.inference_mode():
        text_features = model.get_text_features(**{k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}})
        text_features = getattr(text_features, "pooler_output", text_features)
        if text_features is None:
            raise RuntimeError("Could not obtain text embeddings from SigLIP2 model output")

    return normalize(text_features).cpu().tolist()


def encode_image_batch(model: AutoModel, processor: AutoProcessor, device: torch.device, batch: list[WikiImage]) -> list[list[float]]:
    images = [Image.open(item.image_path).convert("RGB") for item in batch]
    inputs = processor(text=[""] * len(images), images=images, padding="max_length", return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        image_features = getattr(outputs, "image_embeds", None)
        if image_features is None and hasattr(model, "get_image_features"):
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = getattr(image_features, "pooler_output", image_features)
        if image_features is None:
            raise RuntimeError("Could not obtain image embeddings from SigLIP2 model output")

    return normalize(image_features).cpu().tolist()


def row_from_entry(entry: WikiEntry, model_name: str, text_vector: list[float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "code": entry.code,
        "english_name": entry.english_name,
        "chinese_name": entry.chinese_name,
        "source_dataset": entry.source_dataset,
        "model_name": model_name,
        "has_reference_image": True,
        "embedding_text": entry.embedding_text,
        "sparse_text": entry.sparse_text,
        "similar_chinese_classes": entry.similar_chinese_classes,
        "similar_english_classes": entry.similar_english_classes,
        "reference_images": entry.reference_images,
        "local_reference_images": entry.local_reference_images,
        "alias_en": entry.alias_en,
        "alias_cn": entry.alias_cn,
        "payload": entry.payload,
        "text_vector": text_vector,
        "sparse_vector": entry.sparse_vector,
    }
    row.update(entry.contents)
    return row


def row_from_image(image: WikiImage, model_name: str, image_vector: list[float]) -> dict[str, Any]:
    return {
        "image_id": image.image_id,
        "entry_id": image.entry_id,
        "code": image.code,
        "english_name": image.english_name,
        "chinese_name": image.chinese_name,
        "source_dataset": image.source_dataset,
        "model_name": model_name,
        "reference_image": image.reference_image,
        "local_reference_image": image.local_reference_image,
        "image_vector": image_vector,
    }


def main() -> int:
    args = parse_args()
    entries, stats = load_entries(Path(args.knowledge_base), Path(args.image_dir))
    if args.limit > 0:
        entries = entries[: args.limit]
    images = images_from_entries(entries)

    print(f"data_stats={stats}", flush=True)
    print(f"selected_entries={len(entries)}", flush=True)
    print(f"selected_images={len(images)}", flush=True)
    if not entries:
        raise RuntimeError("No classes to ingest")

    device = model_device(args.device)
    print(f"loading_model={args.model_name} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    client = connect(args)
    first_text_vectors = encode_text_batch(model, processor, device, entries[:1])
    first_image_vectors = encode_image_batch(model, processor, device, images[:1])
    dim = len(first_text_vectors[0])
    if dim != len(first_image_vectors[0]):
        raise RuntimeError(f"Text/image dimensions differ: {dim} vs {len(first_image_vectors[0])}")
    print(f"embedding_dim={dim}", flush=True)
    class_collection, image_collection = derived_collection_names(args.collection, args.class_collection, args.image_collection)
    if args.drop_existing and args.collection not in {class_collection, image_collection} and client.has_collection(args.collection):
        client.drop_collection(args.collection)
        print(f"dropped_legacy_collection={args.collection}", flush=True)
    create_class_collection(client, class_collection, dim, args.drop_existing)
    create_image_collection(client, image_collection, dim, args.drop_existing)

    inserted = 0
    for batch in batches(entries, args.batch_size):
        text_vectors = encode_text_batch(model, processor, device, batch)
        rows = [row_from_entry(item, args.model_name, text_vector) for item, text_vector in zip(batch, text_vectors, strict=True)]
        client.insert(collection_name=class_collection, data=rows)
        inserted += len(rows)
        print(f"inserted={inserted}/{len(entries)}", flush=True)

    inserted_images = 0
    for batch in batches(images, args.batch_size):
        image_vectors = encode_image_batch(model, processor, device, batch)
        rows = [row_from_image(item, args.model_name, image_vector) for item, image_vector in zip(batch, image_vectors, strict=True)]
        client.insert(collection_name=image_collection, data=rows)
        inserted_images += len(rows)
        print(f"inserted_images={inserted_images}/{len(images)}", flush=True)

    client.flush(class_collection)
    client.flush(image_collection)
    print(f"ingest_complete class_collection={class_collection} inserted_classes={inserted} image_collection={image_collection} inserted_images={inserted_images}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
