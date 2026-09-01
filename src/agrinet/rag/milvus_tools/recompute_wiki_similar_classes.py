#!/usr/bin/env python3
"""Recompute AgriNet wiki similar-class fields with SigLIP2 embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .ingest_agrinet_wiki import CONTENT_FIELDS, as_str, as_str_list, batches, model_device, normalize, resolve_reference_images


GROUP_PREFIXES = ("N04", "N05")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", default="datasets/AgriNet-1K/wiki/base.json")
    parser.add_argument("--image-dir", default="datasets/AgriNet-1K/wiki/images")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--report", default="outputs/milvus/wiki_similar_classes_siglip2_report.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke limit. 0 means all classes.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--write", action="store_true", help="Update the wiki JSON in place after writing the report.")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    descriptions = payload.get("description")
    if not isinstance(descriptions, dict):
        raise RuntimeError(f"Expected {path} to contain a description object")
    return payload


def class_group(code: str) -> str:
    for prefix in GROUP_PREFIXES:
        if code.startswith(prefix):
            return prefix
    raise RuntimeError(f"Unsupported class code prefix: {code!r}")


def build_similarity_text(item: dict[str, Any]) -> str:
    """Build SigLIP text without class codes or existing similar-class fields."""
    desc = item.get("description") or {}
    parts = [
        as_str(item.get("english_name")),
        as_str(item.get("chinese_name")),
        " ".join(as_str_list(item.get("alias_en"))),
        " ".join(as_str_list(item.get("alias_cn"))),
    ]
    parts.extend(as_str(desc.get(field)) for field in CONTENT_FIELDS)
    return "\n".join(part for part in parts if part.strip())


def selected_items(payload: dict[str, Any], image_dir: Path, limit: int) -> list[tuple[str, dict[str, Any], Path]]:
    items: list[tuple[str, dict[str, Any], Path]] = []
    descriptions: dict[str, dict[str, Any]] = payload["description"]
    for entry_id, item in descriptions.items():
        code = as_str(item.get("code"))
        class_group(code)
        local_images = resolve_reference_images(as_str_list(item.get("reference_images")), image_dir)
        if not local_images:
            raise RuntimeError(f"Missing local reference image for {entry_id}")
        items.append((entry_id, item, Path(local_images[0])))
        if limit > 0 and len(items) >= limit:
            break
    return items


def encode_text_batch(model: AutoModel, processor: AutoProcessor, device: torch.device, texts: list[str]) -> torch.Tensor:
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        features = model.get_text_features(**{key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}})
        features = getattr(features, "pooler_output", features)
        if features is None:
            raise RuntimeError("Could not obtain text embeddings from SigLIP2 model output")
    return normalize(features).cpu()


def encode_image_batch(model: AutoModel, processor: AutoProcessor, device: torch.device, paths: list[Path]) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    inputs = processor(text=[""] * len(images), images=images, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        features = getattr(outputs, "image_embeds", None)
        if features is None and hasattr(model, "get_image_features"):
            features = model.get_image_features(pixel_values=inputs["pixel_values"])
        features = getattr(features, "pooler_output", features)
        if features is None:
            raise RuntimeError("Could not obtain image embeddings from SigLIP2 model output")
    return normalize(features).cpu()


def encode_all(
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
    texts: list[str],
    image_paths: list[Path],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_vectors = [encode_text_batch(model, processor, device, batch) for batch in batches(texts, batch_size)]
    image_vectors = [encode_image_batch(model, processor, device, batch) for batch in batches(image_paths, batch_size)]
    return torch.cat(text_vectors, dim=0), torch.cat(image_vectors, dim=0)


def rank_similar(
    items: list[tuple[str, dict[str, Any], Path]],
    text_vectors: torch.Tensor,
    image_vectors: torch.Tensor,
    top_k: int,
    text_weight: float,
    image_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    text_similarity = text_vectors @ text_vectors.T
    image_similarity = image_vectors @ image_vectors.T
    combined = text_weight * text_similarity + image_weight * image_similarity
    groups = [class_group(as_str(item.get("code"))) for _, item, _ in items]

    ranked: dict[str, list[dict[str, Any]]] = {}
    for index, (entry_id, item, _) in enumerate(items):
        source_code = as_str(item.get("code"))
        candidates = []
        for candidate_index, (candidate_entry_id, candidate, _) in enumerate(items):
            candidate_code = as_str(candidate.get("code"))
            if candidate_index == index or candidate_code == source_code or groups[candidate_index] != groups[index]:
                continue
            candidates.append(
                {
                    "entry_id": candidate_entry_id,
                    "code": candidate_code,
                    "english_name": as_str(candidate.get("english_name")),
                    "chinese_name": as_str(candidate.get("chinese_name")),
                    "score": float(combined[index, candidate_index]),
                    "text_score": float(text_similarity[index, candidate_index]),
                    "image_score": float(image_similarity[index, candidate_index]),
                }
            )
        candidates.sort(key=lambda row: (-row["score"], row["code"], row["entry_id"]))
        if len(candidates) < top_k:
            raise RuntimeError(f"Only {len(candidates)} candidates available for {entry_id}; need top_k={top_k}")
        ranked[entry_id] = candidates[:top_k]
    return ranked


def update_payload(payload: dict[str, Any], ranked: dict[str, list[dict[str, Any]]]) -> int:
    updated = 0
    descriptions: dict[str, dict[str, Any]] = payload["description"]
    for entry_id, top_items in ranked.items():
        item = descriptions[entry_id]
        item["similar_chinese_classes"] = [candidate["chinese_name"] for candidate in top_items]
        item["similar_english_classes"] = [candidate["english_name"] for candidate in top_items]
        updated += 1
    return updated


def report_payload(
    args: argparse.Namespace,
    items: list[tuple[str, dict[str, Any], Path]],
    ranked: dict[str, list[dict[str, Any]]],
    classes_updated: int,
) -> dict[str, Any]:
    group_counts = {prefix: 0 for prefix in GROUP_PREFIXES}
    for _, item, _ in items:
        group_counts[class_group(as_str(item.get("code")))] += 1
    return {
        "source_wiki_path": args.knowledge_base,
        "model_path": args.model_name,
        "grouping_policy": "Top-k candidates are restricted to the same class-code prefix group: N04 or N05.",
        "weights": {"text_similarity": args.text_weight, "image_similarity": args.image_weight},
        "top_k": args.top_k,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "write_enabled": bool(args.write),
        "summary": {
            "total_entries": len(items),
            "N04": group_counts["N04"],
            "N05": group_counts["N05"],
            "classes_updated": classes_updated,
        },
        "classes": [
            {
                "entry_id": entry_id,
                "code": as_str(item.get("code")),
                "english_name": as_str(item.get("english_name")),
                "chinese_name": as_str(item.get("chinese_name")),
                "group": class_group(as_str(item.get("code"))),
                "reference_image": str(image_path),
                "similar": ranked[entry_id],
            }
            for entry_id, item, image_path in items
        ],
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_rankings(items: Iterable[tuple[str, dict[str, Any], Path]], ranked: dict[str, list[dict[str, Any]]], top_k: int) -> None:
    item_by_entry = {entry_id: item for entry_id, item, _ in items}
    for entry_id, top_items in ranked.items():
        source = item_by_entry[entry_id]
        source_group = class_group(as_str(source.get("code")))
        if len(top_items) != top_k:
            raise RuntimeError(f"Expected {top_k} similar classes for {entry_id}, got {len(top_items)}")
        for candidate in top_items:
            if candidate["entry_id"] == entry_id:
                raise RuntimeError(f"Self candidate selected for {entry_id}")
            if candidate["code"] == as_str(source.get("code")):
                raise RuntimeError(f"Same-code candidate selected for {entry_id}: {candidate['entry_id']}")
            if class_group(candidate["code"]) != source_group:
                raise RuntimeError(f"Cross-group candidate selected for {entry_id}: {candidate['code']}")


def main() -> int:
    args = parse_args()
    if args.text_weight < 0 or args.image_weight < 0 or args.text_weight + args.image_weight <= 0:
        raise RuntimeError("Similarity weights must be non-negative and have a positive sum")

    wiki_path = Path(args.knowledge_base)
    payload = load_payload(wiki_path)
    items = selected_items(payload, Path(args.image_dir), args.limit)
    if not items:
        raise RuntimeError("No wiki entries selected")
    print(f"selected_entries={len(items)}", flush=True)

    device = model_device(args.device)
    print(f"loading_model={args.model_name} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    texts = [build_similarity_text(item) for _, item, _ in items]
    image_paths = [image_path for _, _, image_path in items]
    text_vectors, image_vectors = encode_all(model, processor, device, texts, image_paths, args.batch_size)
    print(f"encoded_text_vectors={tuple(text_vectors.shape)} encoded_image_vectors={tuple(image_vectors.shape)}", flush=True)

    ranked = rank_similar(items, text_vectors, image_vectors, args.top_k, args.text_weight, args.image_weight)
    validate_rankings(items, ranked, args.top_k)
    classes_updated = update_payload(payload, ranked) if args.write else 0

    report = report_payload(args, items, ranked, classes_updated)
    dump_json(Path(args.report), report)
    if args.write:
        dump_json(wiki_path, payload)

    print(f"report={args.report}", flush=True)
    print(f"write_enabled={args.write} classes_updated={classes_updated}", flush=True)
    print("similar_classes_recompute_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
