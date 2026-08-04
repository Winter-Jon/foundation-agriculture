from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoModel, AutoProcessor

from agrinet.common.contracts import RagSearchRequest


CLASS_FIELDS = [
    "entry_id", "code", "english_name", "chinese_name",
    "source_dataset", "local_reference_images", "alias_en", "alias_cn",
]
IMAGE_FIELDS = [
    "image_id", "entry_id", "code", "english_name", "chinese_name",
    "source_dataset", "local_reference_image",
]


class MilvusSiglipBackend:
    """Local SigLIP2 image retrieval backend for the AgriNet Milvus Lite index."""

    def __init__(self, db_path: Path, model_path: Path, device: str = "auto") -> None:
        self.db_path = db_path
        self.model_path = model_path
        self.client = MilvusClient(uri=str(db_path))
        self.class_collection = "agrinet_wiki_siglip2_classes"
        self.image_collection = "agrinet_wiki_siglip2_images"
        requested = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if requested == "auto" else requested)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        self.client.load_collection(self.class_collection)
        self.client.load_collection(self.image_collection)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "database": str(self.db_path),
            "device": str(self.device),
            "collections": {
                self.class_collection: self.client.get_collection_stats(self.class_collection)["row_count"],
                self.image_collection: self.client.get_collection_stats(self.image_collection)["row_count"],
            },
        }

    def _encode_image(self, path: Path) -> list[float]:
        with Image.open(path) as image:
            inputs = self.processor(
                text=[""], images=[image.convert("RGB")], padding="max_length", return_tensors="pt"
            ).to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            features = output.image_embeds
            features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].float().cpu().tolist()

    def search(self, request: RagSearchRequest) -> list[dict[str, Any]]:
        if request.retrieval_type not in {"image", "image-to-class"}:
            raise ValueError(f"unsupported retrieval_type: {request.retrieval_type}")
        if not request.query_image.is_file():
            raise FileNotFoundError(request.query_image)
        vector = self._encode_image(request.query_image)
        hits = self.client.search(
            collection_name=self.image_collection, data=[vector], anns_field="image_vector",
            limit=max(request.top_k * 8, 32), output_fields=IMAGE_FIELDS,
        )[0]
        best: dict[str, dict[str, Any]] = {}
        for hit in hits:
            entity = dict(hit.get("entity") or {})
            entry_id = str(entity.get("entry_id") or "")
            score = float(hit.get("distance") or 0.0)
            if entry_id and (entry_id not in best or score > best[entry_id]["score"]):
                best[entry_id] = {"score": score, "matched_image": entity.get("local_reference_image")}
        ids = list(best)
        if not ids:
            return []
        quoted = ", ".join(json.dumps(value) for value in ids)
        rows = self.client.query(
            collection_name=self.class_collection, filter=f"entry_id in [{quoted}]", output_fields=CLASS_FIELDS
        )
        output = []
        for row in rows:
            entry_id = str(row["entry_id"])
            output.append({"id": entry_id, **best[entry_id], **row})
        return sorted(output, key=lambda item: (-item["score"], item["id"]))[: request.top_k]
