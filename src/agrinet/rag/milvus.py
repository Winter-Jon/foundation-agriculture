from __future__ import annotations

import json
import re
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

    def _encode_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query_text is required for text retrieval")
        inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.get_text_features(
                **{key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}}
            )
            features = getattr(output, "pooler_output", output)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].float().cpu().tolist()

    def _class_rows(self, entry_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not entry_ids:
            return {}
        quoted = ", ".join(json.dumps(value) for value in entry_ids)
        rows = self.client.query(
            collection_name=self.class_collection, filter=f"entry_id in [{quoted}]", output_fields=CLASS_FIELDS
        )
        normalized = [self._plain_row(row) for row in rows]
        return {str(row["entry_id"]): row for row in normalized}

    @staticmethod
    def _plain_row(row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        for key in ("local_reference_images", "alias_en", "alias_cn"):
            value = output.get(key)
            if value is not None and not isinstance(value, list):
                output[key] = list(value)
        return output

    def _text_hits(self, text: str, limit: int) -> list[dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.class_collection, data=[self._encode_text(text)], anns_field="text_vector",
            limit=limit, output_fields=CLASS_FIELDS,
        )[0]
        return [{"score": float(hit.get("distance") or 0.0), **self._plain_row(dict(hit.get("entity") or {}))} for hit in hits]

    def _image_hits(self, path: Path, limit: int) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        hits = self.client.search(
            collection_name=self.image_collection, data=[self._encode_image(path)], anns_field="image_vector",
            limit=max(limit * 8, 32), output_fields=IMAGE_FIELDS,
        )[0]
        best: dict[str, dict[str, Any]] = {}
        for hit in hits:
            entity = dict(hit.get("entity") or {}); entry_id = str(entity.get("entry_id") or "")
            score = float(hit.get("distance") or 0.0)
            if entry_id and (entry_id not in best or score > best[entry_id]["score"]):
                best[entry_id] = {"score": score, "matched_image": entity.get("local_reference_image")}
        rows = self._class_rows(list(best))
        output = [{"id": entry_id, **values, **rows[entry_id]} for entry_id, values in best.items() if entry_id in rows]
        return sorted(output, key=lambda item: (-item["score"], item["id"]))[:limit]

    @staticmethod
    def _fuse(text_hits: list[dict[str, Any]], image_hits: list[dict[str, Any]], top_k: int, ranker: str, text_weight: float, image_weight: float) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for source, hits, weight in (("text", text_hits, text_weight), ("image", image_hits, image_weight)):
            for rank, hit in enumerate(hits, 1):
                entry_id = str(hit.get("entry_id") or hit.get("id") or "")
                if not entry_id:
                    continue
                item = fused.setdefault(entry_id, {**hit, "score": 0.0, "component_scores": {}})
                contribution = 1.0 / (60 + rank) if ranker == "rrf" else weight * float(hit.get("score") or 0.0)
                item["score"] += contribution; item["component_scores"][source] = contribution
                if source == "image" and hit.get("matched_image"):
                    item["matched_image"] = hit["matched_image"]
        return sorted(fused.values(), key=lambda item: (-float(item["score"]), str(item.get("entry_id") or item.get("id"))))[:top_k]

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))

    def _name_hits(self, query: str, top_k: int) -> list[dict[str, Any]]:
        normalized = self._normalize_name(query)
        if not normalized:
            raise ValueError("query_text is required for name retrieval")
        rows = self.client.query(collection_name=self.class_collection, filter="entry_id != ''", output_fields=CLASS_FIELDS, limit=1000)
        matches: dict[str, dict[str, Any]] = {}
        for raw_row in rows:
            row = self._plain_row(raw_row)
            names = [row.get("english_name"), row.get("chinese_name"), *(row.get("alias_en") or []), *(row.get("alias_cn") or [])]
            if any(self._normalize_name(str(name or "")) == normalized for name in names):
                key = self._normalize_name(str(row.get("english_name") or row.get("chinese_name") or row.get("entry_id")))
                matches.setdefault(key, {"score": 1.0, **row})
        return sorted(matches.values(), key=lambda item: str(item.get("entry_id")))[:top_k]

    def search(self, request: RagSearchRequest) -> list[dict[str, Any]]:
        kind = request.retrieval_type
        if kind in {"image", "image-to-class", "visual"}:
            return self._image_hits(request.query_image, request.top_k)
        if kind == "semantic":
            return self._text_hits(request.query_text, request.top_k)
        if kind == "name":
            return self._name_hits(request.query_text, request.top_k)
        if kind not in {"balanced", "rrf"}:
            raise ValueError(f"unsupported retrieval_type: {kind}")
        overfetch = max(request.top_k * 8, 32)
        text_hits = self._text_hits(request.query_text, overfetch)
        image_hits = self._image_hits(request.query_image, overfetch)
        ranker = "rrf" if kind == "rrf" else (request.ranker or "weighted")
        text_weight = float(request.weights.get("text", 0.5)); image_weight = float(request.weights.get("image", 0.5))
        return self._fuse(text_hits, image_hits, request.top_k, ranker, text_weight, image_weight)
