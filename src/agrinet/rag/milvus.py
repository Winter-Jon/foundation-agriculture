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
    # Curated same-domain neighbours stored in the wiki class collection.
    # Return them with a hit so callers can compare alternatives.
    "similar_english_classes", "similar_chinese_classes",
    "public_description", "visual_descriptions",
]
IMAGE_FIELDS = [
    "image_id", "entry_id", "code", "english_name", "chinese_name",
    "source_dataset", "local_reference_image",
]
CATALOG_HOST_TERMS = {
    "apple", "apricot", "bean", "cherry", "corn", "cotton",
    "grape", "orange", "peach", "pepper", "potato", "rice",
    "soybean", "strawberry", "tomato", "wheat", "coffee",
}
CANDIDATE_COMPARE_PREFIX = "Compare public agricultural candidate classes and their host, organ, and symptom distinctions:"


class MilvusSiglipBackend:
    """Local SigLIP2 image retrieval backend for the AgriNet Milvus Lite index."""

    def __init__(self, db_path: Path, model_path: Path, device: str = "auto") -> None:
        self.db_path = db_path
        self.model_path = model_path
        self.client = MilvusClient(uri=str(db_path))
        self.class_collection = "agrinet_wiki_siglip2_classes"
        self.image_collection = "agrinet_wiki_siglip2_images"
        class_description = self.client.describe_collection(self.class_collection)
        # pymilvus exposes fields directly for Lite but nests them under
        # ``schema`` in other client/server combinations. Support both so text
        # retrieval always returns readable class metadata.
        class_schema = class_description.get("schema") or class_description
        self.class_fields = {str(field.get("name")) for field in class_schema.get("fields") or []}
        self._catalog_similar_classes = self._load_catalog_similar_classes()
        requested = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.device = torch.device("cpu" if requested == "auto" else requested)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        self.client.load_collection(self.class_collection)
        self.client.load_collection(self.image_collection)

    def _load_catalog_similar_classes(self) -> dict[str, dict[str, Any]]:
        """Load public curated neighbours when serving a pre-extension index.

        Existing Lite files predate the two similar-class scalar fields.  The
        canonical wiki remains the source used to build those fields, so this
        fallback preserves the same public evidence without rebuilding or
        mutating a live index.
        """
        try:
            wiki_path = self.db_path.parents[2] / "datasets/AgriNet-1K/wiki/base.json"
            payload = json.loads(wiki_path.read_text(encoding="utf-8"))
            descriptions = payload.get("description") or {}
            if not isinstance(descriptions, dict):
                return {}
            catalog: dict[str, dict[str, Any]] = {}
            for entry_id, item in descriptions.items():
                if not isinstance(item, dict):
                    continue
                description = item.get("description") or {}
                text_parts = [
                    str(value).strip() for key, value in sorted(description.items())
                    if str(key).startswith("content_") and str(value).strip()
                ] if isinstance(description, dict) else []
                visual_parts = [
                    str(value).strip() for value in (item.get("gemini_desc") or [])
                    if str(value).strip()
                ]
                name_text = " ".join(str(item.get(key) or "") for key in ("english_name", "chinese_name"))
                name_hosts = {term for term in CATALOG_HOST_TERMS if re.search(rf"\b{term}\b", name_text.lower())}
                def compatible(text: str) -> bool:
                    text_hosts = {term for term in CATALOG_HOST_TERMS if re.search(rf"\b{term}\b", text.lower())}
                    return not name_hosts or not text_hosts or bool(name_hosts & text_hosts)
                text_parts = [value for value in text_parts if compatible(value)]
                visual_parts = [value for value in visual_parts if compatible(value)]
                catalog[str(entry_id)] = {
                    "english_name": str(item.get("english_name") or "").strip(),
                    "chinese_name": str(item.get("chinese_name") or "").strip(),
                    "similar_english_classes": [str(value) for value in (item.get("similar_english_classes") or []) if str(value).strip()],
                    "similar_chinese_classes": [str(value) for value in (item.get("similar_chinese_classes") or []) if str(value).strip()],
                    # These are public catalogue facts, bounded here before
                    # they cross the retrieval HTTP boundary.
                    "public_description": text_parts[0][:420] if text_parts else "",
                    "visual_descriptions": [value[:300] for value in visual_parts[:2]],
                }
            return catalog
        except (OSError, json.JSONDecodeError, IndexError):
            return {}

    @property
    def _class_output_fields(self) -> list[str]:
        # Milvus rejects unknown output fields.  Restrict dynamically so an
        # already-serving old Lite index keeps working during the transition.
        return [field for field in CLASS_FIELDS if field in self.class_fields]

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
            collection_name=self.class_collection, filter=f"entry_id in [{quoted}]", output_fields=self._class_output_fields
        )
        normalized = [self._plain_row(row) for row in rows]
        return {str(row["entry_id"]): row for row in normalized}

    def _with_catalog_similar_classes(self, row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        entry_id = str(output.get("entry_id") or output.get("id") or "")
        fallback = self._catalog_similar_classes.get(entry_id, {})
        if not fallback:
            row_name = self._normalize_name(str(output.get("english_name") or output.get("chinese_name") or ""))
            if row_name:
                fallback = next(
                    (value for value in self._catalog_similar_classes.values()
                     if self._normalize_name(str(value.get("english_name") or value.get("chinese_name") or "")) == row_name),
                    {},
                )
        for key in ("similar_english_classes", "similar_chinese_classes", "visual_descriptions"):
            if not output.get(key) and fallback.get(key):
                output[key] = list(fallback[key])
        # The canonical wiki is authoritative for public evidence.  Always
        # replace a non-empty indexed description when a same-name canonical
        # record exists; keeping a stale index payload can bind another class's
        # description to this candidate and corrupt HCV adjudication.
        if fallback.get("public_description"):
            output["public_description"] = str(fallback["public_description"])
        return output

    def _plain_row(self, row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        for key in (
            "local_reference_images", "alias_en", "alias_cn",
            "similar_english_classes", "similar_chinese_classes",
            "visual_descriptions",
        ):
            value = output.get(key)
            if value is not None and not isinstance(value, list):
                output[key] = list(value)
        return self._with_catalog_similar_classes(output)

    def _text_hits(self, text: str, limit: int) -> list[dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.class_collection, data=[self._encode_text(text)], anns_field="text_vector",
            limit=limit, output_fields=self._class_output_fields,
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
        rows = self.client.query(collection_name=self.class_collection, filter="entry_id != ''", output_fields=self._class_output_fields, limit=1000)
        matches: dict[str, list[dict[str, Any]]] = {}
        for raw_row in rows:
            row = self._plain_row(raw_row)
            names = [row.get("english_name"), row.get("chinese_name"), *(row.get("alias_en") or []), *(row.get("alias_cn") or [])]
            if any(self._normalize_name(str(name or "")) == normalized for name in names):
                key = self._normalize_name(str(row.get("english_name") or row.get("chinese_name") or row.get("entry_id")))
                matches.setdefault(key, []).append({"score": 1.0, **row})
        # Multiple historical source datasets can expose the same public name.
        # The canonical agricultural wiki is the authoritative description
        # lineage; lexical entry_id ordering is not a safe tie-breaker because
        # it can select a cross-dataset image/description mismatch.
        selected: list[dict[str, Any]] = []
        for candidates in matches.values():
            selected.append(min(candidates, key=self._name_hit_priority))
        return sorted(selected, key=lambda item: str(item.get("entry_id")))[:top_k]

    @staticmethod
    def _name_hit_priority(row: dict[str, Any]) -> tuple[int, int, str]:
        source = str(row.get("source_dataset") or "")
        entry_id = str(row.get("entry_id") or "")
        canonical = int(source != "agri_disease_pest_wiki" and not entry_id.startswith("agri_disease_pest_wiki::"))
        compatible = int(not MilvusSiglipBackend._catalog_description_is_name_compatible(row))
        return canonical, compatible, entry_id

    def _candidate_compare_hits(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Resolve a public candidate set without a second unconstrained ANN search."""
        if CANDIDATE_COMPARE_PREFIX not in query:
            return []
        suffix = query.split(CANDIDATE_COMPARE_PREFIX, 1)[1]
        names = [value.strip() for value in suffix.split(";") if value.strip()]
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name in names:
            for row in self._name_hits(name, top_k=4):
                key = str(row.get("entry_id") or "")
                if not key or key in seen:
                    continue
                # Prefer the canonical AgriNet wiki record when duplicate
                # source datasets disagree on the same public class.
                if not self._catalog_description_is_name_compatible(row):
                    continue
                seen.add(key)
                rows.append(row)
                break
            if len(rows) >= top_k:
                break
        return rows

    @staticmethod
    def _catalog_description_is_name_compatible(row: dict[str, Any]) -> bool:
        description = str(row.get("public_description") or "").lower()
        name = " ".join(str(row.get(key) or "") for key in ("english_name", "chinese_name")).lower()
        if not description:
            return True
        # A description with a clearly different crop host is not safe as
        # discriminative evidence for this class.
        hosts = ["apple", "apricot", "bean", "carambola", "cherry", "citrus", "corn", "cotton", "grape", "orange", "peach", "pepper", "potato", "rice", "soybean", "strawberry", "tomato", "wheat", "coffee"]
        name_hosts = {host for host in hosts if re.search(rf"\b{host}\b", name)}
        desc_hosts = {host for host in hosts if re.search(rf"\b{host}\b", description)}
        return not name_hosts or not desc_hosts or bool(name_hosts & desc_hosts)

    def search(self, request: RagSearchRequest) -> list[dict[str, Any]]:
        kind = request.retrieval_type
        if kind in {"image", "image-to-class", "visual"}:
            return self._image_hits(request.query_image, request.top_k)
        if kind == "semantic":
            candidate_hits = self._candidate_compare_hits(request.query_text, request.top_k)
            if candidate_hits:
                return candidate_hits
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
