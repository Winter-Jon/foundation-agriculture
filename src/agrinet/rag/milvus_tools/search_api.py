#!/usr/bin/env python3
"""HTTP API for SigLIP2 query embedding and AgriNet wiki Milvus search."""

from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image
from pymilvus import MilvusClient

from .search_agrinet_wiki import (
    encode_sparse_text,
    encode_image_queries,
    encode_pil_image_queries,
    encode_text_queries,
    load_siglip,
    resolve_collection_names,
    run_hybrid_search,
    run_search,
    simplify_hits,
)


PRESETS = {
    "balanced": {
        "ranker": "weighted",
        "text_weight": 0.35,
        "image_weight": 0.45,
        "sparse_weight": 0.00,
        "description": "Default dense multimodal search. Prefer visual evidence slightly while keeping semantic text signal; sparse name lookup is excluded.",
    },
    "visual": {
        "ranker": "weighted",
        "text_weight": 0.10,
        "image_weight": 0.80,
        "sparse_weight": 0.00,
        "description": "Image-first diagnosis when query images are the primary evidence.",
    },
    "semantic": {
        "ranker": "weighted",
        "text_weight": 0.70,
        "image_weight": 0.00,
        "sparse_weight": 0.00,
        "description": "Dense text-first semantic search over class names, aliases, and wiki descriptions; sparse name lookup is excluded.",
    },
    "name": {
        "ranker": "weighted",
        "text_weight": 0.00,
        "image_weight": 0.00,
        "sparse_weight": 1.00,
        "description": "Exact or near-exact class-name and alias lookup only.",
    },
    "rrf": {
        "ranker": "rrf",
        "text_weight": 0.35,
        "image_weight": 0.45,
        "sparse_weight": 0.00,
        "description": "Dense text-image rank-fusion fallback; sparse name lookup is excluded.",
    },
}


class SearchService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.collection = args.collection
        self.default_top_k = args.top_k
        self.default_ranker = args.ranker
        self.default_text_weight = args.text_weight
        self.default_image_weight = args.image_weight
        self.default_sparse_weight = args.sparse_weight
        self.client = MilvusClient(uri=args.lite_db if args.mode == "lite" else args.milvus_uri)
        self.class_collection, self.image_collection = resolve_collection_names(self.client, args.collection, args.class_collection, args.image_collection)
        self.client.load_collection(self.class_collection)
        if self.image_collection != self.class_collection:
            self.client.load_collection(self.image_collection)
        self.processor, self.model, self.device = load_siglip(args.model_name, args.device)
        self.model_name = args.model_name
        self.mode = args.mode
        self.lite_db = args.lite_db
        self.milvus_uri = args.milvus_uri

    def health(self) -> dict[str, Any]:
        class_stats = self.client.get_collection_stats(self.class_collection)
        image_stats = self.client.get_collection_stats(self.image_collection)
        return {
            "ok": True,
            "collection": self.collection,
            "class_collection": self.class_collection,
            "image_collection": self.image_collection,
            "class_stats": class_stats,
            "image_stats": image_stats,
            "model_name": self.model_name,
            "device": str(self.device),
            "mode": self.mode,
            "lite_db": self.lite_db,
            "milvus_uri": self.milvus_uri,
            "presets": PRESETS,
        }

    def search_text(self, texts: list[str], top_k: int, expr: str) -> list[dict[str, Any]]:
        vector = encode_text_queries(self.processor, self.model, self.device, texts)
        hits = run_search(self.client, self.class_collection, "text_vector", vector, top_k, expr)
        return simplify_hits(hits)

    def search_image_paths(self, image_paths: list[Path], top_k: int, expr: str) -> list[dict[str, Any]]:
        vector = encode_image_queries(self.processor, self.model, self.device, image_paths)
        hits = run_search(self.client, self.image_collection, "image_vector", vector, top_k, "" if self.image_collection != self.class_collection else expr)
        return simplify_hits(hits)

    def search_image_base64(self, image_base64_values: list[str], top_k: int, expr: str) -> list[dict[str, Any]]:
        images = []
        for image_base64 in image_base64_values:
            image_bytes = base64.b64decode(image_base64, validate=True)
            images.append(Image.open(BytesIO(image_bytes)))
        vector = encode_pil_image_queries(self.processor, self.model, self.device, images)
        hits = run_search(self.client, self.image_collection, "image_vector", vector, top_k, "" if self.image_collection != self.class_collection else expr)
        return simplify_hits(hits)

    def search_hybrid(
        self,
        texts: list[str],
        image_paths: list[Path],
        image_base64_values: list[str],
        top_k: int,
        expr: str,
        ranker: str,
        text_weight: float,
        image_weight: float,
        sparse_weight: float,
    ) -> list[dict[str, Any]]:
        vectors: dict[str, Any] = {}
        query_text = "\n".join(texts)
        if texts and text_weight > 0:
            vectors["text_vector"] = encode_text_queries(self.processor, self.model, self.device, texts)
        if texts and sparse_weight > 0:
            vectors["sparse_vector"] = encode_sparse_text(query_text)
        if image_paths:
            vectors["image_vector"] = encode_image_queries(self.processor, self.model, self.device, image_paths)
        elif image_base64_values:
            images = []
            for image_base64 in image_base64_values:
                image_bytes = base64.b64decode(image_base64, validate=True)
                images.append(Image.open(BytesIO(image_bytes)))
            vectors["image_vector"] = encode_pil_image_queries(self.processor, self.model, self.device, images)
        hits = run_hybrid_search(self.client, self.class_collection, self.image_collection, vectors, query_text, top_k, expr, ranker, text_weight, image_weight, sparse_weight)
        return simplify_hits(hits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--collection", default="agrinet_wiki_siglip2")
    parser.add_argument("--class-collection", default="", help="Override class collection name. Default: <collection>_classes.")
    parser.add_argument("--image-collection", default="", help="Override image collection name. Default: <collection>_images.")
    parser.add_argument("--model-name", default="models/siglip2-so400m-patch16-naflex")
    parser.add_argument("--mode", choices=("standalone", "lite"), default="lite")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--lite-db", default="outputs/milvus/agrinet_wiki_lite.db")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ranker", choices=("weighted", "rrf"), default="weighted")
    parser.add_argument("--text-weight", type=float, default=0.35)
    parser.add_argument("--image-weight", type=float, default=0.45)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    return parser.parse_args()


def make_handler(service: SearchService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgriNetMilvusSearch/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/health":
                self.write_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
                return
            self.write_json(service.health())

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self.read_json()
                top_k = int(body.get("top_k") or service.default_top_k)
                expr = str(body.get("filter") or "")
                preset_path = self.preset_from_path(parsed.path)
                if preset_path:
                    body = {**body, "preset": preset_path}
                params = self.search_params(body)
                ranker = params["ranker"]
                text_weight = params["text_weight"]
                image_weight = params["image_weight"]
                sparse_weight = params["sparse_weight"]
                if parsed.path == "/search" or preset_path:
                    texts = self.collect_strings(body, "text", "texts")
                    image_paths, image_base64_values = self.collect_images(body, required=False)
                    if preset_path == "name" and not texts:
                        raise ValueError("Field 'text' or 'texts' is required for /search/name")
                    if preset_path == "visual" and not image_paths and not image_base64_values:
                        raise ValueError("Field 'image_path', 'image_paths', 'image_base64', or 'image_base64_list' is required for /search/visual")
                    if preset_path == "semantic" and not texts:
                        raise ValueError("Field 'text' or 'texts' is required for /search/semantic")
                    if not texts and not image_paths and not image_base64_values:
                        raise ValueError("Provide 'text', 'image_path', or 'image_base64'")
                    if body.get("separate"):
                        result: dict[str, Any] = {}
                        if texts:
                            result["text_vector"] = service.search_text(texts, top_k, expr)
                        if image_paths:
                            result["image_vector"] = service.search_image_paths(image_paths, top_k, expr)
                        elif image_base64_values:
                            result["image_vector"] = service.search_image_base64(image_base64_values, top_k, expr)
                        self.write_json(result)
                    else:
                        self.write_json({"hybrid": service.search_hybrid(texts, image_paths, image_base64_values, top_k, expr, ranker, text_weight, image_weight, sparse_weight)})
                    return
                self.write_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            except Exception as exc:
                self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

        def preset_from_path(self, path: str) -> str:
            prefix = "/search/"
            if not path.startswith(prefix):
                return ""
            preset_name = path[len(prefix) :]
            return preset_name if preset_name in PRESETS else ""

        def handle_image_search(self, body: dict[str, Any], top_k: int, expr: str) -> list[dict[str, Any]]:
            image_paths, image_base64_values = self.collect_images(body)
            if image_paths:
                return service.search_image_paths(image_paths, top_k, expr)
            return service.search_image_base64(image_base64_values, top_k, expr)

        def collect_images(self, body: dict[str, Any], required: bool = True) -> tuple[list[Path], list[str]]:
            image_paths = [Path(value) for value in self.collect_strings(body, "image_path", "image_paths")]
            image_base64_values = self.collect_strings(body, "image_base64", "image_base64_list")
            if image_paths and image_base64_values:
                raise ValueError("Use either image paths or base64 images in one request, not both")
            if image_paths:
                for image_path in image_paths:
                    if not image_path.exists():
                        raise ValueError(f"Image does not exist: {image_path}")
                return image_paths, []
            if image_base64_values:
                return [], image_base64_values
            if required:
                raise ValueError("Field 'image_path', 'image_paths', 'image_base64', or 'image_base64_list' is required")
            return [], []

        def collect_strings(self, body: dict[str, Any], single_key: str, list_key: str) -> list[str]:
            values: list[str] = []
            single_value = body.get(single_key)
            if single_value:
                values.append(str(single_value))
            list_value = body.get(list_key)
            if list_value is not None:
                if not isinstance(list_value, list):
                    raise ValueError(f"Field '{list_key}' must be a list")
                values.extend(str(value) for value in list_value if str(value).strip())
            return [value.strip() for value in values if value.strip()]

        def get_value(self, body: dict[str, Any], key: str, default: Any) -> Any:
            value = body.get(key, None)
            return default if value is None else value

        def search_params(self, body: dict[str, Any]) -> dict[str, Any]:
            preset_name = str(body.get("preset") or "balanced")
            if preset_name not in PRESETS:
                raise ValueError(f"Unknown preset: {preset_name}. Available presets: {sorted(PRESETS)}")
            params = dict(PRESETS[preset_name])
            params.pop("description", None)
            params["ranker"] = self.get_value(body, "ranker", params["ranker"])
            params["text_weight"] = float(self.get_value(body, "text_weight", params["text_weight"]))
            params["image_weight"] = float(self.get_value(body, "image_weight", params["image_weight"]))
            params["sparse_weight"] = float(self.get_value(body, "sparse_weight", params["sparse_weight"]))
            if params["ranker"] not in {"weighted", "rrf"}:
                raise ValueError("Field 'ranker' must be 'weighted' or 'rrf'")
            if preset_name != "name":
                params["sparse_weight"] = 0.0
            if params["ranker"] == "rrf":
                params["sparse_weight"] = 0.0
            return params

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def write_error(self, status: HTTPStatus, message: str) -> None:
            self.write_json({"ok": False, "error": message}, status)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main() -> int:
    args = parse_args()
    service = SearchService(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"serving host={args.host} port={args.port} collection={args.collection}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
