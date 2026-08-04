from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_milvus_lite(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    from pymilvus import MilvusClient

    client = MilvusClient(uri=str(path))
    collections = {}
    for name in sorted(client.list_collections()):
        description = client.describe_collection(name)
        collections[name] = {
            "row_count": int(client.get_collection_stats(name).get("row_count", 0)),
            "fields": [field["name"] for field in description.get("fields", [])],
        }
    return {"path": str(path), "size_bytes": path.stat().st_size, "collections": collections}
