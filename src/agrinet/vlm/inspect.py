from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_model(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    shards = sorted(path.glob("*.safetensors"))
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    return {
        "path": str(path),
        "loadable_layout": bool(config is not None and shards),
        "architecture": (config or {}).get("architectures", []),
        "shards": [{"name": item.name, "bytes": item.stat().st_size} for item in shards],
    }
