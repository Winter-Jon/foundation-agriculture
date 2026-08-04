from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class DataError(ValueError):
    pass


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    with stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise DataError(f"record at {path}:{line_number} must be an object")
            yield line_number, value


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
