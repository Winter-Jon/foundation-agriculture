from __future__ import annotations

from pathlib import Path
from typing import Any

from agrinet.data.contracts import PreparedSample
from agrinet.data.io import DataError, read_jsonl, write_jsonl_atomic


def prepare_records(source: Path, destination: Path) -> int:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in read_jsonl(source):
        sample_id = str(raw.get("sample_id") or raw.get("id") or "").strip()
        image = str(raw.get("query_image") or raw.get("image") or "").strip()
        domain = str(raw.get("task_domain") or raw.get("domain") or "").strip()
        if not sample_id or not image or not domain:
            raise DataError(f"missing sample_id, image, or task_domain at {source}:{line_number}")
        if sample_id in seen:
            raise DataError(f"duplicate sample_id {sample_id!r} at {source}:{line_number}")
        seen.add(sample_id)
        consumed = {"sample_id", "id", "query_image", "image", "task_domain", "domain", "label"}
        record = PreparedSample(
            sample_id=sample_id,
            query_image=image,
            task_domain=domain,
            label=raw.get("label"),
            metadata={key: value for key, value in raw.items() if key not in consumed},
        )
        records.append(record.model_dump(mode="json"))
    if not records:
        raise DataError(f"source contains no records: {source}")
    write_jsonl_atomic(destination, records)
    return len(records)
