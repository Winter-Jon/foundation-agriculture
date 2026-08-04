from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from agrinet.data.contracts import ConvertedSftRecord, PreparedSample, TeacherRecord
from agrinet.data.io import DataError, read_jsonl

SchemaKind = Literal["prepared", "teacher", "sft"]
MODELS: dict[SchemaKind, type[BaseModel]] = {
    "prepared": PreparedSample,
    "teacher": TeacherRecord,
    "sft": ConvertedSftRecord,
}


def validate_records(path: Path, kind: SchemaKind) -> int:
    seen: set[str] = set()
    count = 0
    for line_number, raw in read_jsonl(path):
        try:
            record = MODELS[kind].model_validate(raw)
        except ValidationError as exc:
            raise DataError(f"invalid {kind} record at {path}:{line_number}: {exc}") from exc
        sample_id = str(record.sample_id)
        if sample_id in seen:
            raise DataError(f"duplicate sample_id {sample_id!r} at {path}:{line_number}")
        seen.add(sample_id)
        count += 1
    if count == 0:
        raise DataError(f"file contains no records: {path}")
    return count
