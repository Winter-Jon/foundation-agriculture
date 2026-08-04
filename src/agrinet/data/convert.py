from __future__ import annotations

from pathlib import Path

from agrinet.data.contracts import ConvertedSftRecord, TeacherRecord
from agrinet.data.io import read_jsonl, write_jsonl_atomic


def convert_teacher_records(source: Path, destination: Path, source_artifact_id: str) -> int:
    rows: list[dict] = []
    for _, raw in read_jsonl(source):
        teacher = TeacherRecord.model_validate(raw)
        converted = ConvertedSftRecord(
            sample_id=teacher.sample_id,
            messages=teacher.messages,
            images=teacher.images,
            tools=teacher.metadata.get("tools", []),
            source_artifact_id=source_artifact_id,
        )
        rows.append(converted.model_dump(mode="json"))
    write_jsonl_atomic(destination, rows)
    return len(rows)
