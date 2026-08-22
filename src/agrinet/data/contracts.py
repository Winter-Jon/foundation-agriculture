from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from agrinet.common.contracts import StrictModel, StudentMessage


class PreparedSample(StrictModel):
    schema_version: Literal["agrinet.data.sample/v1"] = "agrinet.data.sample/v1"
    sample_id: str
    query_image: str
    task_domain: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeacherRecord(StrictModel):
    schema_version: Literal["agrinet.data.teacher/v1"] = "agrinet.data.teacher/v1"
    sample_id: str
    messages: list[StudentMessage]
    images: list[str] = Field(default_factory=list)
    provider: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConvertedSftRecord(StrictModel):
    schema_version: Literal["agrinet.sft.student/v1"] = "agrinet.sft.student/v1"
    sample_id: str
    messages: list[StudentMessage]
    images: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] | str = Field(default_factory=list)
    source_artifact_id: str


@runtime_checkable
class TeacherDataProvider(Protocol):
    name: str

    def generate(self, samples_path: str, output_path: str, **options: Any) -> None: ...
