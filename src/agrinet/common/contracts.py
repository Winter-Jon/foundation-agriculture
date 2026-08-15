from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Domain(StrEnum):
    DATA = "data"
    RAG = "rag"
    VLM = "vlm"


class Lifecycle(StrEnum):
    ACTIVE = "active"
    BASELINE = "baseline"
    ARCHIVED = "archived"


class DisplayName(StrictModel):
    en: str
    zh: str


class ExperimentSpec(StrictModel):
    schema_version: Literal["agrinet.experiment/v1"] = "agrinet.experiment/v1"
    id: str
    display_name: DisplayName
    domain: Domain
    task: str
    description: str = ""
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    components: dict[str, str] = Field(default_factory=dict)
    runtime_profile: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class ArtifactRef(StrictModel):
    schema_version: str
    artifact_id: str
    artifact_type: str
    path: Path
    producer_run_id: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeInfo(StrictModel):
    python: str
    pytorch: str | None = None
    cuda: str | None = None
    gpu: list[str] = Field(default_factory=list)
    dependency_lock_sha256: str | None = None


class RunManifest(StrictModel):
    schema_version: Literal["agrinet.run/v1"] = "agrinet.run/v1"
    experiment_id: str
    run_id: str
    git_commit: str
    git_tag: str | None = None
    git_dirty: bool
    resolved_config: dict[str, Any]
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    runtime: RuntimeInfo
    local_pid: int | None = None
    # Historical manifests may retain a scheduler identifier after migration.
    slurm_job_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: Literal["pending", "running", "complete", "failed"]
    exit_code: int | None = None
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    checkpoint_paths: list[Path] = Field(default_factory=list)
    metrics_path: Path | None = None


class RagEvidence(StrictModel):
    artifact_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchRequest(StrictModel):
    schema_version: Literal["agrinet.rag.search/v1"] = "agrinet.rag.search/v1"
    retrieval_type: str
    query_image: Path
    query_text: str = ""
    top_k: int = Field(default=5, ge=1)
    ranker: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class RagSearchResponse(StrictModel):
    schema_version: Literal["agrinet.rag.search/v1"] = "agrinet.rag.search/v1"
    evidence: list[RagEvidence]


class StudentMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class StudentSftRecord(StrictModel):
    schema_version: Literal["agrinet.sft.student/v1"] = "agrinet.sft.student/v1"
    sample_id: str
    messages: list[StudentMessage]
    images: list[Path] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    source_artifact: ArtifactRef


class EvaluationResult(StrictModel):
    schema_version: Literal["agrinet.evaluation/v1"] = "agrinet.evaluation/v1"
    protocol_version: str
    model_artifact: ArtifactRef
    samples: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
