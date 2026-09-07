from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agrinet.common.contracts import Domain, ExperimentSpec
from agrinet.common.paths import experiment_root

EXPERIMENT_ID_PATTERN = re.compile(
    r"^(data|rag|vlm|vision)-[a-z0-9]+(?:-[a-z0-9]+){2,}-v[1-9][0-9]*$"
)


class ConfigError(ValueError):
    pass


def validate_experiment_id(experiment_id: str) -> None:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ConfigError(
            f"invalid experiment id {experiment_id!r}; expected lower-case kebab-case "
            "ending in -v<major>"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    return value


def load_experiment(experiment_id: str) -> ExperimentSpec:
    validate_experiment_id(experiment_id)
    domain = experiment_id.split("-", 1)[0]
    path = experiment_root() / domain / f"{experiment_id}.yaml"
    if not path.is_file():
        raise ConfigError(f"experiment not found: {experiment_id}")
    try:
        spec = ExperimentSpec.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid experiment {experiment_id}: {exc}") from exc
    if spec.id != experiment_id or spec.domain.value != domain:
        raise ConfigError(f"experiment identity does not match its path: {path}")
    return spec


def list_experiments(domain: Domain) -> list[ExperimentSpec]:
    root = experiment_root() / domain.value
    return [load_experiment(path.stem) for path in sorted(root.glob("*.yaml"))]


def parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"invalid override {raw!r}; expected key=value")
    key, raw_value = raw.split("=", 1)
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ConfigError(f"invalid override key: {key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid override value for {key!r}: {exc}") from exc
    return parts, value


def resolve_config(spec: ExperimentSpec, overrides: list[str] | None = None) -> dict[str, Any]:
    resolved = copy.deepcopy(spec.model_dump(mode="json"))
    for raw in overrides or []:
        parts, value = parse_override(raw)
        cursor = resolved
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                current = {}
                cursor[part] = current
            if not isinstance(current, dict):
                raise ConfigError(f"override traverses a non-mapping key: {raw!r}")
            cursor = current
        cursor[parts[-1]] = value
    return resolved
