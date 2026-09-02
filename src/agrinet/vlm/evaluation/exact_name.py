"""Versioned exact-name prediction scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agrinet.common.contracts import ArtifactRef, EvaluationResult


def evaluate_predictions(predictions: Path, model_artifact: ArtifactRef, output: Path) -> EvaluationResult:
    """Score JSONL predictions with the ``agrinet.exact-name/v1`` contract.

    Empty records are skipped. Rows without expected and predicted values are
    reported as failures, preserving the original evaluator semantics.
    """
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    correct = 0
    total = 0
    with predictions.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = str(row.get("expected") or row.get("label") or "").strip().casefold()
            predicted = str(row.get("predicted") or row.get("prediction") or "").strip().casefold()
            if not expected or not predicted:
                failures.append({"line": line_no, "reason": "missing expected or predicted"})
                continue
            matched = expected == predicted
            total += 1
            correct += int(matched)
            samples.append({
                "sample_id": row.get("sample_id", str(line_no)),
                "expected": expected,
                "predicted": predicted,
                "correct": matched,
            })
    result = EvaluationResult(
        protocol_version="agrinet.exact-name/v1",
        model_artifact=model_artifact,
        samples=samples,
        metrics={"accuracy": correct / total if total else 0.0, "evaluated": float(total)},
        failures=failures,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result
