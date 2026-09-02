"""Predeclared, fail-closed gates for staged M1 Direct collection."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from agrinet.data.io import DataError

CELLS = tuple(
    (question_type, language, domain)
    for question_type in ("open", "option")
    for language in ("en", "zh")
    for domain in ("disease", "pest")
)
OPEN_PEST_CELLS = (("open", "en", "pest"), ("open", "zh", "pest"))


def load_gates(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "agrinet.m1-direct-acceptance-gates/v1":
        raise DataError(f"invalid M1 Direct gate configuration: {path}")
    return payload


def cell_of(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
    return tuple(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))


def _hard_errors(row: dict[str, Any], zero_tolerance: set[str]) -> set[str]:
    errors = {str(value) for value in row.get("errors") or []}
    if str(row.get("delivery_status") or "") == "unknown":
        errors.add("unknown_delivery")
    return errors & zero_tolerance


def _rejection_errors(row: dict[str, Any], gate: dict[str, Any]) -> set[str]:
    """Return defects that reject this one row, not its whole cohort.

    Older immutable gate files do not have ``rejection_errors``.  Falling back
    to ``zero_tolerance`` preserves their original semantics.
    """
    errors = {str(value) for value in row.get("errors") or []}
    if str(row.get("delivery_status") or "") == "unknown":
        errors.add("unknown_delivery")
    policy = set(gate.get("rejection_errors", gate.get("zero_tolerance", [])))
    return errors & policy


def _accepted_under_policy(row: dict[str, Any], gate: dict[str, Any], *, field: str = "accepted") -> bool:
    return bool(row.get(field)) and not _rejection_errors(row, gate)


def _meets_rate(value: float, threshold: float, *, strictly_greater: bool = False) -> bool:
    """Evaluate a declared rate threshold without obscuring its boundary."""
    return value > threshold if strictly_greater else value >= threshold


def validate_sample_preflight(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    attempts = int(gate["attempts"])
    zero = set(gate["zero_tolerance"])
    domains = Counter(cell_of(row)[2] for row in rows)
    accepted = sum(_accepted_under_policy(row, gate) for row in rows)
    teacher_correct = sum(row.get("teacher_correct") is True for row in rows)
    auditor_correct = sum(row.get("auditor_correct") is True for row in rows)
    reasoning_passed = sum(row.get("reasoning_checks_passed") is True for row in rows)
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    rejected = Counter(error for row in rows for error in _rejection_errors(row, gate))
    checks = {
        "exact_attempts": len(rows) == attempts,
        "required_domains": set(domains) == set(gate["required_domains"]),
        "exact_accepted": accepted == int(gate["accepted"]),
        "acceptance_rate": accepted / attempts >= float(gate["minimum_acceptance_rate"]) if attempts else False,
        "teacher_accuracy": teacher_correct / attempts >= float(gate["minimum_teacher_answer_accuracy"]) if attempts else False,
        "auditor_accuracy": auditor_correct / attempts >= float(gate["minimum_blind_auditor_accuracy"]) if attempts else False,
        "reasoning_checks": reasoning_passed == attempts if gate.get("require_all_reasoning_checks") else True,
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {"schema_version": "agrinet.m1-direct-sample-preflight-gate/v1", "attempts": len(rows), "accepted": accepted, "domains": dict(domains), "hard_errors": hard, "rejection_reasons": dict(sorted(rejected.items())), "checks": checks, "passed": all(checks.values())}


def validate_stratified_pilot(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    per_cell = int(gate["attempts_per_cell"]); expected = per_cell * len(CELLS)
    zero = set(gate["zero_tolerance"]); counts = Counter(cell_of(row) for row in rows)
    accepted = Counter(cell_of(row) for row in rows if _accepted_under_policy(row, gate))
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    rejected = Counter(error for row in rows for error in _rejection_errors(row, gate))
    teacher_correct = sum(row.get("teacher_correct") is True for row in rows)
    auditor_correct = sum(row.get("auditor_correct") is True for row in rows)
    reasoning = sum(row.get("reasoning_checks_passed") is True for row in rows)
    accepted_total = sum(accepted.values())
    cell_checks = {
        "/".join(cell): {
            "attempts": counts[cell], "accepted": accepted[cell],
            "passed": counts[cell] == per_cell
            and accepted[cell] >= int(gate["minimum_accepted_per_cell"])
            and accepted[cell] / per_cell >= float(gate["minimum_acceptance_rate_per_cell"]),
        }
        for cell in CELLS
    }
    checks = {
        "exact_attempts": len(rows) == expected,
        "all_cells_pass": all(value["passed"] for value in cell_checks.values()),
        "overall_acceptance_rate": _meets_rate(
            accepted_total / expected,
            float(gate["minimum_overall_acceptance_rate"]),
            strictly_greater=bool(gate.get("overall_acceptance_rate_strictly_greater", False)),
        ),
        "teacher_accuracy": teacher_correct / expected >= float(gate["minimum_teacher_answer_accuracy"]),
        "auditor_accuracy": auditor_correct / expected >= float(gate["minimum_blind_auditor_accuracy"]),
        "reasoning_check_rate": reasoning / expected >= float(gate["minimum_reasoning_check_rate"]),
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {"schema_version": "agrinet.m1-direct-stratified-pilot-gate/v1", "attempts": len(rows), "accepted": accepted_total, "cells": cell_checks, "hard_errors": hard, "rejection_reasons": dict(sorted(rejected.items())), "checks": checks, "passed": all(checks.values())}


def validate_open_pest_preflight(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    """Validate the diagnostic Open/pest preflight without relaxing Level-2."""
    per_cell = int(gate["attempts_per_cell"]); expected = per_cell * len(OPEN_PEST_CELLS)
    zero = set(gate["zero_tolerance"]); counts = Counter(cell_of(row) for row in rows)
    accepted = Counter(cell_of(row) for row in rows if row.get("accepted"))
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    teacher_correct = sum(row.get("teacher_correct") is True for row in rows)
    auditor_correct = sum(row.get("auditor_correct") is True for row in rows)
    reasoning = sum(row.get("reasoning_checks_passed") is True for row in rows)
    accepted_total = sum(accepted.values())
    cell_checks = {
        "/".join(cell): {
            "attempts": counts[cell], "accepted": accepted[cell],
            "passed": counts[cell] == per_cell
            and accepted[cell] >= int(gate["minimum_accepted_per_cell"])
            and accepted[cell] / per_cell >= float(gate["minimum_acceptance_rate_per_cell"]),
        }
        for cell in OPEN_PEST_CELLS
    }
    checks = {
        "exact_attempts": len(rows) == expected,
        "only_target_cells": set(counts) <= set(OPEN_PEST_CELLS),
        "all_cells_pass": all(value["passed"] for value in cell_checks.values()),
        "overall_acceptance_rate": accepted_total / expected >= float(gate["minimum_overall_acceptance_rate"]),
        "teacher_accuracy": teacher_correct / expected >= float(gate["minimum_teacher_answer_accuracy"]),
        "auditor_accuracy": auditor_correct / expected >= float(gate["minimum_blind_auditor_accuracy"]),
        "reasoning_check_rate": reasoning / expected >= float(gate["minimum_reasoning_check_rate"]),
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {
        "schema_version": "agrinet.m1-direct-open-pest-preflight-gate/v1",
        "attempts": len(rows), "accepted": accepted_total, "cells": cell_checks,
        "hard_errors": hard, "checks": checks, "passed": all(checks.values()),
    }


def validate_open_pest_distinguishability_screen(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before promoting screened images to teacher collection.

    The model remains label blind; correctness is aligned only in this private
    gate.  A screen result must show actual visual exclusions for all three
    alternatives, rather than merely report a high-confidence class choice.
    """
    per_cell = int(gate["candidates_per_cell"]); expected = per_cell * len(OPEN_PEST_CELLS)
    zero = set(gate["zero_tolerance"]); counts = Counter(cell_of(row) for row in rows)
    accepted = Counter(cell_of(row) for row in rows if row.get("passed"))
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    cell_checks = {
        "/".join(cell): {
            "attempts": counts[cell], "passed_images": accepted[cell],
            "passed": counts[cell] == per_cell and accepted[cell] >= int(gate["minimum_passed_per_cell"]),
        }
        for cell in OPEN_PEST_CELLS
    }
    checks = {
        "exact_attempts": len(rows) == expected,
        "only_target_cells": set(counts) <= set(OPEN_PEST_CELLS),
        "all_cells_pass": all(value["passed"] for value in cell_checks.values()),
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {
        "schema_version": "agrinet.m1-direct-open-pest-distinguishability-screen-gate/v1",
        "attempts": len(rows), "passed_images": sum(accepted.values()),
        "cells": cell_checks, "hard_errors": hard, "checks": checks, "passed": all(checks.values()),
        "passed_rows": [{"sample_id": row["sample_id"]} for row in rows if row.get("passed")],
    }


def validate_single_cell_distinguishability_screen(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    """Validate a declared one-cell screen before a replacement preflight."""
    cell = tuple(str(value) for value in gate["cell"])
    candidates = int(gate["candidates"]); zero = set(gate["zero_tolerance"])
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    passed = [row for row in rows if row.get("passed") is True]
    checks = {
        "exact_attempts": len(rows) == candidates,
        "only_declared_cell": all(cell_of(row) == cell for row in rows),
        "minimum_passed_images": len(passed) >= int(gate["minimum_passed_images"]),
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {
        "schema_version": "agrinet.m1-direct-single-cell-screen-gate/v1",
        "cell": "/".join(cell), "attempts": len(rows), "passed_images": len(passed),
        "hard_errors": hard, "checks": checks, "passed": all(checks.values()),
        "passed_rows": [{"sample_id": row["sample_id"]} for row in passed],
    }


def validate_stratified_distinguishability_screen(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    """Validate an eight-cell, label-blind candidate-distinguishability screen."""
    per_cell = int(gate["candidates_per_cell"]); expected = per_cell * len(CELLS)
    zero = set(gate["zero_tolerance"]); counts = Counter(cell_of(row) for row in rows)
    accepted = Counter(cell_of(row) for row in rows if _accepted_under_policy(row, gate, field="passed"))
    hard = sorted({error for row in rows for error in _hard_errors(row, zero)})
    rejected = Counter(error for row in rows for error in _rejection_errors(row, gate))
    known_terminal = sum(str(row.get("delivery_status") or "") == "known" for row in rows)
    cell_checks = {
        "/".join(cell): {
            "attempts": counts[cell], "passed_images": accepted[cell],
            "passed": counts[cell] == per_cell and accepted[cell] >= int(gate["minimum_passed_per_cell"]),
        }
        for cell in CELLS
    }
    checks = {
        "exact_attempts": len(rows) == expected,
        "only_declared_cells": set(counts) <= set(CELLS),
        "all_cells_pass": all(value["passed"] for value in cell_checks.values()),
        "known_delivery_rate": _meets_rate(
            known_terminal / expected,
            float(gate.get("minimum_known_delivery_rate", 1.0)),
            strictly_greater=bool(gate.get("known_delivery_rate_strictly_greater", False)),
        ) if expected else False,
        "acceptance_rate": _meets_rate(
            sum(accepted.values()) / expected,
            float(gate.get("minimum_acceptance_rate", 0.0)),
            strictly_greater=bool(gate.get("acceptance_rate_strictly_greater", False)),
        ) if expected else False,
        "zero_tolerance": not hard,
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in rows}) == len(rows) and all(row.get("sample_id") for row in rows),
    }
    return {
        "schema_version": "agrinet.m1-direct-stratified-distinguishability-screen-gate/v1",
        "attempts": len(rows), "known_terminal": known_terminal, "passed_images": sum(accepted.values()),
        "cells": cell_checks, "hard_errors": hard, "rejection_reasons": dict(sorted(rejected.items())),
        "checks": checks, "passed": all(checks.values()),
        "passed_rows": [{"sample_id": row["sample_id"]} for row in rows if _accepted_under_policy(row, gate, field="passed")],
    }
