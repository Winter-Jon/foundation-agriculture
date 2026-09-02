"""Resumable, fail-closed remote execution for M1 Direct collection."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from agrinet.data.io import DataError
from agrinet.research.m1.collection import auditor_request, canonical_hash, teacher_request

Request = Callable[[dict[str, Any]], dict[str, Any]]


class KnownInvalidResponse(RuntimeError):
    """The provider replied, but its structured content failed validation."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def register_contacted_images(
    public_rows: list[dict[str, Any]], ledger_path: Path, *, contact_stage: str, contact_round: str,
    permitted_prior_stage: str | None = None, permitted_prior_round: str | None = None,
    permitted_prior_rounds: set[str] | None = None,
    permitted_prior_refs: set[tuple[str, str]] | None = None,
) -> dict[str, int | str]:
    """Atomically retire an entire remote-request plan before its first POST.

    Re-running the same named round is allowed only when every planned hash was
    already registered by that round. A later stage can reuse hashes only by an
    explicit, whole-plan promotion from one named prior stage and round. This
    keeps screened-out images retired while allowing a passed screen to feed
    its declared successor without silently reopening the image pool.
    """
    hashes = [str(row.get("image_sha256") or "") for row in public_rows]
    if not hashes or any(not value for value in hashes):
        raise DataError("remote plan must contain non-empty image hashes")
    existing: list[dict[str, Any]] = []
    if ledger_path.is_file():
        for line_number, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid contacted ledger at {ledger_path}:{line_number}") from exc
            if not isinstance(row, dict) or not row.get("image_sha256"):
                raise DataError(f"invalid contacted ledger record at {ledger_path}:{line_number}")
            existing.append(row)
    by_hash = {str(row["image_sha256"]): row for row in existing}
    if len(by_hash) != len(existing):
        raise DataError(f"contacted ledger has duplicate hashes: {ledger_path}")
    plan = set(hashes); overlap = plan & set(by_hash)
    if overlap:
        allowed = {
            digest for digest in overlap
            if by_hash[digest].get("contact_stage") == contact_stage
            and by_hash[digest].get("round") == contact_round
        }
        permitted_rounds = (
            {permitted_prior_round} if permitted_prior_round is not None else set()
        ) | set(permitted_prior_rounds or set())
        permitted_refs = set(permitted_prior_refs or set())
        promoted = {
            digest for digest in overlap
            if (
                (permitted_prior_stage is not None and permitted_rounds
                 and by_hash[digest].get("contact_stage") == permitted_prior_stage
                 and by_hash[digest].get("round") in permitted_rounds)
                or (str(by_hash[digest].get("contact_stage") or ""),
                    str(by_hash[digest].get("round") or "")) in permitted_refs
            )
        }
        if allowed == plan:
            return {"registered": 0, "already_registered": len(plan), "contact_stage": contact_stage, "round": contact_round}
        if promoted != overlap:
            raise DataError("remote plan overlaps previously contacted images")
        # A replenishment plan may contain both newly isolated hashes and a
        # narrowly authorized subset of retired hashes. Promote only that
        # subset, then append genuinely new members atomically.
        promoted_rows = [
            {
                "image_sha256": str(row["image_sha256"]), "contact_stage": contact_stage, "round": contact_round,
                "promoted_from": {
                    "contact_stage": permitted_prior_stage,
                    "round": by_hash[str(row["image_sha256"])].get("round"),
                },
            } if str(row["image_sha256"]) in plan else row
            for row in existing
        ]
        promoted_rows.extend(
            {"image_sha256": digest, "contact_stage": contact_stage, "round": contact_round}
            for digest in sorted(plan - overlap)
        )
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in promoted_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, ledger_path)
        return {"registered": len(plan - overlap), "promoted": len(promoted), "contact_stage": contact_stage, "round": contact_round}
    additions = [
        {"image_sha256": digest, "contact_stage": contact_stage, "round": contact_round}
        for digest in sorted(plan)
    ]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in [*existing, *additions]:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, ledger_path)
    return {"registered": len(additions), "already_registered": 0, "contact_stage": contact_stage, "round": contact_round}


def _confirmed(path: Path, request_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("request_hash") != request_hash:
        raise DataError(f"request hash changed for existing checkpoint: {path}")
    if row.get("delivery_status") == "unknown":
        raise DataError(f"unknown delivery requires explicit provider resolution: {path}")
    if row.get("delivery_status") == "in_progress":
        # A prior process disappeared before it durably recorded a response.
        # The POST may have reached the provider, so this is not safely
        # replayable.  Persist the conservative status before refusing it.
        atomic_json(path, {
            "schema_version": "agrinet.m1-direct-request-checkpoint/v1",
            "sample_id": row.get("sample_id"), "request_hash": request_hash,
            "delivery_status": "unknown", "error_type": "InterruptedInProgressRequest",
            "decision": "do_not_replay_without_provider_resolution",
        })
        raise DataError(f"interrupted in-progress request requires provider resolution: {path}")
    return row if row.get("delivery_status") == "complete" else None


def execute_request(payload: dict[str, Any], path: Path, request_fn: Request) -> dict[str, Any]:
    """Execute one POST at most once unless its confirmed response exists."""
    request_hash = canonical_hash(payload)
    existing = _confirmed(path, request_hash)
    if existing is not None:
        return dict(existing["result"])
    atomic_json(path, {
        "schema_version": "agrinet.m1-direct-request-checkpoint/v1",
        "sample_id": payload["sample_id"], "request_hash": request_hash,
        "delivery_status": "in_progress",
    })
    try:
        result = request_fn(payload)
        if not isinstance(result, dict):
            raise KnownInvalidResponse("remote structured result must be an object")
    except KnownInvalidResponse as exc:
        result = {
            "response_status": "invalid",
            "error_type": type(exc).__name__,
            "error_detail": str(exc)[:512],
        }
    except BaseException as exc:
        atomic_json(path, {
            "schema_version": "agrinet.m1-direct-request-checkpoint/v1",
            "sample_id": payload["sample_id"], "request_hash": request_hash,
            "delivery_status": "unknown", "error_type": type(exc).__name__,
            "decision": "do_not_replay_without_provider_resolution",
        })
        raise
    atomic_json(path, {
        "schema_version": "agrinet.m1-direct-request-checkpoint/v1",
        "sample_id": payload["sample_id"], "request_hash": request_hash,
        "delivery_status": "complete", "result": result,
    })
    return result


def run_collection(
    public_rows: list[dict[str, Any]], private_rows: list[dict[str, Any]],
    output_dir: Path, teacher_fn: Request, auditor_fn: Request, *, workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run independent sample pipelines with durable no-replay checkpoints.

    A worker processes one sample's teacher request before its label-blind
    auditor request.  Different samples may run concurrently because their
    checkpoints have distinct paths.  This deliberately never parallelizes a
    teacher and its own auditor, and it never schedules a second attempt for a
    checkpoint that is already ``unknown``.
    """
    private = {str(row["sample_id"]): row for row in private_rows}
    if len(private) != len(private_rows) or {row["sample_id"] for row in public_rows} != set(private):
        raise DataError("public/private sample alignment mismatch")
    if workers < 1:
        raise DataError("M1 Direct collection workers must be at least one")

    def collect_one(task: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        sample_id = str(task["sample_id"]); truth = private[sample_id]
        # A deliberate operator stop can retire one ambiguous in-flight request
        # as unexecuted for this round.  It is never replayed from this plan;
        # the enclosing image group is replenished with a fresh image later.
        # This is distinct from an accidental transport failure (`unknown`).
        checkpoints = (
            output_dir / "teacher" / f"{sample_id}.json",
            output_dir / "auditor" / f"{sample_id}.json",
        )
        if any(
            path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("delivery_status") == "abandoned"
            for path in checkpoints
        ):
            return None, None, sample_id
        try:
            teacher_payload = teacher_request(task, truth)
            teacher = execute_request(teacher_payload, output_dir / "teacher" / f"{sample_id}.json", teacher_fn)
            teacher_row = {"sample_id": sample_id, **teacher}
            audit_payload = auditor_request(task, teacher)
            # The runner supplies image bytes/path but no label, code, or correct letter.
            audit_payload["image"] = truth["query_image"]
            auditor = execute_request(audit_payload, output_dir / "auditor" / f"{sample_id}.json", auditor_fn)
            return teacher_row, {"sample_id": sample_id, **auditor}, None
        except Exception:
            # A checkpoint that already says unknown is not replayable. Continue
            # the remaining, independently checkpointed samples in this same
            # pre-registered round so the private ledger records all available
            # evidence. The final gate remains fail-closed on this unknown.
            teacher_checkpoint = output_dir / "teacher" / f"{sample_id}.json"
            auditor_checkpoint = output_dir / "auditor" / f"{sample_id}.json"
            unknown_checkpoint = any(
                path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("delivery_status") == "unknown"
                for path in (teacher_checkpoint, auditor_checkpoint)
            )
            if not unknown_checkpoint:
                raise
            teacher_checkpoint = output_dir / "teacher" / f"{sample_id}.json"
            auditor_checkpoint = output_dir / "auditor" / f"{sample_id}.json"
            teacher_row = None
            auditor_row = None
            if teacher_checkpoint.is_file():
                saved = json.loads(teacher_checkpoint.read_text(encoding="utf-8"))
                if saved.get("delivery_status") == "complete" and isinstance(saved.get("result"), dict):
                    teacher_row = {"sample_id": sample_id, **saved["result"]}
            # The in-memory teacher response is valid evidence even if its
            # subsequent auditor POST becomes an unknown delivery. Keep it in
            # the aggregate just as the serial runner did.
            if teacher_row is None and "teacher" in locals():
                teacher_row = {"sample_id": sample_id, **teacher}
            if auditor_checkpoint.is_file():
                saved = json.loads(auditor_checkpoint.read_text(encoding="utf-8"))
                if saved.get("delivery_status") == "complete" and isinstance(saved.get("result"), dict):
                    auditor_row = {"sample_id": sample_id, **saved["result"]}
            return teacher_row, auditor_row, sample_id

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="m1-direct") as executor:
        outcomes = list(executor.map(collect_one, public_rows))
    teachers = [teacher for teacher, _auditor, _unknown in outcomes if teacher is not None]
    auditors = [auditor for _teacher, auditor, _unknown in outcomes if auditor is not None]
    stopped_samples = [sample_id for _teacher, _auditor, sample_id in outcomes if sample_id is not None]
    unknown_samples: list[str] = []
    abandoned_samples: list[str] = []
    for sample_id in stopped_samples:
        statuses = []
        for stage in ("teacher", "auditor"):
            path = output_dir / stage / f"{sample_id}.json"
            if path.is_file():
                statuses.append(json.loads(path.read_text(encoding="utf-8")).get("delivery_status"))
        if "abandoned" in statuses:
            abandoned_samples.append(sample_id)
        else:
            unknown_samples.append(sample_id)
    report = {
        "schema_version": "agrinet.m1-direct-collection-run/v1",
        "attempts": len(public_rows), "teacher_complete": len(teachers),
        "auditor_complete": len(auditors),
        "delivery_status": "unknown" if unknown_samples else "complete",
        "unknown_samples": unknown_samples,
        "abandoned_samples": abandoned_samples,
    }
    atomic_json(output_dir / "run_report.json", report)
    return teachers, auditors, report
