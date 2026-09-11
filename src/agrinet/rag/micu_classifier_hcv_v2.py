"""Deterministic readiness gate for the Micu/classifier/HCV v2 smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


EXPECTED_CELLS = {
    "open-en-disease", "open-en-pest", "open-zh-disease", "open-zh-pest",
    "option-en-disease", "option-en-pest", "option-zh-disease", "option-zh-pest",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_collection_source(source: Path, dataset_root: Path,
                               merged_predictions: Path | None = None, *,
                               stage: str = "smoke") -> dict[str, Any]:
    """Check a Scheme-B source before any row may reach a public request."""
    rows = _read_jsonl(source)
    expected_rows = 32 if stage in {"smoke", "replacement_smoke"} else 160
    expected_per_cell = 4 if expected_rows == 32 else 20
    if stage not in {"smoke", "replacement_smoke", "exploration"}:
        raise ValueError("unsupported collection stage")
    if len(rows) != expected_rows:
        raise ValueError(f"{stage} source must contain exactly {expected_rows} rows")
    evaluation = {row["image_sha256"] for row in _read_jsonl(dataset_root / "manifests" / "images.jsonl")
                  if row.get("image_split") in {"dev", "test"}}
    merged_by_sha: dict[str, dict[str, Any]] | None = None
    if merged_predictions is not None:
        merged_by_sha = {row["image_sha256"]: row for row in _read_jsonl(merged_predictions)}
        if len(merged_by_sha) != len(_read_jsonl(merged_predictions)):
            raise ValueError("merged OOF predictions have duplicate image identities")
    cells: Counter[str] = Counter(); identities = {"image_sha256": set(), "image_group_id": set(),
                                                   "source_group_id": set(), "near_duplicate_group_id": set()}
    patterns: Counter[str] = Counter()
    for row in rows:
        for key, seen in identities.items():
            value = row.get(key)
            if not isinstance(value, str) or not value or value in seen:
                raise ValueError(f"smoke source has duplicate or missing {key}")
            seen.add(value)
        if sha256(Path(row.get("image_path", ""))) != row["image_sha256"]:
            raise ValueError("smoke source image SHA mismatch")
        if row["image_sha256"] in evaluation:
            raise ValueError("smoke source overlaps frozen evaluation image")
        cell = "-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))
        if cell not in EXPECTED_CELLS:
            raise ValueError("smoke source contains invalid cell")
        cells[cell] += 1
        prediction = row.get("prediction")
        if not isinstance(prediction, dict) or prediction.get("kind") not in {"out_of_fold", "p6_class_holdout"}:
            raise ValueError(f"{stage} source contains an invalid classifier output kind")
        if prediction.get("kind") == "out_of_fold" and (prediction.get("folds") != 3 or prediction.get("held_out_fold") not in (0, 1, 2)):
            raise ValueError(f"{stage} source has invalid OOF fold provenance")
        if prediction.get("kind") == "p6_class_holdout":
            excluded = prediction.get("excluded_supervised_codes")
            label_codes = prediction.get("label_codes")
            truth = (row.get("private") or {}).get("truth_code")
            if stage != "exploration" or not isinstance(excluded, list) or len(excluded) != 8:
                raise ValueError("P6 prediction requires one frozen eight-class holdout group")
            if truth not in excluded or not isinstance(label_codes, list) or truth in label_codes:
                raise ValueError("P6 truth must be excluded from the classifier label space")
        if merged_by_sha is not None and prediction.get("kind") == "out_of_fold":
            merged = merged_by_sha.get(row["image_sha256"])
            if merged is None or merged.get("prediction") != prediction:
                raise ValueError("smoke source prediction differs from audited OOF merge")
        top5 = prediction.get("top5")
        if not isinstance(top5, list) or len(top5) != 5 or len({item.get("code") for item in top5 if isinstance(item, dict)}) != 5:
            raise ValueError("smoke source has invalid classifier Top-5")
        private = row.get("private")
        if not isinstance(private, dict) or private.get("class_role") != "known":
            raise ValueError(f"{stage} source must remain Known-only")
        if prediction.get("kind") == "p6_class_holdout" and private.get("simulated_unknown") is not True:
            raise ValueError("P6 rows must be marked simulated_unknown")
        if prediction.get("kind") == "out_of_fold" and private.get("simulated_unknown") is not False:
            raise ValueError("ordinary rows may not be marked simulated_unknown")
        pattern = str(private.get("primary_pattern") or private.get("target_pattern") or "")
        if pattern not in {f"P{i}" for i in range(1, 11)}:
            raise ValueError("each query requires exactly one valid primary pattern")
        patterns[pattern] += 1
    if set(cells) != EXPECTED_CELLS or any(count != expected_per_cell for count in cells.values()):
        raise ValueError(f"{stage} source does not have {expected_per_cell} rows per fixed cell")
    arms = Counter(str((row.get("private") or {}).get("sampling_arm") or "") for row in rows)
    if stage == "exploration" and arms != {"targeted": 80, "random": 80}:
        raise ValueError("exploration source must contain disjoint 80 targeted + 80 random arms")
    return {"rows": len(rows), "stage": stage, "cells": dict(sorted(cells.items())),
            "arms": dict(sorted(arms.items())), "patterns": dict(sorted(patterns.items()))}


def validate_smoke_source(source: Path, dataset_root: Path,
                          merged_predictions: Path | None = None) -> dict[str, Any]:
    """Backward-compatible strict 32-image smoke validator."""
    return validate_collection_source(source, dataset_root, merged_predictions, stage="smoke")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_rag_health(endpoint: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {"status": "invalid_response"}
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"status": "unreachable", "error": str(exc)[:240]}


def classifier_fresh_capacity(dataset_root: Path, training_manifest: Path) -> dict[str, Any]:
    """Summarize whether frozen v3 inventory contains classifier-independent Known images."""
    train_hashes = set()
    with training_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                train_hashes.add(str(json.loads(line)["image_sha256"]))
    roles: dict[str, str] = {}
    with (dataset_root / "manifests/class_split.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                roles[str(row["canonical_class_code"])] = str(row["class_role"])
    splits: Counter[str] = Counter()
    known_not_trained = 0
    with (dataset_root / "manifests/images.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            role = roles.get(str(row["canonical_class_code"]))
            split = str(row["image_split"])
            splits[f"{role}:{split}"] += 1
            if role == "known" and row["image_sha256"] not in train_hashes and split not in {"dev", "test", "milvus_reference"}:
                known_not_trained += 1
    return {
        "classifier_training_images": len(train_hashes),
        "images_by_role_split": dict(sorted(splits.items())),
        "known_images_outside_classifier_train_and_eval_splits": known_not_trained,
    }


def _read_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "agrinet.micu-classifier-hcv-v2/v1":
        raise ValueError("invalid v2 contract schema")
    return raw


def validate_contract(contract: dict[str, Any]) -> None:
    if contract["teacher"] != {"service": "micu_slb", "model": "gpt-5.6-terra", "prompt_version": "classifier-hcv-v2-unified-router-v1"}:
        raise ValueError("teacher contract drift")
    if contract["student"].get("model") != "models/Qwen3-VL-4B-Instruct":
        raise ValueError("student contract drift")
    stage = str(contract.get("data", {}).get("stage") or "smoke")
    exploration = stage == "exploration"
    classifier = contract.get("classifier", {})
    allowed_kinds = classifier.get("prediction_kinds", [classifier.get("prediction_kind")])
    if "out_of_fold" not in allowed_kinds or not classifier.get("oof_audit"):
        raise ValueError("Scheme B requires an explicit grouped OOF classifier audit")
    if exploration and set(allowed_kinds) != {"out_of_fold", "p6_class_holdout"}:
        raise ValueError("exploration requires OOF and P6 class-holdout predictions")
    expected_rows, expected_per_cell = (160, 20) if exploration else (32, 4)
    if set(contract["data"].get("cells", [])) != EXPECTED_CELLS or contract["data"].get("images_per_cell") != expected_per_cell:
        raise ValueError(f"{stage} cells must be the eight fixed {expected_per_cell}-image cells")
    if contract["data"].get("independent_images") != expected_rows:
        raise ValueError(f"{stage} must contain {expected_rows} independent images")
    if contract["retrieval"].get("exposed_modes") != ["visual", "semantic"] or contract["retrieval"].get("top_k") != 3:
        raise ValueError("retrieval surface drift")
    if contract["retrieval"].get("budget_schedule") != [0, 1, 3, 5]:
        raise ValueError("retrieval budget schedule drift")
    if contract["routing"].get("one_primary_pattern_per_query") is not True:
        raise ValueError("one pattern per query is required")
    expected_budgets = (1650, 900) if exploration else (340, 200)
    if (contract["budgets"].get("total_micu"), contract["budgets"].get("rag_total")) != expected_budgets:
        raise ValueError("global budget drift")
    if contract["runtime"].get("training_eligible") is not False:
        raise ValueError("smoke may not be training eligible")


def readiness(*, contract_path: Path, dataset_root: Path, rag_health: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "agrinet.micu-classifier-hcv-v2-readiness/v1",
        "ready_for_live_collection": False,
        "training_eligible": False,
        "blockers": [],
        "inputs": {},
    }
    try:
        contract = _read_contract(contract_path)
        validate_contract(contract)
        report["inputs"]["contract"] = {"path": str(contract_path), "sha256": sha256(contract_path)}
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        report["blockers"].append(f"contract: {exc}")
        return report
    root = contract_path.parents[3]
    dataset_root = dataset_root if dataset_root.is_absolute() else root / dataset_root
    report["inputs"]["dataset_root"] = {"path": str(dataset_root), "exists": dataset_root.is_dir()}
    required_dataset = (
        "taxonomy/canonical_label_registry.jsonl",
        "taxonomy/approval.json",
        "manifests/class_split.jsonl",
        "manifests/images.jsonl",
    )
    for relative in required_dataset:
        path = dataset_root / relative
        if not path.is_file():
            report["blockers"].append(f"missing dataset metadata: {dataset_root / relative}")
        else:
            report["inputs"][f"dataset:{relative}"] = {"path": str(path), "sha256": sha256(path)}
    required_inputs = {
        "student": contract["student"]["model"],
        "classifier_checkpoint": contract["classifier"]["checkpoint"],
        "classifier_training_manifest": contract["classifier"]["training_manifest"],
        "classifier_oof_audit": contract["classifier"]["oof_audit"],
        "classifier_merged_predictions": contract["classifier"]["merged_predictions"],
        "source": contract["data"]["source"],
        "exclusions": contract["data"]["exclusions"],
    }
    for index, relative in enumerate(contract.get("classifier", {}).get("p6_artifact_roots", [])):
        required_inputs[f"p6_artifact_{index}"] = relative
    for name, relative in required_inputs.items():
        path = root / relative
        if not path.exists():
            report["blockers"].append(f"missing {name}: {relative}")
        else:
            report["inputs"][name] = {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
    audit_path = root / contract["classifier"]["oof_audit"]
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            report["inputs"]["classifier_oof_audit"]["ready"] = audit.get("ready")
            if audit.get("schema_version") != "agrinet.open-agri-v3.grouped-oof-audit/v1" or audit.get("ready") is not True:
                report["blockers"].append("grouped OOF audit is not ready")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            report["blockers"].append(f"grouped OOF audit unreadable: {exc}")
    source_path = root / contract["data"]["source"]
    merged_path = root / contract["classifier"]["merged_predictions"]
    if source_path.is_file() and dataset_root.is_dir() and merged_path.is_file():
        try:
            report["source_validation"] = validate_collection_source(
                source_path, dataset_root, merged_path, stage=str(contract["data"].get("stage") or "smoke"))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            report["blockers"].append(f"smoke source validation: {exc}")
    training_manifest = root / contract["classifier"]["training_manifest"]
    if (dataset_root / "manifests/images.jsonl").is_file() and (dataset_root / "manifests/class_split.jsonl").is_file() and training_manifest.is_file() and contract["classifier"].get("prediction_kind") != "out_of_fold":
        try:
            report["dataset_capacity"] = classifier_fresh_capacity(dataset_root, training_manifest)
            if report["dataset_capacity"]["known_images_outside_classifier_train_and_eval_splits"] < 32:
                report["blockers"].append(
                    "insufficient classifier-independent Known fresh capacity: "
                    + str(report["dataset_capacity"]["known_images_outside_classifier_train_and_eval_splits"])
                    + " available, 32 required"
                )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            report["blockers"].append(f"dataset capacity audit: {exc}")
    if rag_health is not None:
        report["inputs"]["rag_health"] = rag_health
        if rag_health.get("status") != "ok":
            report["blockers"].append("RAG health status is not ok")
        expected = {"open_agri_v3_classes", "open_agri_v3_images"}
        actual = set((rag_health.get("collections") or {}).keys())
        if actual != expected:
            report["blockers"].append(f"RAG collections mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    report["ready_for_live_collection"] = not report["blockers"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    contract = _read_contract(args.contract)
    report = readiness(
        contract_path=args.contract,
        dataset_root=args.dataset_root,
        rag_health=local_rag_health(str(contract["retrieval"]["endpoint"])),
    )
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = args.output_root / attempt / "readiness.json"
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "ready": report["ready_for_live_collection"], "blockers": report["blockers"]}, ensure_ascii=False))
    return 0 if report["ready_for_live_collection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
