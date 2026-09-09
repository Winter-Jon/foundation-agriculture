from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agrinet.rag.classifier_distill import (
    REGISTRY_SHA, budget_summary, candidate_card, check_training_isolation,
    file_sha, inspect_sources, preflight, teacher_views, validate_contract,
    validate_source_row,
)


@pytest.fixture
def contract():
    return yaml.safe_load(Path("configs/sampling/hcv-classifier-distill-contract-v1.yaml").read_text())


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"independent image fixture")
    training = tmp_path / "training.jsonl"
    training.write_text(json.dumps({
        "image_group_id": "train-image", "source_group_id": "train-source",
        "near_duplicate_group_id": "train-near", "image_sha256": "train-sha",
        "canonical_class_code": "N04001",
    }) + "\n")
    return {
        "sample_id": "sample-1", "image_group_id": "image-1",
        "source_group_id": "source-1", "near_duplicate_group_id": "near-1",
        "image_sha256": file_sha(image), "image_path": str(image),
        "question": "Identify the visible disease.", "question_type": "open",
        "language": "en", "task_domain": "disease",
        "dataset_version": "open_agri_v3", "split": "train_candidate",
        "private": {"truth_code": "N04001", "class_role": "known",
                    "target_pattern": "P1", "sampling_bucket": "ordinary",
                    "status": "fresh", "intervention": False,
                    "simulated_unknown": False},
        "prediction": {
            "kind": "fresh", "classifier_version": "fixture-v1",
            "checkpoint_sha256": "a" * 64, "registry_sha256": REGISTRY_SHA,
            "image_sha256": file_sha(image), "training_manifest": str(training),
            "training_manifest_sha256": file_sha(training),
            "top5": [{"code": f"N0400{i}", "score": score}
                     for i, score in enumerate((.3, .2, .15, .1, .05), 1)],
        },
    }


@pytest.fixture
def registry():
    return SimpleNamespace(
        digest=REGISTRY_SHA, rows_by_code={f"N0400{i}": {"domain": "disease"} for i in range(1, 7)},
        display_name=lambda code, language: f"public-{language}-{code}",
    )


@pytest.fixture
def roles():
    return {f"N0400{i}": "known" for i in range(1, 7)}


def test_contract_and_paired_budget(contract):
    validate_contract(contract)
    assert budget_summary(contract, 32) == {
        "independent_images": 32, "trajectories": 64, "rag_calls_max": 320,
        "generation_requests_max": 448, "private_audits_max": 64,
        "total_micu_requests_max": 512,
    }
    assert budget_summary(contract, 160)["total_micu_requests_max"] == 2560


@pytest.mark.parametrize("section,key,value", [
    ("teacher", "model", "another-model"),
    ("gates", "pilot_training_eligible", True),
    ("budgets", "micu_requests_per_trajectory", 8),
    ("dataset", "formal_unknown_sft", True),
])
def test_contract_rejects_boundary_changes(contract, section, key, value):
    contract[section][key] = value
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_views_are_allowlisted_and_independent(source, registry):
    source["unrecognized_secret"] = "must never be copied"
    source["retrieval_history"] = ["other-view-result"]
    views = teacher_views(source, registry)
    blind, assisted = views.values()
    assert set(blind) == {"image_path", "messages"}
    assert set(assisted) == {"image_path", "messages", "candidate_card"}
    assert len(assisted["candidate_card"]) == 3
    assert "must never be copied" not in json.dumps(views)
    assert "other-view-result" not in json.dumps(views)
    assisted["messages"].append({"role": "assistant", "content": "independent"})
    assert len(blind["messages"]) == 1
    with pytest.raises(ValueError, match="expansion"):
        candidate_card(source, registry, expanded=True)
    event = {"type": "expand_candidates", "view": "with_candidates", "reason": "need alternatives"}
    assert len(candidate_card(source, registry, expanded=True, expansion_event=event)) == 5
    assert all("score" not in c for c in candidate_card(source, registry, scores=False))


def test_valid_source_and_training_isolation(source, registry, roles, contract):
    validate_source_row(source, registry, roles, contract)
    check_training_isolation(source, {})
    source["source_group_id"] = "train-source"
    with pytest.raises(ValueError, match="overlap"):
        check_training_isolation(source, {})


def test_oof_training_manifest_may_derive_image_group_from_sha(source, registry, roles, contract, tmp_path):
    training = tmp_path / "oof_training.jsonl"
    training.write_text(json.dumps({
        "image_sha256": "other-sha", "source_group_id": "other-source",
        "near_duplicate_group_id": "other-near", "canonical_class_code": "N04001",
    }) + "\n")
    source["prediction"].update(kind="out_of_fold", folds=3, held_out_fold=0,
                                  training_manifest=str(training),
                                  training_manifest_sha256=file_sha(training))
    validate_source_row(source, registry, roles, contract)
    check_training_isolation(source, {})


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(split="test"),
    lambda r: r["prediction"].update(kind="in_sample"),
    lambda r: r["prediction"].update(image_sha256="wrong"),
    lambda r: r["private"].update(class_role="unknown"),
    lambda r: r["private"].update(status="unknown_delivery"),
    lambda r: r["private"].update(intervention=True),
    lambda r: r["prediction"]["top5"][0].update(score=float("nan")),
    lambda r: r["prediction"].update(kind="out_of_fold", folds=3, held_out_fold=3),
    lambda r: r["private"].update(target_pattern="P6"),
])
def test_source_rejects_unsafe_inputs(source, registry, roles, contract, mutation):
    mutation(source)
    with pytest.raises(ValueError):
        validate_source_row(source, registry, roles, contract)


def test_simulated_unknown_requires_supervised_exclusion(source, registry, roles, contract):
    source["private"].update(truth_code="N04006", simulated_unknown=True,
                             target_pattern="P6", mae_saw_related_unlabeled="unknown")
    with pytest.raises(ValueError, match="held out"):
        validate_source_row(source, registry, roles, contract)
    source["prediction"]["excluded_supervised_codes"] = ["N04006"]
    validate_source_row(source, registry, roles, contract)
    check_training_isolation(source, {})


def test_balanced_smoke_and_no_duplicate_padding(source, registry, roles, contract, tmp_path):
    rows = []
    for question_type in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                for i in range(4):
                    row = deepcopy(source)
                    suffix = f"{question_type}-{language}-{domain}-{i}"
                    for key in ("sample_id", "image_group_id", "source_group_id", "near_duplicate_group_id"):
                        row[key] = suffix
                    image = tmp_path / suffix
                    image.write_bytes(suffix.encode())
                    row.update(image_path=str(image), image_sha256=file_sha(image),
                               question_type=question_type, language=language, task_domain=domain)
                    row["prediction"]["image_sha256"] = row["image_sha256"]
                    if domain == "pest":
                        row["private"]["truth_code"] = "N04006"
                    rows.append(row)
    registry.rows_by_code["N04006"]["domain"] = "pest"
    ledger = {"schema_version": "agrinet.hcv-classifier-exclusions/v1",
              "complete": True, "provenance": ["fixture-only"], "records": []}
    kwargs = dict(registry=registry, roles=roles, contract=contract,
                  exclusions=ledger, evaluation_hashes=set(), stage="smoke")
    report = inspect_sources(rows, **kwargs)
    assert report["source_ready"]
    assert report["valid_independent_images"] == 32
    assert set(report["per_cell"].values()) == {4}
    assert not inspect_sources(rows[:-1] + [rows[0]], **kwargs)["source_ready"]
    kwargs["evaluation_hashes"] = {rows[0]["image_sha256"]}
    assert not inspect_sources(rows, **kwargs)["source_ready"]
    kwargs["evaluation_hashes"] = set()
    ledger["records"] = [{**rows[0], "status": "contacted"}]
    assert not inspect_sources(rows, **kwargs)["source_ready"]


def test_missing_inputs_produce_report_without_training_eligibility(tmp_path):
    result = preflight(
        contract_path=Path("configs/sampling/hcv-classifier-distill-contract-v1.yaml"),
        dataset_root=tmp_path, source=tmp_path / "source.jsonl",
        exclusions=tmp_path / "exclusions.json", stage="smoke")
    assert not result["ready"] and not result["training_eligible"]
    assert not result["live_collection_implemented"]
    assert len(result["blockers"]) == 3
    assert result["planned_budget"]["trajectories"] == 64
