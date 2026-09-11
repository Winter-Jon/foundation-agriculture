import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eligibility_loop", ROOT / "scripts/rag/manage_e2_e3_training_eligibility_loop.py")
assert SPEC and SPEC.loader
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


def test_contract_locks_and_append_only_patch_chain(tmp_path: Path) -> None:
    e2, e3 = tmp_path / "e2.yaml", tmp_path / "e3.yaml"
    e2.write_text("e2: immutable\n", encoding="utf-8")
    e3.write_text("e3: immutable\n", encoding="utf-8")
    loop.initialize(root=tmp_path, e2_contract=e2, e3_contract=e3)
    locks = json.loads((tmp_path / "contract-locks.json").read_text())
    patch = {
        "schema_version": "agrinet.e2-e3-training-eligibility-contract-patch/v1",
        "patch_id": "rewrite-v2", "base_contract_sha256": locks["contracts"][0]["sha256"],
        "parent_patch_sha256": None,
        "prohibited_changes": ["historical_contract", "strict_canary_history", "old_ledgers_or_request_ids", "sft_authorization"],
    }
    path = tmp_path / "patches/0001-rewrite-v2.json"; path.parent.mkdir(); path.write_text(json.dumps(patch), encoding="utf-8")
    assert loop.validate_patch(root=tmp_path, patch=path)["valid"] is True


def test_rewrite_analysis_reports_contract_not_delivery_failure(tmp_path: Path) -> None:
    item = tmp_path / "rounds/r0/public/group/rewrite.json"
    item.parent.mkdir(parents=True)
    item.write_text(json.dumps({"route": "direct", "errors": ["rewrite_observations_under_three", "direct_rewrite_mentions_tool"]}), encoding="utf-8")
    report = loop.rewrite_analysis(rewrite_root=tmp_path)
    assert report["conclusion"] == "prompt_validator_contract_mismatch"
    assert report["error_counts"]["direct_rewrite_mentions_tool"] == 1
    assert report["provider_retry_authorized"] is False


def test_e3_aggregate_refuses_nonterminal_or_incomplete_fold_evidence(tmp_path: Path) -> None:
    run = tmp_path / "runs"
    for fold in range(3):
        status = run / f"vision-openagri-v3-known-vitl-e3-fold{fold}-v1/20260910T235014-5fc04488-a01/status.json"
        status.parent.mkdir(parents=True); status.write_text(json.dumps({"status": "complete", "exit_code": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks (frozen adjacency|final)"):
        loop.e3_aggregate(run_root=run, artifact_root=tmp_path / "artifacts")


def test_e3_aggregate_rejects_held_out_label_leakage(tmp_path: Path) -> None:
    run, artifacts = tmp_path / "runs", tmp_path / "artifacts"
    assignments = []
    for fold in range(3):
        status = run / f"vision-openagri-v3-known-vitl-e3-fold{fold}-v1/20260910T235014-5fc04488-a01/status.json"
        status.parent.mkdir(parents=True); status.write_text(json.dumps({"status": "complete", "exit_code": 0}), encoding="utf-8")
        base = artifacts / f"fold-{fold}"; (base / "classifier").mkdir(parents=True); (base / "manifests").mkdir()
        (base / "classifier/classifier_dev_metrics.json").write_text(json.dumps([{"epoch": i + 1, "macro_f1": 0.5, "disease_macro_f1": 0.5, "pest_macro_f1": 0.5} for i in range(50)]), encoding="utf-8")
        (base / "classifier/model_best.pth.tar").write_bytes(b"checkpoint")
        (base / "classifier/metrics_test_known.json").write_text(json.dumps({"macro_f1": 0.5, "balanced_accuracy": 0.5}), encoding="utf-8")
        (base / "label_map.json").write_text(json.dumps([{"canonical_class_code": "held"}]), encoding="utf-8")
        (base / "manifests/train.jsonl").write_text(json.dumps({"canonical_class_code": "train", "image_sha256": "train-image"}) + "\n", encoding="utf-8")
        (base / "manifests/holdout_train_candidate.jsonl").write_text(json.dumps({"canonical_class_code": "held", "image_sha256": "held-image"}) + "\n", encoding="utf-8")
        assignments.extend({"canonical_class_code": f"code-{fold}-{index}", "training_neighbour_witnesses": ["neighbour"]} for index in range(36 if fold < 2 else 35))
    assignments[0]["canonical_class_code"] = "held"
    (artifacts / "class_assignments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in assignments), encoding="utf-8")
    (artifacts / "folds.json").write_text(json.dumps({"bridge_contract": {"all_held_out_classes_have_training_neighbour": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="isolation"):
        loop.e3_aggregate(run_root=run, artifact_root=artifacts)


def test_e3_aggregate_accepts_complete_isolated_terminal_evidence(tmp_path: Path) -> None:
    run, artifacts = tmp_path / "runs", tmp_path / "artifacts"
    assignments = []
    for fold, count in enumerate((36, 36, 35)):
        status = run / f"vision-openagri-v3-known-vitl-e3-fold{fold}-v1/20260910T235014-5fc04488-a01/status.json"
        status.parent.mkdir(parents=True); status.write_text(json.dumps({"status": "complete", "exit_code": 0}), encoding="utf-8")
        base = artifacts / f"fold-{fold}"; (base / "classifier").mkdir(parents=True); (base / "manifests").mkdir()
        held = f"held-{fold}"
        assignments.extend({"canonical_class_code": held if index == 0 else f"unused-{fold}-{index}", "training_neighbour_witnesses": ["train-neighbour"]} for index in range(count))
        (base / "classifier/classifier_dev_metrics.json").write_text(json.dumps([{"epoch": i + 1, "macro_f1": 0.7 + fold / 100, "disease_macro_f1": 0.8, "pest_macro_f1": 0.6} for i in range(50)]), encoding="utf-8")
        (base / "classifier/model_best.pth.tar").write_bytes(f"checkpoint-{fold}".encode())
        (base / "classifier/metrics_test_known.json").write_text(json.dumps({"macro_f1": 0.65 + fold / 100, "balanced_accuracy": 0.6}), encoding="utf-8")
        (base / "label_map.json").write_text(json.dumps([{"canonical_class_code": f"train-{fold}"}]), encoding="utf-8")
        (base / "manifests/train.jsonl").write_text(json.dumps({"canonical_class_code": f"train-{fold}", "image_sha256": f"train-image-{fold}"}) + "\n", encoding="utf-8")
        (base / "manifests/holdout_train_candidate.jsonl").write_text(json.dumps({"canonical_class_code": held, "image_sha256": f"held-image-{fold}"}) + "\n", encoding="utf-8")
    (artifacts / "class_assignments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in assignments), encoding="utf-8")
    (artifacts / "folds.json").write_text(json.dumps({"bridge_contract": {"all_held_out_classes_have_training_neighbour": True}}), encoding="utf-8")
    report = loop.e3_aggregate(run_root=run, artifact_root=artifacts)
    assert report["generalization_evidence_complete"] is True
    assert [row["test_known_macro_f1"] for row in report["folds"]] == [0.65, 0.66, 0.67]


def test_training_qualification_requires_balanced_32_converted_rows(tmp_path: Path) -> None:
    rows, accepted = [], []
    index = 0
    for question in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                for _ in range(4):
                    group = f"g{index}"; index += 1
                    rows.append({"image_group_id": group, "question_type": question, "language": language, "task_domain": domain})
                    accepted.append({"image_group_id": group})
    source, gate = tmp_path / "source.json", tmp_path / "gate.json"
    source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    gate.write_text(json.dumps({"accepted": accepted}), encoding="utf-8")
    report = loop.training_qualification(source=source, conversion_gate=gate)
    assert report["training_eligible"] is True and report["sft_may_start"] is False
    gate.write_text(json.dumps({"accepted": accepted[:-1]}), encoding="utf-8")
    assert loop.training_qualification(source=source, conversion_gate=gate)["training_eligible"] is False


def test_pilot_preflight_requires_e3_aggregate_and_one_row_per_cell(tmp_path: Path) -> None:
    aggregate = tmp_path / "e3.json"; aggregate.write_text(json.dumps({"generalization_evidence_complete": True}), encoding="utf-8")
    patch = tmp_path / "patch.json"; patch.write_text(json.dumps({"patch_id": "0001-rewrite-structure-v2"}), encoding="utf-8")
    rows = [{"image_group_id": f"g{i}", "question_type": question, "language": language, "task_domain": domain}
            for i, (question, language, domain) in enumerate((q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest"))]
    source = tmp_path / "source.json"; source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    report = loop.pilot_preflight(e3_aggregate_path=aggregate, patch=patch, source=source)
    assert report["new_lineage_required"] is True
    assert report["provider_request_authorized"] is False


def test_pilot_preflight_allows_only_explicit_e3_running_exception(tmp_path: Path) -> None:
    rewrite = tmp_path / "rewrite.json"
    rewrite.write_text(json.dumps({"patch_id": "0001-rewrite-structure-v2"}), encoding="utf-8")
    exception = tmp_path / "exception.json"
    exception.write_text(json.dumps({"patch_id": "0002-e3-running-pilot-risk-exception",
                                     "parent_patch_sha256": loop.sha256(rewrite)}), encoding="utf-8")
    rows = [{"image_group_id": f"g{i}", "question_type": question, "language": language, "task_domain": domain}
            for i, (question, language, domain) in enumerate((q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest"))]
    source = tmp_path / "source.json"; source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    report = loop.pilot_preflight(e3_aggregate_path=None, patch=rewrite, source=source, risk_exception_patch=exception)
    assert report["provider_request_authorized"] is True
    assert report["e3_generalization_evidence_complete"] is False


def test_freeze_pilot_source_excludes_every_historical_group_kind(tmp_path: Path) -> None:
    cells = [(q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")]
    prior = tmp_path / "prior.json"; prior.write_text(json.dumps({"rows": [{"image_group_id": "old-image", "source_group_id": "old-source", "near_duplicate_group_id": "old-near"}]}), encoding="utf-8")
    rows = [{"sample_id": f"s{i}", "image_group_id": f"image{i}", "source_group_id": f"source{i}", "near_duplicate_group_id": f"near{i}", "question_type": q, "language": l, "task_domain": d} for i, (q, l, d) in enumerate(cells)]
    pool = tmp_path / "pool.json"; pool.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    result = loop.freeze_pilot_source(pool=pool, prior_sources=[prior])
    assert len(result["rows"]) == 8


def test_freeze_campaign_source_requires_four_isolated_rows_per_cell(tmp_path: Path) -> None:
    cells = [(q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")]
    rows = [{"sample_id": f"s{cell_index}-{item}", "image_group_id": f"image{cell_index}-{item}",
             "source_group_id": f"source{cell_index}-{item}", "near_duplicate_group_id": f"near{cell_index}-{item}",
             "question_type": question, "language": language, "task_domain": domain}
            for cell_index, (question, language, domain) in enumerate(cells) for item in range(4)]
    pool, prior = tmp_path / "pool.json", tmp_path / "prior.json"
    pool.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    prior.write_text(json.dumps({"rows": [{"image_group_id": "old-image", "source_group_id": "old-source", "near_duplicate_group_id": "old-near"}]}), encoding="utf-8")
    result = loop.freeze_campaign_source(pool=pool, prior_sources=[prior])
    assert len(result["rows"]) == 32
    assert set(result["cell_counts"].values()) == {4}


def test_freeze_campaign_source_can_prefer_lower_prescreen_pattern(tmp_path: Path) -> None:
    cells = [(q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")]
    rows = [{"sample_id": f"s{cell_index}-{pattern}", "image_group_id": f"image{cell_index}-{pattern}",
             "source_group_id": f"source{cell_index}-{pattern}", "near_duplicate_group_id": f"near{cell_index}-{pattern}",
             "question_type": question, "language": language, "task_domain": domain,
             "private": {"candidate_pattern": f"P{pattern}"}}
            for cell_index, (question, language, domain) in enumerate(cells) for pattern in range(1, 8)]
    pool, prior = tmp_path / "pool.json", tmp_path / "prior.json"
    pool.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    prior.write_text(json.dumps({"rows": []}), encoding="utf-8")
    result = loop.freeze_campaign_source(pool=pool, prior_sources=[prior], selection="lower_pattern_first")
    assert result["selection"] == "lower_pattern_first"
    assert {row["private"]["candidate_pattern"] for row in result["rows"]} == {"P1", "P2", "P3", "P4"}


def test_freeze_campaign_source_can_prefer_p6_with_lower_pattern_fallback(tmp_path: Path) -> None:
    cells = [(q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")]
    rows = [{"sample_id": f"s{cell_index}-{pattern}", "image_group_id": f"image{cell_index}-{pattern}",
             "source_group_id": f"source{cell_index}-{pattern}", "near_duplicate_group_id": f"near{cell_index}-{pattern}",
             "question_type": question, "language": language, "task_domain": domain,
             "private": {"candidate_pattern": pattern}}
            for cell_index, (question, language, domain) in enumerate(cells)
            for pattern in ("P1", "P2", "P6", "P6a", "P6b")]
    # Distinct strings P6a/P6b model independent P6 rows only for selection-order testing.
    for row in rows:
        if row["private"]["candidate_pattern"] in {"P6a", "P6b"}:
            row["private"]["candidate_pattern"] = "P6"
    pool, prior = tmp_path / "pool.json", tmp_path / "prior.json"
    pool.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    prior.write_text(json.dumps({"rows": []}), encoding="utf-8")
    result = loop.freeze_campaign_source(pool=pool, prior_sources=[prior], selection="p6_first")
    assert result["selection"] == "p6_first"
    for question, language, domain in cells:
        chosen = [row["private"]["candidate_pattern"] for row in result["rows"]
                  if (row["question_type"], row["language"], row["task_domain"]) == (question, language, domain)]
        assert chosen.count("P6") == 3
        assert chosen.count("P1") == 1


def test_freeze_campaign_source_can_prefer_higher_oof_confidence(tmp_path: Path) -> None:
    cells = [(q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")]
    rows = [{"sample_id": f"s{cell_index}-{item}", "image_group_id": f"image{cell_index}-{item}",
             "source_group_id": f"source{cell_index}-{item}", "near_duplicate_group_id": f"near{cell_index}-{item}",
             "question_type": question, "language": language, "task_domain": domain,
             "private": {"candidate_pattern": "P9"},
             "prediction": {"top5": [{"score": item / 10}]}}
            for cell_index, (question, language, domain) in enumerate(cells) for item in range(6)]
    pool, prior = tmp_path / "pool.json", tmp_path / "prior.json"
    pool.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    prior.write_text(json.dumps({"rows": []}), encoding="utf-8")
    result = loop.freeze_campaign_source(pool=pool, prior_sources=[prior], selection="higher_confidence_first")
    assert result["selection"] == "higher_confidence_first"
    assert {row["prediction"]["top5"][0]["score"] for row in result["rows"]} == {0.2, 0.3, 0.4, 0.5}
