import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from agrinet.data.io import DataError
from agrinet.research.open_agri_v2 import supplement


def test_initial_plan_is_staging_only_and_has_expected_v2_coverage(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    report = supplement.build_initial_plan(root, dataset, tmp_path, seed="test-seed")
    assert report["formal_dataset_modified"] is False
    assert report["known_classes"] == 109
    assert report["classes_short"] == 42
    assert report["accepted_image_shortfall"] == 134
    assert report["planned_images"] == 213
    assert report["planned_views"] == {"direct": 852, "rag": 852}
    assert len(report["direct_gate_config_sha256"]) == 64
    assert len(supplement.read_jsonl(tmp_path / "public/direct_plan.jsonl")) == 852
    rag = supplement.read_jsonl(tmp_path / "private/rag_plan.jsonl")
    assert len(rag) == 852
    assert all(row["source_sample_id"] == row["sample_id"] for row in rag)
    assert not (tmp_path / "public/rag_plan.jsonl").exists()


def test_image_decision_rejects_entire_group_when_one_view_fails() -> None:
    views = []
    for route in ("direct", "rag"):
        for index in range(4):
            views.append({
                "sample_id": f"x-{route}-{index}", "class_code": "N04001",
                "image_sha256": "image", "query_image": "image.jpg",
            })
    direct = [{"sample_id": f"x-direct-{index}", "accepted": index != 2, "errors": ["format_error"] if index == 2 else []} for index in range(4)]
    rag_accepted = [{"sample_id": f"x-rag-{index}"} for index in range(4)]
    decisions, accepted = supplement.image_group_decisions(direct, rag_accepted, [], views)
    assert not accepted
    assert decisions[0]["accepted"] is False
    assert "direct:format_error" in decisions[0]["rejection_reasons"]


def test_human_approval_requires_explicit_reviewer_fields(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    (review / "approval.json").write_text(json.dumps({"approved": True, "reviewer": "r", "reviewed_at": "now"}))
    assert supplement.require_human_approval(tmp_path)["approved"] is True


def test_rag_preflight_rejects_http_error_before_teacher_contact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "plan.jsonl"
    supplement.write_rows(plan, [{"sample_id": "s", "image_sha256": "h", "query_image": "image.jpg"}])

    def fail(*_args: object, **_kwargs: object) -> object:
        raise HTTPError("http://127.0.0.1:8077/search/visual", 502, "Bad Gateway", {}, None)

    monkeypatch.setattr(supplement.urllib.request, "urlopen", fail)
    report = tmp_path / "preflight.json"
    with pytest.raises(DataError, match="no Micu request was issued"):
        supplement.rag_service_preflight(plan, report, rag_api="http://127.0.0.1:8077")
    result = json.loads(report.read_text())
    assert result["status"] == "failed"
    assert result["teacher_contacted"] is False


def test_round_one_excludes_every_contacted_hash(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplement.build_initial_plan(root, dataset, tmp_path, seed="test-rounds")
    first = {row["image_sha256"] for row in supplement.read_jsonl(tmp_path / "public/direct_plan.jsonl")}
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": digest, "contact_stage": "test", "round": "v1"} for digest in sorted(first)],
    )
    report = supplement.build_replenishment_plan(root, dataset, tmp_path, round_index=1, seed="test-rounds")
    second = {row["image_sha256"] for row in supplement.read_jsonl(tmp_path / "rounds/v1/public/direct_plan.jsonl")}
    assert report["retired_images_excluded"] == len(first)
    assert not first & second


def test_merge_sources_reads_initial_rag_v2_and_round_rag(tmp_path: Path) -> None:
    for rel, sample_id, digest in [
        ("collection/direct/accepted.jsonl", "d0", "a"),
        ("collection/rag-v2/accepted.jsonl", "r0", "a"),
        ("rounds/v1/collection/direct/accepted.jsonl", "d1", "b"),
        ("rounds/v1/collection/rag/accepted.jsonl", "r1", "b"),
    ]:
        supplement.write_rows(tmp_path / rel, [{"sample_id": sample_id, "metadata": {"v2_image_sha256": digest}}])
    direct, rag = supplement.merge_sources(tmp_path, {"a", "b"})
    assert {row["sample_id"] for row in supplement.read_jsonl(direct)} == {"d0", "d1"}
    assert {row["sample_id"] for row in supplement.read_jsonl(rag)} == {"r0", "r1"}


def test_final_shortfall_report_stops_after_two_rounds(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.write_final_shortfall_report(root, dataset, tmp_path, completed_rounds=3)
    assert report["automatic_replenishment_rounds_completed"] == 3
    assert report["automatic_replenishment_stopped"] is True
    assert report["formal_dataset_modified"] is False
    assert (tmp_path / "reports/final_shortfall_after_round3.json").is_file()


def test_round_seven_terminal_status_is_included_in_aggregate_review(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.write_final_shortfall_report(root, dataset, tmp_path, completed_rounds=7)
    assert report["automatic_replenishment_rounds_completed"] == 7
    assert (tmp_path / "reports/final_shortfall_after_round7.json").is_file()


def test_capacity_limited_round_records_unavoidable_shortfall(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplemental = tmp_path / "round"
    report = supplement.build_initial_plan(
        root, dataset, supplemental, seed="capacity-limited", round_index=3,
        retired_images={row["image_sha256"] for row in supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl") if row.get("class_code") == "N04094"} - {""},
        allow_capacity_limited=True,
    )
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0
    assert report["capacity_limited"]["N04094"]["unavoidable_floor_shortfall"] > 0


def test_round_five_remains_fresh_and_allows_only_capacity_limited_candidates(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplement.build_initial_plan(root, dataset, tmp_path, seed="test-round-five")
    all_hashes = {
        row["image_sha256"]
        for row in supplement.read_jsonl(tmp_path / "public/direct_plan.jsonl")
    }
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": digest, "contact_stage": "test", "round": "prior"} for digest in sorted(all_hashes)],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(root, dataset, tmp_path, round_index=5, seed="test-round-five")
    fifth = {
        row["image_sha256"]
        for row in supplement.read_jsonl(tmp_path / "rounds/v5/public/direct_plan.jsonl")
    }
    assert report["round"] == 5
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert not all_hashes & fifth


def test_round_seven_is_fresh_only_and_records_capacity_limits(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    pool = supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in pool if row.get("class_code") == "N04094"],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=7, seed="round-seven", oversample=3.0,
    )
    assert report["round"] == 7
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0


def test_round_eight_is_fresh_only_and_allows_capacity_limits(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    pool = supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in pool if row.get("class_code") == "N04094"],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=8, seed="round-eight", oversample=3.0,
    )
    assert report["round"] == 8
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0


def test_round_nine_is_fresh_only_and_allows_capacity_limits(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    pool = supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in pool if row.get("class_code") == "N04094"],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=9, seed="round-nine", oversample=3.0,
    )
    assert report["round"] == 9
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0


def test_round_ten_is_fresh_only_and_allows_capacity_limits(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    pool = supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in pool if row.get("class_code") == "N04094"],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=10, seed="round-ten", oversample=3.0,
    )
    assert report["round"] == 10
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0


def test_round_eleven_is_fresh_only_and_allows_capacity_limits(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    pool = supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in pool if row.get("class_code") == "N04094"],
    )
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=11, seed="round-eleven", oversample=3.0,
    )
    assert report["round"] == 11
    assert report["capacity_limited_strategy"] == "use_all_remaining_fresh_images_only"
    assert report["capacity_limited"]["N04094"]["selected_images"] == 0


def test_round_six_replay_is_scoped_to_explicit_rejected_images(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    n04094 = [
        row for row in supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
        if row.get("class_code") == "N04094" and row.get("image_split") == "train_candidate"
    ]
    assert n04094
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"], "contact_stage": "prior", "round": "v1"} for row in n04094],
    )
    supplement.write_rows(
        tmp_path / "review/image_decisions.jsonl",
        [{"image_sha256": row["image_sha256"], "class_code": "N04094", "accepted": False} for row in n04094],
    )
    report = supplement.build_replenishment_plan(
        root, dataset, tmp_path, round_index=6, seed="round-six-replay", oversample=3.0,
        authorized_replay_codes={"N04094"},
    )
    rows = supplement.read_jsonl(tmp_path / "rounds/v6/private/alignment.jsonl")
    replayed = [row for row in rows if row.get("authorized_replay") is True]
    assert replayed
    assert {row["class_code"] for row in replayed} == {"N04094"}
    assert all(row.get("special_replay", {}).get("scope") == "N04094_N04117_N04113_only" for row in replayed)
    assert report["authorized_replay_selected_images"]["N04094"] > 0


def test_oracle_recovery_plan_is_private_rag_only_and_non_promoting(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [])
    supplement.write_rows(tmp_path / "collection/contacted_images.jsonl", [])
    report = supplement.build_oracle_recovery_plan(
        root, dataset, tmp_path, seed="oracle-plan-test", oversample=3.0,
    )
    plan = tmp_path / "recoveries/oracle-n04094-n04113-v1/private/rag_plan.jsonl"
    rows = supplement.read_jsonl(plan)
    assert report["formal_dataset_modified"] is False
    assert report["ordinary_blind_review_superseded"] is False
    assert rows and not (plan.parents[1] / "public").exists()
    assert {row["final_label"] for row in rows} <= {"N04094", "N04113"}
    assert all(row["generation_route"] == "oracle_grounded" for row in rows)
    assert all(row["label_visible_to_teacher"] is True for row in rows)
    assert all(row["special_oracle"]["standard_answer_private_to_micu"] is True for row in rows)


def test_oracle_continuation_excludes_images_assigned_to_prior_oracle_batch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    n04094 = [
        row for row in supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
        if row.get("class_code") == "N04094" and row.get("image_split") == "train_candidate"
    ]
    supplement.write_rows(
        tmp_path / "review/image_decisions.jsonl",
        [{"image_sha256": row["image_sha256"], "class_code": "N04094", "accepted": False} for row in n04094],
    )
    supplement.write_rows(
        tmp_path / "collection/contacted_images.jsonl",
        [{"image_sha256": row["image_sha256"]} for row in n04094],
    )
    first = supplement.build_oracle_recovery_plan(
        root, dataset, tmp_path, seed="oracle-continuation", oversample=3.0,
        recovery_id="oracle-n04094-first-v1", recovery_codes={"N04094"},
    )
    first_hashes = {row["image_sha256"] for row in supplement.read_jsonl(Path(first["recovery_root"]) / "private/rag_plan.jsonl")}
    second = supplement.build_oracle_recovery_plan(
        root, dataset, tmp_path, seed="oracle-continuation-next", oversample=3.0,
        recovery_id="oracle-n04094-continuation-v1", recovery_codes={"N04094"},
    )
    second_hashes = {row["image_sha256"] for row in supplement.read_jsonl(Path(second["recovery_root"]) / "private/rag_plan.jsonl")}
    assert first_hashes and second_hashes
    assert not first_hashes & second_hashes
    assert second["oracle_contacted_images_excluded"] == len(first_hashes)


def test_oracle_replay_allows_prior_oracle_images_but_excludes_complete_images(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets/AgriNet-1K/open_agri_v2"
    candidates = [
        row for row in supplement.read_jsonl(dataset / "vlm_data/candidates/image_pool.jsonl")
        if row.get("class_code") == "N04094" and row.get("image_split") == "train_candidate"
    ]
    supplement.write_rows(tmp_path / "collection/contacted_images.jsonl", [{"image_sha256": row["image_sha256"]} for row in candidates])
    supplement.write_rows(tmp_path / "review/image_decisions.jsonl", [{"image_sha256": row["image_sha256"], "class_code": "N04094", "accepted": False} for row in candidates])
    first = supplement.build_oracle_recovery_plan(root, dataset, tmp_path, seed="oracle-first", oversample=3.0, recovery_id="oracle-n04094-first-v1", recovery_codes={"N04094"})
    first_rows = supplement.read_jsonl(Path(first["recovery_root"]) / "private/rag_plan.jsonl")
    complete_digest = first_rows[0]["image_sha256"]
    supplement.write_rows(Path(first["recovery_root"]) / "collection/rag/accepted.jsonl", [
        {"sample_id": row["sample_id"]} for row in first_rows if row["image_sha256"] == complete_digest
    ])
    second = supplement.build_oracle_recovery_plan(root, dataset, tmp_path, seed="oracle-replay", oversample=3.0, recovery_id="oracle-n04094-replay-v1", recovery_codes={"N04094"}, allow_oracle_replay=True)
    replay_hashes = {row["image_sha256"] for row in supplement.read_jsonl(Path(second["recovery_root"]) / "private/rag_plan.jsonl")}
    assert complete_digest not in replay_hashes
    assert second["allow_oracle_replay"] is True
    assert second["oracle_contacted_images_excluded"] == 0


def test_oracle_provisional_audit_is_separate_and_human_gated(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    recovery = tmp_path / "recoveries/oracle-n04094-n04113-v1"
    plan = []
    accepted = []
    for code, digest in (("N04094", "d094"), ("N04113", "d113")):
        for index, (language, question_type) in enumerate((("en", "open"), ("en", "option"), ("zh", "open"), ("zh", "option"))):
            sample_id = f"sample-{code}-{index}"
            plan.append({"sample_id": sample_id, "final_label": code, "image_sha256": digest})
            accepted.append({"sample_id": sample_id, "metadata": {"image_sha256": digest, "canonical_class": code, "generation_route": "oracle_grounded", "label_visible_to_teacher": True, "label_influence_audit": {"final_evidence_supported": True, "premature_target_query": False}}, "messages": []})
    supplement.write_rows(recovery / "private/rag_plan.jsonl", plan)
    supplement.write_rows(recovery / "collection/rag/accepted.jsonl", accepted)
    supplement.write_rows(recovery / "collection/rag/rejected.jsonl", [])
    report = supplement.write_oracle_provisional_review(tmp_path)
    assert report["eligible_for_formal_merge"] is False
    assert report["training_eligible"] is False
    assert report["accepted_images_by_class"] == {"N04094": 1, "N04113": 1}
    rows = supplement.read_jsonl(tmp_path / "oracle_provisional/accepted/agent_sft.accepted.jsonl")
    assert len(rows) == 8
    assert all((row["metadata"])["source"] == "oracle_rag" for row in rows)
    assert all((row["metadata"])["oracle_provisional"] is True for row in rows)
    assert all((row["metadata"])["eligible_for_formal_merge"] is False for row in rows)
    assert (tmp_path / "review/image_decisions.jsonl").exists() is False
    exported = supplement.write_oracle_final_export(tmp_path)
    assert exported["terminal_trajectories"] == 8
    assert exported["accepted_complete_four_view_trajectories"] == 8
    assert exported["rejected_collector_or_audit_trajectories"] == 0
    assert len(supplement.read_jsonl(tmp_path / "oracle_final_export/review/terminal_records.jsonl")) == 8
