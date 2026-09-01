from copy import deepcopy
import json
from pathlib import Path

import pytest

from agrinet.data.io import DataError
from agrinet.cli import data as data_cli
from agrinet.data.m1_direct_collection import (
    audit_and_convert, auditor_request, build_open_pest_preflight_plan, build_plan,
    build_replenishment_plan, build_sample_preflight_plan, build_stratified_pilot_plan, freeze,
    derive_open_only_view, promote_open_pest_screened_plan, promote_stratified_screened_plan, teacher_request,
    interim_training_authorization, render_m1_comparison_reasoning, terminal_training_authorization,
)
from agrinet.data.m1_direct_collection import MAX_PUBLIC_KNOWLEDGE_CHARS, _public_class


def fixtures():
    classes = [
        {"code": f"N04{i:03d}", "task_domain": "disease", "english_name": f"Disease {i}", "chinese_name": f"病害{i}", "public_knowledge": f"Knowledge {i}"}
        for i in range(1, 5)
    ] + [
        {"code": f"N05{i:03d}", "task_domain": "pest", "english_name": f"Pest {i}", "chinese_name": f"虫害{i}", "public_knowledge": f"Knowledge {i}"}
        for i in range(1, 5)
    ]
    images = [
        {"class_code": row["code"], "query_image": f"data/{row['code']}.jpg", "image_sha256": f"hash-{row['code']}"}
        for row in classes
    ]
    similar = {row["code"]: [other["code"] for other in classes if other != row and other["task_domain"] == row["task_domain"]] for row in classes}
    return classes, images, similar


def test_plan_has_private_truth_separation_and_independent_option_orders():
    classes, images, similar = fixtures()
    public, private, report = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    assert report["ready_for_teacher"] and len(public) == len(private) == 32
    assert all("class_code" not in row and "correct_option" not in row for row in public)
    assert all(len(row["candidate_classes"]) == 4 for row in public)
    option = [row for row in private if row["correct_option"]]
    assert len(option) == 16
    assert {row["sample_id"].rsplit("-", 1)[-1] for row in option} == {"en", "zh"}


def test_plan_fails_on_isolation_shortage_and_cross_domain_negative():
    classes, images, similar = fixtures()
    with pytest.raises(DataError, match="insufficient isolated images"):
        build_plan(classes, images, similar, {"eval": {images[0]["image_sha256"]}}, images_per_class=1, class_target=8)
    similar[classes[0]["code"]][0] = classes[4]["code"]
    with pytest.raises(DataError, match="same domain"):
        build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)


def test_plan_rejects_hashes_from_a_runtime_contact_ledger():
    classes, images, similar = fixtures()
    contacted = {images[0]["image_sha256"]}
    with pytest.raises(DataError, match="insufficient isolated images"):
        build_plan(
            classes, images, similar, {"pilot_contacted": contacted},
            images_per_class=1, class_target=8,
        )


def test_plan_rejects_ambiguous_chinese_candidate_names():
    classes, images, similar = fixtures()
    classes[1]["chinese_name"] = classes[0]["chinese_name"]
    similar[classes[0]["code"]] = [classes[1]["code"], classes[2]["code"], classes[3]["code"]]
    with pytest.raises(DataError, match="ambiguous Chinese candidate names"):
        build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)


def test_stratified_screen_promotion_uses_current_prompt_provenance():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(
        classes, images, similar, {}, images_per_class=1, class_target=8,
    )
    selected, selected_hashes = [], set()
    for cell in (
        (question_type, language, domain)
        for question_type in ("open", "option")
        for language in ("en", "zh")
        for domain in ("disease", "pest")
    ):
        candidates = [
            row for row in public
            if (row["question_type"], row["language"], row["task_domain"]) == cell
            and row["image_sha256"] not in selected_hashes
        ]
        selected.append(candidates[0])
        selected_hashes.add(candidates[0]["image_sha256"])
    selected_ids = {row["sample_id"] for row in selected}
    screened_private = [row for row in private if row["sample_id"] in selected_ids]
    gate = {"passed": True, "schema_version": "test-screen/v1", "passed_rows": [{"sample_id": row["sample_id"]} for row in selected]}
    promoted, _, report = promote_stratified_screened_plan(
        selected, screened_private, gate, gate_config_sha256="a" * 64, attempts_per_cell=1,
    )
    assert report["ready_for_teacher"]
    assert {row["teacher_prompt_version"] for row in promoted} == {"agrinet.m1-direct-teacher/v5"}
    assert {row["auditor_prompt_version"] for row in promoted} == {"agrinet.m1-direct-auditor/v5"}


def completed_records(public, private):
    truths = {row["sample_id"]: row for row in private}
    teachers, auditors = [], []
    for task in public:
        truth = truths[task["sample_id"]]
        names = [row["name"] for row in task["candidate_classes"]]
        answer = truth["correct_option"] if task["question_type"] == "option" else (truth["truth_name_zh"] if task["language"] == "zh" else truth["truth_name"])
        teachers.append({"sample_id": task["sample_id"], "answer": answer, "reasoning": {"visual_evidence": ["visible one", "visible two", "visible three"], "candidate_comparisons": [{"name": name, "observation_or_conflict": f"specific conflict for {name}"} for name in names], "knowledge_verification": "Public cards checked", "uncertainty": {"level": "low", "reason": "three consistent traits"}}})
        auditors.append({"sample_id": task["sample_id"], "independent_choice": truth["truth_name"], "checks": {"visual_facts_supported": True, "all_candidates_compared": True, "no_invisible_facts": True, "no_label_leakage": True, "format_complete": True}})
    return teachers, auditors


def test_audit_conversion_and_freeze_contract():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, report = audit_and_convert(public, private, teachers, auditors)
    assert report["accepted"] == 32 and report["rejected"] == 0
    assert freeze(rows, {}, class_target=8, images_per_class=1)["training_authorized"]
    assert all([message["role"] for message in row["messages"]] == ["system", "user", "assistant"] for row in rows)
    with pytest.raises(DataError, match="freeze invariants"):
        freeze(rows, {"historical": {"hash-N04001"}}, class_target=8, images_per_class=1)


def test_m1_comparison_conversion_preserves_audited_reasoning_without_option_mapping():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, report = audit_and_convert(
        public, private, teachers, auditors, reasoning_render="m1_comparison_v1",
    )
    assert report["accepted"] == 32 and report["rejected"] == 0
    option_row = next(row for row in rows if row["metadata"]["question_type"] == "option")
    text = option_row["messages"][-1]["content"]
    assert "Visual observations:" in text
    assert "Candidate comparison:" in text
    assert "Knowledge verification:" in text
    assert "Uncertainty:" in text
    assert all(candidate["name"] in text for candidate in next(
        task for task in public if task["sample_id"] == option_row["sample_id"]
    )["candidate_classes"])
    assert not any(f"{letter} =" in text or f"{letter}:" in text for letter in "ABCD")
    assert option_row["metadata"]["reasoning_render_version"].endswith("m1-comparison-v1")


def test_m1_comparison_renderer_fails_closed_on_private_or_option_mapping_text():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, _ = completed_records(public, private)
    task = next(row for row in public if row["question_type"] == "option")
    teacher = next(row for row in teachers if row["sample_id"] == task["sample_id"])
    teacher["reasoning"]["knowledge_verification"] = "A = a private candidate mapping"
    with pytest.raises(DataError, match="option-mapping"):
        render_m1_comparison_reasoning(task, teacher["reasoning"])


def test_task_guidance_conversion_describes_format_without_leaking_option_mapping():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, report = audit_and_convert(
        public, private, teachers, auditors, reasoning_render="m1_comparison_task_guidance_v2",
    )
    assert report["accepted"] == 32 and report["rejected"] == 0
    open_row = next(row for row in rows if row["metadata"]["question_type"] == "open" and row["metadata"]["language"] == "en")
    option_row = next(row for row in rows if row["metadata"]["question_type"] == "option" and row["metadata"]["language"] == "en")
    open_text = open_row["messages"][-1]["content"]
    option_text = option_row["messages"][-1]["content"]
    assert "one canonical disease or pest name" in open_text
    assert "choose one of the candidates given in the question" in option_text
    assert "output only one option letter at the end" in option_text
    assert not any(f"{letter} =" in option_text or f"{letter}:" in option_text for letter in "ABCD")
    assert option_row["metadata"]["reasoning_render_version"].endswith("task-guidance-v2")


def test_user_guidance_conversion_puts_task_contract_in_user_not_think():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, report = audit_and_convert(
        public, private, teachers, auditors, reasoning_render="m1_comparison_user_guidance_v3",
    )
    assert report["accepted"] == 32 and report["rejected"] == 0
    open_row = next(row for row in rows if row["metadata"]["question_type"] == "open" and row["metadata"]["language"] == "en")
    option_row = next(row for row in rows if row["metadata"]["question_type"] == "option" and row["metadata"]["language"] == "en")
    open_user, open_assistant = open_row["messages"][-2:]
    option_user, option_assistant = option_row["messages"][-2:]
    assert "one canonical disease or pest name" in open_user["content"]
    assert "choose one of the candidates given below" in option_user["content"]
    assert "A. " in option_user["content"]
    assert "Task:" not in open_assistant["content"]
    assert "Task:" not in option_assistant["content"]
    assert not any(f"{letter} =" in option_assistant["content"] or f"{letter}:" in option_assistant["content"] for letter in "ABCD")
    assert option_row["metadata"]["reasoning_render_version"].endswith("user-guidance-v3")


def test_open_only_view_preserves_only_audited_open_rows_and_all_images():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, _ = audit_and_convert(
        public, private, teachers, auditors, reasoning_render="m1_comparison_user_guidance_v3",
    )
    selected, report = derive_open_only_view(rows)
    assert len(selected) == len(rows) // 2 == 16
    assert report["unique_images"] == len(images)
    assert set(report["cells"].values()) == {4}
    assert all(row["metadata"]["question_type"] == "open" for row in selected)
    assert {row["sample_id"] for row in selected} == {row["sample_id"] for row in rows if row["metadata"]["question_type"] == "open"}


def test_freeze_requires_exact_per_class_image_quota_when_private_mapping_is_available():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, _ = audit_and_convert(public, private, teachers, auditors)
    class_by_sample = {row["sample_id"]: row["class_code"] for row in private}
    assert freeze(
        rows, {}, class_target=8, images_per_class=1, class_by_sample=class_by_sample,
    )["invariants"]["class_image_quota"]
    wrong_mapping = dict(class_by_sample)
    for sample_id in list(wrong_mapping)[:4]:
        wrong_mapping[sample_id] = classes[1]["code"]
    with pytest.raises(DataError, match="class_image_quota"):
        freeze(rows, {}, class_target=8, images_per_class=1, class_by_sample=wrong_mapping)


def test_terminal_authorization_requires_v5_but_keeps_complete_safe_groups():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, _ = audit_and_convert(public, private, teachers, auditors)
    # A partially accepted image group is never eligible for the terminal
    # artifact either, even though its quota may be incomplete.
    rows = [row for row in rows if row["sample_id"] != public[0]["sample_id"]]
    mapping = {row["sample_id"]: row["class_code"] for row in private}
    with pytest.raises(DataError, match="exactly 5 completed"):
        terminal_training_authorization(
            rows, {}, class_target=8, images_per_class=1, class_by_sample=mapping,
            completed_replenishment_rounds=4, expected_class_codes=[row["code"] for row in classes],
        )
    # The caller filters whole partial groups before authorizing.
    partial_hash = public[0]["image_sha256"]
    rows = [row for row in rows if row["metadata"]["image_sha256"] != partial_hash]
    report = terminal_training_authorization(
        rows, {}, class_target=8, images_per_class=1, class_by_sample=mapping,
        completed_replenishment_rounds=5, expected_class_codes=[row["code"] for row in classes],
    )
    assert report["training_authorized"] is True
    assert report["quota_complete"] is False
    assert len(report["missing_class_quotas"]) == 1
    assert all(count == 4 for count in (
        sum(1 for row in rows if row["metadata"]["image_sha256"] == digest)
        for digest in {row["metadata"]["image_sha256"] for row in rows}
    ))


def test_interim_authorization_is_truth_safe_but_honestly_incomplete():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    rows, _ = audit_and_convert(public, private, teachers, auditors)
    missing_hash = public[0]["image_sha256"]
    rows = [row for row in rows if row["metadata"]["image_sha256"] != missing_hash]
    report = interim_training_authorization(
        rows, {}, class_target=8, images_per_class=1,
        class_by_sample={row["sample_id"]: row["class_code"] for row in private},
        completed_replenishment_rounds=1, expected_class_codes=[row["code"] for row in classes],
    )
    assert report["training_authorized"] is True
    assert report["quota_complete"] is False
    assert report["authorization_type"] == "user_authorized_interim_sft_before_class_shortage_replenishment"


def test_auditor_mismatch_and_vague_comparison_are_rejected():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teachers, auditors = completed_records(public, private)
    teachers[0] = deepcopy(teachers[0]); teachers[0]["reasoning"]["candidate_comparisons"][0]["observation_or_conflict"] = ""
    auditors[1] = deepcopy(auditors[1]); auditors[1]["independent_choice"] = "Wrong"
    rows, report = audit_and_convert(public, private, teachers, auditors)
    assert len(rows) == 30
    reasons = {error for item in report["rejections"] for error in item["errors"]}
    assert {"vague_candidate_comparison", "auditor_choice_mismatch"} <= reasons


def test_replenishment_replaces_an_entire_rejected_image_with_same_class_views():
    classes, images, similar = fixtures()
    # One fresh reserve exists only for the affected first class.
    images.append({
        "class_code": classes[0]["code"], "query_image": "data/fresh.jpg",
        "image_sha256": "hash-fresh",
    })
    public, private, _ = build_plan(classes, images[:-1], similar, {}, images_per_class=1, class_target=8)
    initial = []
    rejected_image = public[0]["image_sha256"]
    for task in public:
        initial.append({
            "sample_id": task["sample_id"], "accepted": task["image_sha256"] != rejected_image,
            "delivery_status": "known", "errors": [],
        })
    replenished, truth, report = build_replenishment_plan(
        classes, images, similar, {}, public, private, initial,
        gate_config_sha256="a" * 64, images_per_class=1, class_target=8,
    )
    assert report["shortages"] == {classes[0]["code"]: 1}
    assert report["unique_images"] == 1 and len(replenished) == len(truth) == 4
    assert {row["image_sha256"] for row in replenished} == {"hash-fresh"}
    assert {row["class_code"] for row in truth} == {classes[0]["code"]}
    assert {row["question_type"] for row in replenished} == {"open", "option"}
    assert {row["language"] for row in replenished} == {"en", "zh"}


def test_capacity_partial_replenishment_collects_only_fully_supplied_classes():
    classes, images, similar = fixtures()
    # The first class needs a replacement and has one fresh candidate; the
    # second class needs a replacement but has no fresh capacity.
    images.append({
        "class_code": classes[0]["code"], "query_image": "data/fresh.jpg",
        "image_sha256": "hash-fresh",
    })
    public, private, _ = build_plan(classes, images[:-1], similar, {}, images_per_class=1, class_target=8)
    rejected = {public[0]["image_sha256"], public[4]["image_sha256"]}
    ledger = [{
        "sample_id": task["sample_id"], "delivery_status": "known",
        "accepted": task["image_sha256"] not in rejected, "errors": [],
    } for task in public]
    repl_public, repl_private, report = build_replenishment_plan(
        classes, images, similar, {}, public, private, ledger,
        gate_config_sha256="a" * 64, images_per_class=1, class_target=8,
        allow_capacity_partial=True, round_name="v2",
    )
    assert report["capacity_partial"] is True
    assert report["capacity_blocked_shortages"] == {classes[1]["code"]: 1}
    assert report["collectable_shortages"] == {classes[0]["code"]: 1}
    assert report["ready_for_teacher"] is True
    assert len(repl_public) == len(repl_private) == 4
    assert {row["class_code"] for row in repl_private} == {classes[0]["code"]}


def test_replenishment_replays_only_rejected_same_class_image_when_authorized():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    retired_hash = public[0]["image_sha256"]
    ledger = [{
        "sample_id": task["sample_id"], "delivery_status": "known",
        "accepted": task["image_sha256"] != retired_hash, "errors": [],
    } for task in public]
    repl_public, repl_private, report = build_replenishment_plan(
        classes, images, similar, {"runtime_contacted": {retired_hash}},
        public, private, ledger, gate_config_sha256="a" * 64,
        images_per_class=1, class_target=8, round_name="v2",
        allow_retired_replay=True,
    )
    assert report["retired_replay_authorized"] is True
    assert report["retired_replay_images"] == 1
    assert {row["image_sha256"] for row in repl_public} == {retired_hash}
    assert {row["class_code"] for row in repl_private} == {classes[0]["code"]}
    assert all(row["replenishment_lineage"]["replay_retired_image"] for row in repl_public)


def test_replenishment_replay_does_not_override_static_isolation():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    retired_hash = public[0]["image_sha256"]
    ledger = [{
        "sample_id": task["sample_id"], "delivery_status": "known",
        "accepted": task["image_sha256"] != retired_hash, "errors": [],
    } for task in public]
    with pytest.raises(DataError, match="insufficient fresh isolated replenishment images"):
        build_replenishment_plan(
            classes, images, similar, {"formal_eval": {retired_hash}}, public, private, ledger,
            gate_config_sha256="a" * 64, images_per_class=1, class_target=8,
            allow_retired_replay=True,
        )


def test_generic_replenishment_merge_keeps_only_complete_image_groups(tmp_path: Path, monkeypatch):
    """The v2+ merger must retain a recovered image, not its retired source."""
    classes, images, similar = fixtures()
    images.append({
        "class_code": classes[0]["code"], "query_image": "data/fresh.jpg",
        "image_sha256": "hash-fresh",
    })
    public, private, _ = build_plan(classes, images[:-1], similar, {}, images_per_class=1, class_target=8)
    retired_hash = public[0]["image_sha256"]
    base_ledger = [{
        "sample_id": task["sample_id"], "delivery_status": "known",
        "accepted": task["image_sha256"] != retired_hash, "errors": [],
    } for task in public]
    repl_public, repl_private, _ = build_replenishment_plan(
        classes, images, similar, {}, public, private, base_ledger,
        gate_config_sha256="a" * 64, images_per_class=1, class_target=8,
    )
    artifact = tmp_path / "artifact"
    full_root = tmp_path / "full-run"
    config = {
        "id": "test-m1-direct",
        "outputs": {
            "public_plan": str(artifact / "public" / "teacher_plan.jsonl"),
            "private_alignment": str(artifact / "private" / "truth_alignment.jsonl"),
            "full_v1_ledger": str(artifact / "private" / "full_v1_ledger.jsonl"),
            "full_v1_run_root": str(full_root),
            "accepted_sft": str(artifact / "staging" / "accepted_sft.jsonl"),
        },
    }
    def write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(Path(config["outputs"]["public_plan"]), public)
    write(Path(config["outputs"]["private_alignment"]), private)
    write(Path(config["outputs"]["full_v1_ledger"]), base_ledger)
    base_teachers, base_auditors = completed_records(public, private)
    write(full_root / "teacher_results.jsonl", base_teachers)
    write(full_root / "auditor_results.jsonl", base_auditors)
    monkeypatch.setattr(data_cli, "repository_root", lambda: tmp_path)
    paths = data_cli._m1_direct_replenishment_paths(config, 2)
    write(paths["public"], repl_public)
    write(paths["private"], repl_private)
    paths["plan_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["plan_report"].write_text(json.dumps({"teacher_rows": 4}))
    # v2+ is only legal after v1 has crossed its private audit/merge boundary.
    # Materialize an empty, completed v1 boundary for this focused fixture.
    v1 = data_cli._m1_direct_replenishment_paths(config, 1)
    write(v1["public"], [])
    write(v1["private"], [])
    write(v1["ledger"], [])
    write(v1["run_root"] / "teacher_results.jsonl", [])
    write(v1["run_root"] / "auditor_results.jsonl", [])
    v1["plan_report"].parent.mkdir(parents=True, exist_ok=True)
    v1["plan_report"].write_text(json.dumps({"teacher_rows": 0}))
    v1["audit_report"].write_text(json.dumps({"freeze_authorized": False}))
    write(paths["ledger"], [{
        "sample_id": task["sample_id"], "delivery_status": "known", "accepted": True, "errors": [],
    } for task in repl_public])
    repl_teachers, repl_auditors = completed_records(repl_public, repl_private)
    write(paths["run_root"] / "teacher_results.jsonl", repl_teachers)
    write(paths["run_root"] / "auditor_results.jsonl", repl_auditors)

    report = data_cli._m1_direct_audit_replenishment_round(config, 2)

    staged = [json.loads(line) for line in Path(config["outputs"]["accepted_sft"]).read_text().splitlines()]
    hashes = {row["metadata"]["image_sha256"] for row in staged}
    assert report["complete_accepted_images"] == 8
    assert retired_hash not in hashes
    assert "hash-fresh" in hashes
    assert len(staged) == 32


def test_freeze_excludes_other_contacted_routes_but_not_its_own_lineage(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact"
    runtime = tmp_path / "contacted.jsonl"
    config = {
        "outputs": {
            "full_v1_ledger": str(artifact / "private" / "full_v1_ledger.jsonl"),
            "public_plan": str(artifact / "public" / "teacher_plan.jsonl"),
        },
        "inputs": {"excluded_hashes": {}, "pilot_contacted_hashes": str(runtime)},
    }
    def write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(Path(config["outputs"]["public_plan"]), [{"image_sha256": "current-base"}])
    write(artifact / "public" / "replenishment_v1_teacher_plan.jsonl", [{"image_sha256": "current-replacement"}])
    write(runtime, [{"image_sha256": "current-base"}, {"image_sha256": "current-replacement"}, {"image_sha256": "earlier-pilot"}])
    monkeypatch.setattr(data_cli, "repository_root", lambda: tmp_path)

    hashes = data_cli._m1_direct_freeze_hash_sets(config)

    assert hashes["runtime_contacted_outside_current_direct"] == {"earlier-pilot"}


def test_full_audit_allows_terminal_unknown_rows_for_replenishment(tmp_path: Path, monkeypatch):
    """Unknown delivery rejects rows; it does not suppress the cohort ledger."""
    planned = [f"sample-{index}" for index in range(4340)]
    public_path = tmp_path / "public.jsonl"
    private_path = tmp_path / "private.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    teacher_path = tmp_path / "run" / "teacher_results.jsonl"
    auditor_path = tmp_path / "run" / "auditor_results.jsonl"
    summary_path = tmp_path / "run" / "collection_summary.json"
    run_report_path = tmp_path / "run" / "run_report.json"
    accepted_path = tmp_path / "accepted.jsonl"
    audit_path = tmp_path / "audit.json"
    public_path.write_text("".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in planned))
    private_path.write_text("".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in planned))
    ledger_path.write_text("".join(json.dumps({
        "sample_id": sample_id,
        "delivery_status": "unknown" if sample_id == planned[0] else "known",
        "accepted": False, "errors": ["unknown_delivery"] if sample_id == planned[0] else [],
    }) + "\n" for sample_id in planned))
    teacher_path.parent.mkdir()
    teacher_path.write_text("")
    auditor_path.write_text("")
    summary_path.write_text(json.dumps({"attempts_planned": 4340, "delivery_status": "unknown"}))
    run_report_path.write_text(json.dumps({"attempts": 4340, "delivery_status": "unknown"}))
    config = {
        "outputs": {
            "full_v1_run_root": str(tmp_path / "run"), "full_v1_ledger": str(ledger_path),
            "public_plan": str(public_path), "private_alignment": str(private_path),
            "accepted_sft": str(accepted_path), "full_v1_audit_report": str(audit_path),
        },
    }
    monkeypatch.setattr(data_cli, "_resolved", lambda *_: config)

    data_cli.m1_direct_full_audit_command("unused")

    report = json.loads(audit_path.read_text())
    assert report["rejected"] == 4340
    assert report["replenishment_required"] is True
    assert report["freeze_authorized"] is False


def test_full_audit_requires_runtime_contact_registration_when_configured(tmp_path: Path, monkeypatch):
    """A terminal ledger cannot substitute for pre-POST image retirement."""
    planned = [f"sample-{index}" for index in range(4340)]
    public_path = tmp_path / "public.jsonl"
    private_path = tmp_path / "private.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    run = tmp_path / "run"
    contacted = tmp_path / "contacted.jsonl"
    public_path.write_text("".join(json.dumps({"sample_id": sample, "image_sha256": f"hash-{sample}"}) + "\n" for sample in planned))
    private_path.write_text("".join(json.dumps({"sample_id": sample}) + "\n" for sample in planned))
    ledger_path.write_text("".join(json.dumps({"sample_id": sample, "delivery_status": "known", "accepted": False, "errors": []}) + "\n" for sample in planned))
    run.mkdir(); (run / "teacher_results.jsonl").write_text(""); (run / "auditor_results.jsonl").write_text("")
    (run / "collection_summary.json").write_text(json.dumps({"attempts_planned": 4340}))
    (run / "run_report.json").write_text(json.dumps({"attempts": 4340, "delivery_status": "complete"}))
    contacted.write_text(json.dumps({"image_sha256": "other-route"}) + "\n")
    config = {"inputs": {"pilot_contacted_hashes": str(contacted)}, "outputs": {
        "full_v1_run_root": str(run), "full_v1_ledger": str(ledger_path),
        "public_plan": str(public_path), "private_alignment": str(private_path),
        "accepted_sft": str(tmp_path / "accepted.jsonl"), "full_v1_audit_report": str(tmp_path / "audit.json"),
    }}
    monkeypatch.setattr(data_cli, "_resolved", lambda *_: config)
    with pytest.raises(DataError, match="retired in the contacted ledger"):
        data_cli.m1_direct_full_audit_command("unused")


def test_replenishment_v1_collect_resumes_exactly_pre_registered_plan(tmp_path: Path, monkeypatch):
    """A startup retry may use its own atomic registration, but no other hash."""
    plan = tmp_path / "replenishment-plan.jsonl"
    report = tmp_path / "replenishment-report.json"
    contacted = tmp_path / "contacted.jsonl"
    plan.write_text(json.dumps({"image_sha256": "replacement"}) + "\n")
    report.write_text(json.dumps({"ready_for_teacher": True, "teacher_rows": 4}))
    contacted.write_text(json.dumps({
        "image_sha256": "replacement", "contact_stage": "full_direct_replenishment", "round": "v1",
    }) + "\n")
    config = {
        "id": "unused",
        "inputs": {"pilot_contacted_hashes": str(contacted), "acceptance_gates": str(tmp_path / "gates.yaml")},
        "outputs": {
            "replenishment_v1_plan_report": str(report), "replenishment_v1_public_plan": str(plan),
            "replenishment_v1_private_alignment": str(tmp_path / "private.jsonl"),
            "replenishment_v1_ledger": str(tmp_path / "ledger.jsonl"),
        },
        "parameters": {"collection_workers": 8},
    }
    (tmp_path / "gates.yaml").write_text("gate: true\n")
    monkeypatch.setattr(data_cli, "_resolved", lambda *_: config)
    monkeypatch.setattr(data_cli, "repository_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(data_cli.subprocess, "run", lambda command, **kwargs: calls.append(command) or type("R", (), {"returncode": 0})())
    data_cli.m1_direct_replenishment_v1_collect_command("unused")
    assert calls and "--workers" in calls[0]

    contacted.write_text(contacted.read_text() + json.dumps({
        "image_sha256": "unrelated", "contact_stage": "other", "round": "v1",
    }) + "\n")
    # An unrelated contact remains harmless: only overlap with the plan is checked.
    data_cli.m1_direct_replenishment_v1_collect_command("unused")


def test_auditor_request_is_label_blind_while_teacher_request_is_private():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    teacher_payload = teacher_request(public[0], private[0])
    audit_payload = auditor_request(public[0], {"reasoning": {"visual_evidence": ["x"]}})
    assert "private_truth" in teacher_payload
    serialized = str(audit_payload).lower()
    assert private[0]["class_code"].lower() not in serialized
    assert "private_truth" not in serialized and "correct_option" not in serialized
    assert audit_payload["answer_under_audit"] is None
    open_payload = auditor_request(public[0], {"answer": "Disease 1", "reasoning": {}})
    if public[0]["question_type"] == "open":
        assert open_payload["answer_under_audit"] == "Disease 1"
    assert [row["name"] for row in audit_payload["candidate_classes"]] != [
        row["name"] for row in public[0]["candidate_classes"]
    ]


def test_request_provenance_uses_the_plan_prompt_versions():
    classes, images, similar = fixtures()
    public, private, _ = build_plan(classes, images, similar, {}, images_per_class=1, class_target=8)
    task = dict(public[0], teacher_prompt_version="agrinet.m1-direct-teacher/v5", auditor_prompt_version="agrinet.m1-direct-auditor/v5")
    assert teacher_request(task, private[0])["prompt_version"] == "agrinet.m1-direct-teacher/v5"
    assert auditor_request(task, {"reasoning": {}})["prompt_version"] == "agrinet.m1-direct-auditor/v5"


def test_sample_preflight_plan_is_two_domains_and_keeps_confirmation_private():
    classes = [
        {"code": f"N04{i:03d}", "task_domain": "disease", "english_name": f"Disease {i}",
         "chinese_name": f"病害{i}", "public_knowledge": f"Disease knowledge {i}"}
        for i in range(1, 146)
    ] + [
        {"code": f"N05{i:03d}", "task_domain": "pest", "english_name": f"Pest {i}",
         "chinese_name": f"虫害{i}", "public_knowledge": f"Pest knowledge {i}"}
        for i in range(1, 73)
    ]
    images = [
        {"class_code": row["code"], "query_image": f"data/{row['code']}.jpg",
         "image_sha256": f"hash-{row['code']}"} for row in classes
    ]
    similar = {
        row["code"]: [other["code"] for other in classes if other["task_domain"] == row["task_domain"] and other != row][:3]
        for row in classes
    }
    supplemental = [{
        "class_code": "N05053", "image_sha256": "hash-N05053",
        "needs_independent_label_confirmation": True,
    }]
    public, private, report = build_sample_preflight_plan(
        classes, images, similar, {}, supplemental, gate_config_sha256="a" * 64,
    )
    assert report["ready_for_teacher"] and report["marked_n05053_included"]
    assert len(public) == len(private) == 8
    assert {row["task_domain"] for row in public} == {"disease", "pest"}
    assert all(row["gate_config_sha256"] == "a" * 64 for row in public)
    assert all("class_code" not in row and "query_image" not in row for row in public)
    assert sum(row["needs_independent_label_confirmation"] for row in private) == 4


def test_open_pest_preflight_plan_is_fresh_strictly_targeted_and_private():
    classes = [
        {"code": f"N04{i:03d}", "task_domain": "disease", "english_name": f"Disease {i}", "chinese_name": f"病害{i}", "public_knowledge": "d"}
        for i in range(1, 146)
    ] + [
        {"code": f"N05{i:03d}", "task_domain": "pest", "english_name": f"Pest {i}", "chinese_name": f"虫害{i}", "public_knowledge": "p"}
        for i in range(1, 73)
    ]
    images = [
        {"class_code": item["code"], "query_image": f"data/{item['code']}.jpg", "image_sha256": f"hash-{item['code']}"}
        for item in classes
    ]
    similar = {
        item["code"]: [other["code"] for other in classes if other["task_domain"] == item["task_domain"] and other != item][:3]
        for item in classes
    }
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, {}, {"hash-N05001"}, gate_config_sha256="b" * 64, attempts_per_cell=8,
    )
    assert report["ready_for_teacher"] and report["unique_images"] == 16
    assert {(row["question_type"], row["language"], row["task_domain"]) for row in public} == {
        ("open", "en", "pest"), ("open", "zh", "pest"),
    }
    assert all(row["image_sha256"] != "hash-N05001" and "class_code" not in row for row in public)
    assert all(row["correct_option"] is None and row["class_code"].startswith("N05") for row in private)


def test_screen_promotion_uses_only_private_gate_passed_rows():
    classes = [
        {"code": f"N04{i:03d}", "task_domain": "disease", "english_name": f"Disease {i}", "chinese_name": f"病害{i}", "public_knowledge": "d"}
        for i in range(1, 146)
    ] + [
        {"code": f"N05{i:03d}", "task_domain": "pest", "english_name": f"Pest {i}", "chinese_name": f"虫害{i}", "public_knowledge": "p"}
        for i in range(1, 73)
    ]
    images = [{"class_code": row["code"], "query_image": f"data/{row['code']}.jpg", "image_sha256": f"hash-{row['code']}"} for row in classes]
    similar = {row["code"]: [other["code"] for other in classes if other["task_domain"] == row["task_domain"] and other != row][:3] for row in classes}
    public, private, _ = build_open_pest_preflight_plan(
        classes, images, similar, {}, set(), gate_config_sha256="s" * 64, attempts_per_cell=8,
    )
    screen_gate = {
        "passed": True, "schema_version": "screen/v1",
        "passed_rows": [{"sample_id": row["sample_id"]} for row in public],
    }
    promoted, promoted_private, report = promote_open_pest_screened_plan(
        public, private, screen_gate, gate_config_sha256="t" * 64, attempts_per_cell=8,
    )
    assert report["ready_for_teacher"] and len(promoted) == len(promoted_private) == 16
    assert all(row["screen_lineage"]["screen_gate_passed"] for row in promoted)
    screen_gate["passed_rows"] = screen_gate["passed_rows"][:-1]
    with pytest.raises(DataError, match="screen lacks"):
        promote_open_pest_screened_plan(public, private, screen_gate, gate_config_sha256="t" * 64, attempts_per_cell=8)


def test_stratified_screen_promotion_requires_each_cell_and_preserves_isolation():
    public, private = [], []
    for index, (question_type, language, domain) in enumerate(
        (item for item in ((q, l, d) for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest")))
    ):
        sample_id = f"screen-{index}"
        public.append({
            "sample_id": sample_id, "image_ref": sample_id, "image_sha256": f"hash-{index}",
            "question_type": question_type, "language": language, "task_domain": domain,
            "candidate_classes": [{"name": name, "name_zh": name, "public_knowledge": "k"} for name in ("Truth", "A", "B", "C")],
            "hard_negative_lineage": {"source": "joint", "count": 3}, "gate_config_sha256": "s" * 64,
        })
        private.append({"sample_id": sample_id, "class_code": f"N{index:05d}", "query_image": f"image-{index}.jpg", "truth_name": "Truth", "truth_name_zh": "真值", "correct_option": None})
    gate = {"passed": True, "schema_version": "screen/v1",
            "passed_rows": [{"sample_id": row["sample_id"]} for row in public]}
    promoted, promoted_private, report = promote_stratified_screened_plan(
        public, private, gate, gate_config_sha256="t" * 64, attempts_per_cell=1,
    )
    assert report["ready_for_teacher"] and len(promoted) == len(promoted_private) == 8
    assert all(row["screen_lineage"]["screen_gate_passed"] for row in promoted)
    gate["passed_rows"] = gate["passed_rows"][:-1]
    with pytest.raises(DataError, match="screen lacks"):
        promote_stratified_screened_plan(public, private, gate, gate_config_sha256="t" * 64, attempts_per_cell=1)


def test_public_knowledge_is_bounded_without_removing_public_evidence():
    item = _public_class({"english_name": "Pest", "chinese_name": "虫害", "public_knowledge": "x" * (MAX_PUBLIC_KNOWLEDGE_CHARS + 10)})
    assert item["public_knowledge"].startswith("x" * MAX_PUBLIC_KNOWLEDGE_CHARS)
    assert item["public_knowledge"].endswith("[Public entry excerpt ends here.]")
