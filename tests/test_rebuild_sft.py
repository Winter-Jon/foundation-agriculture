from __future__ import annotations

from collections import Counter

from pathlib import Path

from agrinet.data.rebuild_sft import CELLS, SAMPLING_CONTRACT, STAGE_SPECS, attempt_budget, load_collection_specification, load_hermes_1to1_specification, long_direct_audit_errors, minimum_classes, next_trajectory, select_rag_rows, strict_rag_trajectory_errors, validate_direct_rows, validate_hermes_1to1_rows, validate_rag_rows


def _row(cell: tuple[str, str, str], index: int, *, route: str = "blind_evidence", cls: str = "N04001") -> dict:
    question_type, language, domain = cell
    answer = "A" if question_type == "option" else "leaf blight"
    return {
        "sample_id": f"{question_type}-{language}-{domain}-{index}",
        "images": [f"images/{question_type}-{language}-{domain}-{index}.jpg"],
        "messages": [{"role": "user", "content": "<image> question"}, {"role": "assistant", "content": "I will retrieve evidence."}, {"role": "tool", "content": "public retrieval result"}, {"role": "assistant", "content": f"<think>visual evidence</think><answer>{answer}</answer>"}],
        "tools": [{"type": "function", "function": {"name": "retrieve"}}],
        "metadata": {"question_type": question_type, "language": language, "task_domain": domain, "generation_route": route, "canonical_class": cls, "image_sha256": f"hash-{question_type}-{language}-{domain}-{index}", "label_visible_to_teacher": route == "oracle_grounded"},
    }


def test_intermediate_direct_quota_and_tool_free_contract() -> None:
    rows = [_row(cell, index) for cell in CELLS for index in range(STAGE_SPECS["intermediate"]["direct_per_cell"])]
    for row in rows:
        row.pop("tools")
        row["messages"] = [row["messages"][0], row["messages"][-1]]
    assert validate_direct_rows(rows, stage="intermediate", image_hashes=set())["valid"]
    rows[0]["tools"] = [{"type": "function"}]
    assert not validate_direct_rows(rows, stage="intermediate", image_hashes=set())["valid"]


def test_direct_allows_only_en_zh_same_image_pairs() -> None:
    rows = [_row(cell, index) for cell in CELLS for index in range(STAGE_SPECS["intermediate"]["direct_per_cell"])]
    for row in rows:
        row.pop("tools")
        row["messages"] = [row["messages"][0], row["messages"][-1]]
    partner = next(row for row in rows if row["metadata"]["question_type"] == "open" and row["metadata"]["task_domain"] == "disease" and row["metadata"]["language"] == "zh")
    original = next(row for row in rows if row["metadata"]["question_type"] == "open" and row["metadata"]["task_domain"] == "disease" and row["metadata"]["language"] == "en")
    partner["metadata"]["image_sha256"] = original["metadata"]["image_sha256"]
    assert validate_direct_rows(rows, stage="intermediate", image_hashes=set())["valid"]
    partner["metadata"]["language"] = "en"
    assert not validate_direct_rows(rows, stage="intermediate", image_hashes=set())["valid"]


def test_rag_rejects_blind_label_leak_and_image_overlap() -> None:
    rows = []
    for cell in CELLS:
        for index in range(35):
            rows.append(_row(cell, index, cls=f"class-{index % 18}"))
    report = validate_rag_rows(rows, stage="intermediate", image_hashes=set())
    assert report["valid"]
    rows[0]["metadata"]["label_visible_to_teacher"] = True
    report = validate_rag_rows(rows, stage="intermediate", image_hashes={rows[1]["metadata"]["image_sha256"]})
    assert "blind_label_exposure" in " ".join(report["errors"])
    assert "rag_image_isolation_overlap" in report["errors"]


def test_rag_oracle_cap_and_class_cap_are_hard_gates() -> None:
    rows = []
    for cell in CELLS:
        for index in range(35):
            route = "oracle_grounded" if index < 8 else "blind_evidence"
            rows.append(_row(cell, index, route=route, cls=f"class-{index % 18}"))
    report = validate_rag_rows(rows, stage="intermediate", image_hashes=set())
    assert not report["valid"]
    assert any(error.endswith("oracle_cap") for error in report["errors"])


def test_final_oracle_cap_and_blind_floor_are_hard_gates() -> None:
    spec = load_hermes_1to1_specification()
    direct, rag = [], []
    for cell in CELLS:
        for index in range(spec["direct_per_cell"]):
            row = _row(cell, index)
            row.pop("tools")
            row["messages"] = [row["messages"][0], row["messages"][-1]]
            row["metadata"].update({"canonical_class": f"class-{index % 35}", "paired_image_id": f"pair-{cell[0]}-{cell[2]}-{index}", "uncertainty": spec["uncertainty"]})
            direct.append(row)
        for index in range(spec["rag_per_cell"]):
            row = _row(cell, index, route="oracle_grounded" if index < 15 else "blind_evidence", cls=f"class-{index % 35}")
            row["metadata"]["uncertainty"] = spec["uncertainty"]
            rag.append(row)
    errors = validate_hermes_1to1_rows(direct, rag, forbidden_hashes=set())["errors"]
    assert any(error.endswith("oracle_cap") for error in errors)
    assert any(error.endswith("blind_floor") for error in errors)


def test_rag_requires_tool_schema_and_returned_evidence() -> None:
    rows = []
    for cell in CELLS:
        for index in range(35):
            rows.append(_row(cell, index, cls=f"class-{index % 18}"))
    rows[0].pop("tools")
    rows[1]["messages"] = [rows[1]["messages"][0], rows[1]["messages"][-1]]
    errors = validate_rag_rows(rows, stage="intermediate", image_hashes=set())["errors"]
    assert any(error.endswith("rag_missing_tool_schema") for error in errors)
    assert any(error.endswith("rag_missing_retrieval_evidence") for error in errors)


def test_rag_accepts_legacy_tool_response_evidence_serialization() -> None:
    rows = []
    for cell in CELLS:
        for index in range(35):
            row = _row(cell, index, cls=f"class-{index % 18}")
            row["messages"][2]["role"] = "tool_response"
            rows.append(row)
    assert validate_rag_rows(rows, stage="intermediate", image_hashes=set())["valid"]


def test_attempt_budget_is_bounded_and_unknown_delivery_fail_closed() -> None:
    valid = attempt_budget("full", {"open/en/disease": {"blind": 704, "oracle": 140}})
    assert valid["valid"]
    invalid = attempt_budget("intermediate", {"open/en/disease": {"blind": 141, "oracle": 28}})
    assert not invalid["valid"]
    assert Counter(error.split(":")[-1] for error in invalid["errors"]) == Counter({"blind_attempt_cap": 1})


def test_rag_label_code_is_valid_canonical_class_fallback() -> None:
    rows = []
    for cell in CELLS:
        for index in range(35):
            row = _row(cell, index, cls=f"class-{index % 18}")
            row["metadata"].pop("canonical_class")
            row["metadata"]["label_code"] = f"class-{index % 18}"
            rows.append(row)
    assert validate_rag_rows(rows, stage="intermediate", image_hashes=set())["valid"]


def test_full_class_floor_is_domain_specific() -> None:
    assert minimum_classes("full", "disease") == 100
    assert minimum_classes("full", "pest") == 50


def test_query_image_accepts_evaluation_manifest_image_path() -> None:
    from agrinet.data.rebuild_sft import query_image

    assert query_image({"image_path": "datasets/example.jpg"}) == "datasets/example.jpg"


def test_external_collection_specification_keeps_quotas_out_of_contract() -> None:
    specification = load_collection_specification(Path("configs/sampling/reconstructive-blind-rag-collection-spec-v1.yaml"))
    assert specification["intermediate"]["rag_per_cell"] == 35
    assert specification["full"]["rag_min_classes"] == {"disease": 100, "pest": 50}
    assert "rag_per_cell" not in SAMPLING_CONTRACT["rag"]


def test_full_option_cells_require_balanced_answer_letters() -> None:
    rows = []
    for cell in CELLS:
        for index in range(175):
            row = _row(cell, index, cls=f"class-{index % (100 if cell[2] == 'disease' else 50)}")
            if cell[0] == "option":
                row["metadata"]["correct_option"] = "A"
            rows.append(row)
    report = validate_rag_rows(rows, stage="full", image_hashes=set())
    assert any("option_letter_floor" in error for error in report["errors"])


def test_full_rag_accepts_domain_floors_and_option_balance() -> None:
    rows = []
    for cell in CELLS:
        class_count = 100 if cell[2] == "disease" else 50
        for index in range(175):
            row = _row(cell, index, cls=f"class-{index % class_count}")
            if cell[0] == "option":
                row["metadata"]["correct_option"] = "ABCD"[index % 4]
            rows.append(row)
    assert validate_rag_rows(rows, stage="full", image_hashes=set())["valid"]


def test_independent_resend_state_machine_stops_after_second_retryable_outcome() -> None:
    first = next_trajectory("hash", [])
    assert first and first["trajectory_number"] == 1 and first["resample_reason"] == "new_query_image"
    resend = next_trajectory("hash", ["unknown_delivery"])
    assert resend and resend["trajectory_number"] == 2 and resend["resample_reason"] == "unknown_delivery_resend"
    assert next_trajectory("hash", ["unknown_delivery", "strict_rejected"]) is None
    assert next_trajectory("hash", ["accepted"]) is None


def test_selection_reaudits_and_keeps_one_lowest_tool_turn_winner() -> None:
    cell = ("open", "en", "disease")
    rows = [_row(cell, index, cls=f"class-{index}") for index in range(35)]
    duplicate = _row(cell, 100, cls="class-0")
    duplicate["metadata"]["image_sha256"] = rows[0]["metadata"]["image_sha256"]
    duplicate["messages"].insert(-1, {"role": "tool", "content": "extra evidence"})
    rows.append(duplicate)
    result = select_rag_rows(rows, stage="intermediate", forbidden_hashes=set())
    chosen = [row for row in result["selected"] if row["metadata"]["image_sha256"] == rows[0]["metadata"]["image_sha256"]]
    assert len(chosen) == 1 and chosen[0]["sample_id"] == rows[0]["sample_id"]
    assert any(item["reason"] == "duplicate_image_lost_tiebreak" for item in result["excluded"])


def test_selection_excludes_historical_protocol_and_isolation_conflicts() -> None:
    row = _row(("open", "en", "disease"), 0)
    result = select_rag_rows([row], stage="intermediate", forbidden_hashes={row["metadata"]["image_sha256"]})
    assert not result["selected"]
    assert result["excluded"][0]["reason"] == "image_isolation_overlap"


def test_selection_reaudits_answer_contracts() -> None:
    row = _row(("option", "en", "disease"), 0)
    row["messages"][-1]["content"] = "<answer>leaf blight</answer>"
    result = select_rag_rows([row], stage="intermediate", forbidden_hashes=set())
    assert not result["selected"]
    assert result["excluded"][0]["reason"] == "option_answer_contract"


def _long_direct(cell: tuple[str, str, str], index: int) -> dict:
    question_type, language, domain = cell
    answer = "A" if question_type == "option" else "leaf blight"
    think = "\n".join([
        "1. Candidate one shows a diffuse pattern.",
        "2. Candidate two has circular lesions.",
        "3. Candidate three is a chewing injury.",
        "4. Candidate four matches the visible margin.",
        "I exclude candidate one because the spot boundary is unlike the image.",
        "I rule out candidate two because its halo is absent.",
        "I exclude candidate three because there is no missing tissue.",
        "The observed vein-adjacent necrosis supports the fourth candidate. " * 7,
    ])
    return {
        "sample_id": f"long-{question_type}-{language}-{domain}-{index}",
        "images": [f"images/{question_type}-{domain}-{index}.jpg"],
        "messages": [{"role": "user", "content": "<image> identify"}, {"role": "assistant", "content": f"<think>{think}</think><answer>{answer}</answer>"}],
        "metadata": {"question_type": question_type, "language": language, "task_domain": domain, "canonical_class": f"class-{index % 35}", "image_sha256": f"direct-{question_type}-{domain}-{index}", "paired_image_id": f"pair-{index}", "correct_option": "ABCD"[index % 4], "uncertainty": load_hermes_1to1_specification()["uncertainty"]},
    }


def _strict_rag(cell: tuple[str, str, str], index: int) -> dict:
    row = _row(cell, index, cls=f"class-{index % 35}")
    row["messages"] = [
        {"role": "user", "content": "<image> identify"},
        {"role": "assistant", "content": "<think>1. Brown leaf spots with pale centers.\n2. Dark blight along the leaf margin.\n3. Chewing injury with missing leaf tissue.</think>"},
        {"role": "tool_call", "content": "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf lesion\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesion\"}}"},
        {"role": "tool", "content": "{\"source\":\"public AgriNet catalog\",\"results\":[\"leaf blight\"]}"},
        {"role": "assistant", "content": f"<think>1. Leaf blight.\n2. Leaf spot.\n3. Chewing injury. Public evidence supports leaf blight. Evidence excludes leaf spot because the catalog margin differs. Evidence rules out chewing injury because no missing tissue appears.</think><answer>{'A' if cell[0] == 'option' else 'leaf blight'}</answer>"},
    ]
    row["metadata"]["correct_option"] = "ABCD"[index % 4]
    row["metadata"]["uncertainty"] = load_hermes_1to1_specification()["uncertainty"]
    if cell[0] == "option": row["messages"][-1]["content"] = row["messages"][-1]["content"].replace("<answer>A</answer>", f"<answer>{row['metadata']['correct_option']}</answer>")
    return row


def test_long_direct_audit_preserves_long_comparison_and_rejects_short_trace() -> None:
    row = _long_direct(("open", "en", "disease"), 0)
    assert not long_direct_audit_errors(row, forbidden_hashes=set())
    row["messages"][-1]["content"] = "<think>short</think><answer>leaf blight</answer>"
    assert "direct_think_too_short" in long_direct_audit_errors(row, forbidden_hashes=set())


def test_long_direct_uses_lower_but_structural_chinese_length_floor() -> None:
    row = _long_direct(("open", "zh", "disease"), 0)
    content = row["messages"][-1]["content"]
    think = content.split("<think>", 1)[1].split("</think>", 1)[0]
    compact = "\n".join(think.splitlines()[:7]) + " " + "可见叶脉附近的褐色坏死与黄色晕圈持续支持这一选择。" * 4
    assert len(compact) >= 280
    row["messages"][-1]["content"] = content.replace(think, compact)
    assert "direct_think_too_short" not in long_direct_audit_errors(row, forbidden_hashes=set())


def test_long_direct_accepts_chinese_natural_negative_comparison_wording() -> None:
    row = _long_direct(("option", "zh", "pest"), 0)
    content = row["messages"][-1]["content"].replace("I exclude candidate three", "不考虑候选三")
    row["messages"][-1]["content"] = content
    assert "insufficient_neighbor_exclusions" not in long_direct_audit_errors(row, forbidden_hashes=set())


def test_strict_rag_requires_adjacent_valid_hermes_call_and_evidence_exclusions() -> None:
    row = _strict_rag(("open", "en", "disease"), 0)
    assert not strict_rag_trajectory_errors(row)
    row["messages"][3]["content"] = "public prose"
    assert "tool_evidence_not_public_json" in strict_rag_trajectory_errors(row)


def test_strict_rag_rejects_unreadable_or_internal_retrieval_evidence() -> None:
    row = _strict_rag(("open", "en", "disease"), 0)
    row["messages"][3]["content"] = '{"hybrid":[{"id":"agri_disease_pest_wiki::N04030","distance":0.9}]}'
    errors = strict_rag_trajectory_errors(row)
    assert "tool_evidence_missing_readable_class_names" in errors
    assert "tool_evidence_internal_identifier_leak" in errors


def test_strict_rag_rejects_bare_or_ranking_only_evidence_candidates() -> None:
    row = _strict_rag(("open", "en", "disease"), 0)
    row["messages"][1]["content"] = "<think>1. Leaf blight\n2. Leaf spot\n3. Chewing injury</think>"
    row["messages"][-1]["content"] = row["messages"][-1]["content"].replace("Evidence excludes leaf spot because the catalog margin differs.", "Evidence excludes leaf spot because its similarity score is lower.")
    errors = strict_rag_trajectory_errors(row)
    assert "pre_tool_candidates_not_visual_descriptions" in errors
    assert "ranking_only_evidence_exclusion" in errors


def test_strict_rag_accepts_inline_numbered_visual_candidates() -> None:
    row = _strict_rag(("open", "en", "disease"), 0)
    row["messages"][1]["content"] = "<think>1. Deep green leaf with intact veins. 2. Brown circular lesion with pale center. 3. Missing tissue hole along leaf margin.</think>"
    assert "pre_tool_candidates_not_visual_descriptions" not in strict_rag_trajectory_errors(row)


def test_1to1_requires_fixed_neutral_uncertainty() -> None:
    row = _long_direct(("open", "en", "disease"), 0)
    row["metadata"]["uncertainty"] = "high confidence"
    assert "uncertainty_not_fixed_neutral" in long_direct_audit_errors(row, forbidden_hashes=set())


def test_hermes_1to1_global_contract_rejects_cross_route_image_overlap() -> None:
    spec = load_hermes_1to1_specification()
    direct = [_long_direct(cell, index) for cell in CELLS for index in range(spec["direct_per_cell"])]
    rag = [_strict_rag(cell, index) for cell in CELLS for index in range(spec["rag_per_cell"])]
    assert validate_hermes_1to1_rows(direct, rag, forbidden_hashes=set())["valid"]
    rag[0]["metadata"]["image_sha256"] = direct[0]["metadata"]["image_sha256"]
    assert "direct_rag_image_overlap" in validate_hermes_1to1_rows(direct, rag, forbidden_hashes=set())["errors"]


def test_large_freeze_discovery_uses_current_names_and_row_provenance(tmp_path: Path) -> None:
    """Current checkpointed runs are ``large-*``, not the historical prefix."""
    from agrinet.data.sft_recovery import write_jsonl
    from archive.source.rag_distill.select_hermes_1to1_large_freeze import accepted, eligible_accepted_paths

    current = tmp_path / "large-004-direct-open-en-disease"
    current.mkdir()
    direct = _long_direct(("open", "en", "disease"), 0)
    direct["metadata"]["generation_route"] = "direct_visual_comparison"
    write_jsonl(current / "accepted.jsonl", [direct])

    re_audit = tmp_path / "large-003-direct-open-zh-disease-re-audit"
    re_audit.mkdir()
    chinese = _long_direct(("open", "zh", "disease"), 0)
    chinese["metadata"]["generation_route"] = "direct_visual_comparison"
    write_jsonl(re_audit / "accepted.jsonl", [chinese])

    oracle = tmp_path / "oracle-rag-recovery-009"
    oracle.mkdir()
    rag = _strict_rag(("open", "en", "disease"), 0)
    rag["metadata"]["generation_route"] = "oracle_grounded"
    write_jsonl(oracle / "accepted.jsonl", [rag])

    historical = tmp_path / "rag-medium-001"
    historical.mkdir()
    write_jsonl(historical / "accepted.jsonl", [rag])

    assert {path.parent.name for path in eligible_accepted_paths(tmp_path)} == {current.name, re_audit.name, oracle.name}
    assert [row["sample_id"] for row in accepted(tmp_path, route="direct")] == [chinese["sample_id"], direct["sample_id"]]
    assert [row["sample_id"] for row in accepted(tmp_path, route="rag")] == [rag["sample_id"]]
