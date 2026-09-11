import json
from pathlib import Path

from agrinet.rag.micu_classifier_hcv_e2_dynamic_smoke import (
    CELLS, ROUTES, _answer_for_student, _canary_gate, _public_registry_names, rewrite_prompt_v3,
    route_contract_errors, validate_rewrite, validate_rewrite_v2, validate_rewrite_v3,
    render_rewrite_reasoning_v2, freeze_source,
)


def _row(index: int, cell: str, arm: str) -> dict:
    question_type, language, domain = cell.split("-")
    return {"sample_id": f"s-{index}", "image_group_id": f"image:{index}", "source_group_id": f"source:{index}",
            "near_duplicate_group_id": f"near:{index}", "question_type": question_type, "language": language,
            "task_domain": domain, "private": {"sampling_arm": arm}}


def test_dynamic_route_contract_is_fixed() -> None:
    assert ROUTES["direct"]["attempts"] == 3 and ROUTES["direct"]["temperature"] == 0.5
    assert ROUTES["classifier"]["attempts"] == 2 and ROUTES["classifier"]["temperature"] == 0.2
    assert ROUTES["rag"]["attempts"] == 1 and ROUTES["rag"]["temperature"] == 0.2
    assert not ROUTES["direct"]["tools"]
    assert "agrinet_rag_search" not in ROUTES["classifier"]["tools"]


def test_freeze_source_selects_two_arms_per_cell_and_excludes_prior_groups(tmp_path: Path) -> None:
    pool = []
    for cell_index, cell in enumerate(CELLS):
        for pattern_index, pattern in enumerate(("P1", "P2", "P3", "P4", "P5")):
            for copy in range(2):
                row = _row(cell_index * 20 + pattern_index * 2 + copy, cell, "targeted")
                row["private"]["target_pattern"] = pattern
                pool.append(row)
    prior = [pool[0]]
    pool_path, prior_path, output = tmp_path / "pool.jsonl", tmp_path / "prior.jsonl", tmp_path / "source.json"
    pool_path.write_text("".join(json.dumps(row) + "\n" for row in pool), encoding="utf-8")
    prior_path.write_text(json.dumps(prior[0]) + "\n", encoding="utf-8")
    report = freeze_source(pool=pool_path, prior_source=prior_path, output=output, seed="fixed")
    frozen = json.loads(output.read_text())
    assert report["rows"] == 32 and len(frozen["rows"]) == 32
    assert report["arms"] == {"targeted": 32}
    assert report["pattern_targets"] == {"P1": 8, "P2": 8, "P3": 8, "P4": 8}
    assert prior[0]["image_group_id"] not in {row["image_group_id"] for row in frozen["rows"]}


def test_route_contract_requires_real_route_tools() -> None:
    direct = {"status": "closed", "final": "Leaf spot", "trace": []}
    assert route_contract_errors(direct, "direct", audit_status="accept") == []
    assert "classifier_predict_missing" in route_contract_errors(direct, "classifier", audit_status="accept")
    classifier = {"status": "closed", "final": "Leaf spot", "trace": [
        {"action": {"type": "tool", "name": "agrinet_classifier_predict"}}]}
    assert route_contract_errors(classifier, "classifier", audit_status="accept") == []
    assert "rag_search_missing" in route_contract_errors(classifier, "rag", audit_status="accept")


def test_dynamic_canary_gate_requires_two_clean_rounds(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = {"endpoint": "https://api-slb.micuapi.ai/v1", "model": "gpt-5.6-sol",
               "generation_ready_for_dynamic_smoke": True,
               "rounds": [{"requests": 10, "valid_deliveries": 10, "pass": True}] * 2}
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert len(_canary_gate(report, endpoint=payload["endpoint"], teacher_model=payload["model"])) == 64
    payload["rounds"][1]["valid_deliveries"] = 9
    report.write_text(json.dumps(payload), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="10/10"):
        _canary_gate(report, endpoint=payload["endpoint"], teacher_model=payload["model"])


def test_rewrite_contract_and_option_semantic_answer() -> None:
    row = {"language": "en", "question_type": "option", "public_options": [
        {"label": "A", "name": "Apple scab", "name_zh": "苹果黑星病"}]}
    parent = {"final": "A"}
    rewrite = {"final": "A", "reasoning": "Visual observations:\n- brown spots\n- curled leaves\n- apple foliage\nCandidate comparison:\n- Apple scab: lesions fit.\n- Apple rust: no orange pustules.\n- Powdery mildew: no white coating.\nEvidence:\n- visible image findings only.\nUncertainty:\n- leaf underside is not visible."}
    assert validate_rewrite(row=row, route="direct", parent=parent, rewrite=rewrite) == []
    assert _answer_for_student(row, "A") == "Apple scab — A"


def test_future_rewrite_v2_schema_renders_and_preserves_direct_boundary() -> None:
    row, parent = {"language": "en", "question_type": "open"}, {"final": "Apple scab"}
    rewrite = {"final": "Apple scab", "visual_observations": ["brown spots", "curled leaves", "rough lesions"],
               "candidate_comparisons": [{"candidate": "Apple scab", "support": "rough lesions", "counterevidence": "no orange pustules"},
                                         {"candidate": "Apple rust", "support": "leaf lesions", "counterevidence": "no orange pustules"},
                                         {"candidate": "Powdery mildew", "support": "leaf damage", "counterevidence": "no white coating"}],
               "evidence": "visible lesion pattern", "uncertainty": "leaf underside unseen"}
    assert validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite) == []
    assert render_rewrite_reasoning_v2(rewrite).startswith("Visual observations:")
    rewrite["evidence"] = "classifier evidence"
    assert "direct_rewrite_mentions_tool" in validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite)
    rewrite["evidence"] = "visible lesion pattern"
    rewrite["candidate_comparisons"][2]["candidate"] = "Apple scab"
    assert "v2_comparisons_not_distinct" in validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite)
    rewrite["candidate_comparisons"][2]["candidate"] = "Powdery mildew"
    rewrite["visual_observations"][0] = "private audit says brown spots"
    assert "rewrite_private_leakage" in validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite)


def test_future_rewrite_v2_supports_chinese_option_and_enforces_character_limit() -> None:
    row = {"language": "zh", "question_type": "option", "public_options": [{"label": "B", "name": "Apple scab", "name_zh": "苹果黑星病"}]}
    parent = {"final": "B"}
    rewrite = {"final": "B", "visual_observations": ["叶片有褐色斑点", "斑点边缘粗糙", "未见橙色孢子堆"],
               "candidate_comparisons": [{"candidate": "苹果黑星病", "support": "褐色粗糙病斑", "counterevidence": "未见橙色孢子堆"},
                                         {"candidate": "苹果锈病", "support": "叶片有病斑", "counterevidence": "未见橙色孢子堆"},
                                         {"candidate": "白粉病", "support": "叶片受损", "counterevidence": "未见白色粉状物"}],
               "evidence": "可见病斑特征", "uncertainty": "叶片背面不可见"}
    assert validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite) == []
    assert _answer_for_student(row, "B") == "苹果黑星病 — B"
    rewrite["evidence"] = "可见" * 901
    assert "rewrite_length_exceeded" in validate_rewrite_v2(row=row, route="direct", parent=parent, rewrite=rewrite)


def test_future_rewrite_v3_attaches_image_and_rejects_nonvisual_fixed_final_claim(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"; image.write_bytes(b"not-a-real-image")
    row = {"image_path": str(image), "language": "en", "question": "Identify it", "question_type": "open", "public_options": []}
    parent = {"final": "Apple scab", "trace": []}
    rewrite = {"final": "Apple scab", "visual_observations": ["brown lesions", "rough edges", "green leaf"],
               "candidate_comparisons": [{"candidate": "Apple scab", "support": "rough lesions", "counterevidence": "no orange pustules"},
                                         {"candidate": "Apple rust", "support": "leaf lesions", "counterevidence": "no orange pustules"},
                                         {"candidate": "Powdery mildew", "support": "leaf damage", "counterevidence": "no white coating"}],
               "evidence": "visible lesion pattern", "uncertainty": "leaf underside unseen"}
    request = rewrite_prompt_v3(row=row, route="direct", parent=parent, registry_names=[])
    assert request["messages"][1]["content"][1]["type"] == "image_url"
    assert validate_rewrite_v3(row=row, route="direct", parent=parent, rewrite=rewrite) == []
    rewrite["evidence"] = "I cannot see the image; I used the fixed final."
    assert "v3_nonvisual_or_fixed_final_claim" in validate_rewrite_v3(row=row, route="direct", parent=parent, rewrite=rewrite)


def test_public_registry_must_match_the_frozen_sha(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({"name": "Apple scab", "name_zh": "苹果黑星病"}) + "\n", encoding="utf-8")
    from agrinet.rag.micu_classifier_hcv_v2 import sha256
    assert _public_registry_names(registry, expected_sha256=sha256(registry)) == [{"name": "Apple scab", "name_zh": "苹果黑星病"}]
    import pytest
    with pytest.raises(ValueError, match="matching frozen public registry"):
        _public_registry_names(registry, expected_sha256="0" * 64)


def test_public_registry_supports_canonical_release_field_names(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({"canonical_english_name": "Apple black rot", "canonical_chinese_name": "苹果黑腐病"}) + "\n", encoding="utf-8")
    from agrinet.rag.micu_classifier_hcv_v2 import sha256
    assert _public_registry_names(registry, expected_sha256=sha256(registry)) == [{"name": "Apple black rot", "name_zh": "苹果黑腐病"}]
