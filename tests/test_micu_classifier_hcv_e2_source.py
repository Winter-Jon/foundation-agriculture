from scripts.rag.build_micu_classifier_hcv_e2_source import assign_targeted


def test_target_assignment_reserves_half_for_p6_holdout() -> None:
    rows = []
    for index in range(30):
        holdout = index >= 25
        pattern = f"P{index // 5 + 1}" if not holdout else "P6"
        rows.append({
            "sample_id": f"s{index}",
            "private": {"candidate_pattern": pattern, "truth_code": "T",
                        "p6_group": 0 if holdout else None},
            "prediction": {"top5": [{"code": "T", "score": 0.9}]},
        })
    qwen = {row["sample_id"]: {"sample_id": row["sample_id"], "output_preview": "truth"}
            for row in rows}
    rag = {row["sample_id"]: {"sample_id": row["sample_id"], "status": "success",
                                      "raw_response": {"evidence": []}} for row in rows}
    selected = assign_targeted(rows, qwen, rag, {
        "T": {"canonical_english_name": "truth", "canonical_chinese_name": "真值"}})
    assert {pattern for _, pattern in selected} == {f"P{i}" for i in range(1, 11)}
    for row, pattern in selected:
        assert (row["private"]["p6_group"] is not None) == (pattern in {f"P{i}" for i in range(6, 11)})
