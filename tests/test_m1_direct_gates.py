from pathlib import Path

from agrinet.data.m1_direct_gates import (
    CELLS, OPEN_PEST_CELLS, load_gates, validate_open_pest_distinguishability_screen, validate_open_pest_preflight,
    validate_sample_preflight, validate_stratified_distinguishability_screen, validate_stratified_pilot,
)

GATES = load_gates(Path("configs/sampling/m1-direct-acceptance-gates-v1.yaml"))
REJECTION_GATES = load_gates(Path("configs/sampling/m1-direct-acceptance-gates-v2.yaml"))
OPEN_PEST_GATES = load_gates(Path("configs/sampling/m1-direct-open-pest-preflight-gates-v1.yaml"))
SCREEN_GATES = load_gates(Path("configs/sampling/m1-direct-open-pest-distinguishability-screen-gates-v1.yaml"))
STRATIFIED_SCREEN_GATES = load_gates(Path("configs/sampling/m1-direct-stratified-distinguishability-screen-gates-v1.yaml"))
STRATIFIED_SCREEN_REJECTION_GATES = load_gates(Path("configs/sampling/m1-direct-stratified-distinguishability-screen-gates-v3.yaml"))


def row(sample_id, cell, *, accepted=True):
    return {"sample_id": sample_id, "question_type": cell[0], "language": cell[1], "task_domain": cell[2], "accepted": accepted, "teacher_correct": True, "auditor_correct": True, "reasoning_checks_passed": True, "delivery_status": "known", "errors": []}


def test_sample_gate_requires_all_eight_attempts_and_both_domains():
    rows = [row(f"s-{i}", (q, l, d)) for i, (q, l, d) in enumerate((
        (q, l, d) for d in ("disease", "pest") for q in ("open", "option") for l in ("en", "zh")
    ))]
    assert validate_sample_preflight(rows, GATES["sample_preflight"])["passed"]
    assert not validate_sample_preflight(rows[:-1], GATES["sample_preflight"])["passed"]


def test_sample_gate_rejects_unknown_delivery_even_if_accepted():
    rows = [row(f"s-{i}", cell) for i, cell in enumerate(CELLS)]
    rows[0]["delivery_status"] = "unknown"
    report = validate_sample_preflight(rows, GATES["sample_preflight"])
    assert not report["passed"] and "unknown_delivery" in report["hard_errors"]


def pilot_rows():
    return [row(f"p-{cell_index}-{i}", cell) for cell_index, cell in enumerate(CELLS) for i in range(8)]


def test_pilot_gate_requires_fixed_eight_cell_denominators():
    rows = pilot_rows()
    assert validate_stratified_pilot(rows, GATES["stratified_pilot"])["passed"]
    assert not validate_stratified_pilot(rows[:-1], GATES["stratified_pilot"])["passed"]


def test_pilot_gate_allows_one_quality_rejection_but_enforces_overall_rate_and_hard_errors():
    rows = pilot_rows()
    rows[0]["accepted"] = False
    assert validate_stratified_pilot(rows, GATES["stratified_pilot"])["passed"]
    rows[0]["errors"] = ["label_leakage"]
    assert not validate_stratified_pilot(rows, GATES["stratified_pilot"])["passed"]


def test_v2_pilot_rejects_an_unknown_row_but_accepts_a_cohort_above_90_percent():
    rows = pilot_rows()
    rows[0]["delivery_status"] = "unknown"
    report = validate_stratified_pilot(rows, REJECTION_GATES["stratified_pilot"])
    assert report["accepted"] == 63
    assert report["rejection_reasons"] == {"unknown_delivery": 1}
    assert report["checks"]["zero_tolerance"]
    assert report["passed"]
    # Fixed 64-row cohort: 58 / 64 is 90.625% and clears "strictly >90%".
    rows = pilot_rows()
    for index in (0, 8, 16, 24, 32, 40):
        rows[index]["accepted"] = False
    report = validate_stratified_pilot(rows, REJECTION_GATES["stratified_pilot"])
    assert report["accepted"] == 58
    assert report["passed"]
    # 57 / 64 is below 90%, regardless of the per-cell minima.
    rows[48]["accepted"] = False
    assert not validate_stratified_pilot(rows, REJECTION_GATES["stratified_pilot"])["passed"]
    rows = pilot_rows()
    for index in range(0, len(rows), 8):
        rows[index]["accepted"] = False
    assert not validate_stratified_pilot(rows, GATES["stratified_pilot"])["passed"]


def test_open_pest_preflight_is_strict_and_cannot_accept_other_cells():
    rows = [row(f"op-{cell_index}-{index}", cell) for cell_index, cell in enumerate(OPEN_PEST_CELLS) for index in range(8)]
    assert validate_open_pest_preflight(rows, OPEN_PEST_GATES["open_pest_targeted_preflight"])["passed"]
    rows[-1]["accepted"] = False
    assert not validate_open_pest_preflight(rows, OPEN_PEST_GATES["open_pest_targeted_preflight"])["passed"]


def test_open_pest_screen_requires_all_declared_candidates_before_promotion():
    gate = SCREEN_GATES["open_pest_distinguishability_screen"]
    rows = []
    for question_type, language, domain in OPEN_PEST_CELLS:
        for index in range(16):
            rows.append({
                "sample_id": f"{language}-{index}", "question_type": question_type,
                "language": language, "task_domain": domain, "passed": index < 8,
                "delivery_status": "known", "errors": [],
            })
    report = validate_open_pest_distinguishability_screen(rows, gate)
    assert report["passed"] and len(report["passed_rows"]) == 16
    rows[0]["errors"] = ["protocol_error"]
    assert not validate_open_pest_distinguishability_screen(rows, gate)["passed"]
    rows = [row(f"bad-{index}", ("option", "en", "pest")) for index in range(16)]
    assert not validate_open_pest_preflight(rows, OPEN_PEST_GATES["open_pest_targeted_preflight"])["passed"]


def test_stratified_screen_requires_passable_images_in_all_eight_cells():
    gate = STRATIFIED_SCREEN_GATES["stratified_distinguishability_screen"]
    rows = []
    for cell_index, cell in enumerate(CELLS):
        for index in range(16):
            rows.append({
                "sample_id": f"screen-{cell_index}-{index}", "question_type": cell[0],
                "language": cell[1], "task_domain": cell[2], "passed": index < 8,
                "delivery_status": "known", "errors": [],
            })
    report = validate_stratified_distinguishability_screen(rows, gate)
    assert report["passed"] and len(report["passed_rows"]) == 64
    rows[-1]["errors"] = ["unknown_delivery"]
    assert not validate_stratified_distinguishability_screen(rows, gate)["passed"]


def test_v3_screen_rejects_unknown_rows_without_vetoing_a_cohort_above_90_percent():
    gate = STRATIFIED_SCREEN_REJECTION_GATES["stratified_distinguishability_screen"]
    rows = []
    for cell_index, cell in enumerate(CELLS):
        for index in range(32):
            rows.append({
                "sample_id": f"screen-v3-{cell_index}-{index}",
                "question_type": cell[0], "language": cell[1], "task_domain": cell[2],
                "passed": True, "delivery_status": "known", "errors": [],
            })
    rows[0]["delivery_status"] = "unknown"
    report = validate_stratified_distinguishability_screen(rows, gate)
    assert report["known_terminal"] == 255
    assert report["rejection_reasons"] == {"unknown_delivery": 1}
    assert report["passed"]
