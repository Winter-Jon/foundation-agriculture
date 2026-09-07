from agrinet.research.hcv.v13_calibration import bootstrap_calibration, build_calibration, direct_permitted


def _rows(correct_count: int, total: int = 120):
    return [
        {
            "confidence": "high", "correct": index < correct_count,
            "question_type": "open", "language": "en", "task_domain": "disease",
        }
        for index in range(total)
    ]


def test_direct_requires_all_conservative_calibration_groups():
    calibration = build_calibration(_rows(120), minimum_rows=100, lower_bound_floor=0.95)
    assert direct_permitted(calibration, question_type="open", language="en", task_domain="disease")
    assert not direct_permitted(calibration, question_type="option", language="en", task_domain="disease")


def test_imperfect_high_confidence_group_fails_95_percent_lower_bound():
    calibration = build_calibration(_rows(115), minimum_rows=100, lower_bound_floor=0.95)
    assert not calibration["groups"]["global"]["direct_permitted"]


def test_bootstrap_calibration_is_explicit_and_fails_closed_for_every_direct_route():
    calibration = bootstrap_calibration()
    assert calibration["mode"] == "bootstrap_force_rag"
    assert calibration["groups"] == {}
    assert not direct_permitted(calibration, question_type="open", language="en", task_domain="disease")
