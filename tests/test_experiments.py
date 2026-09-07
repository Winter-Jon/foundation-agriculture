from datetime import datetime

from agrinet.common.experiments import make_run_id, validate_run_id


def test_run_id_is_deterministic() -> None:
    run_id = make_run_id(attempt=2, now=datetime(2026, 8, 3, 12, 34, 56), sha="b0b031b2")
    assert run_id == "20260803T123456-b0b031b2-a02"
    validate_run_id(run_id)
