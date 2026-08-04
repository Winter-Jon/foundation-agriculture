from pathlib import Path

from agrinet.data.convert import convert_teacher_records
from agrinet.data.prepare import prepare_records
from agrinet.data.validate import validate_records

FIXTURES = Path(__file__).parent / "fixtures" / "data"


def test_prepare_validate_convert_fixture(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.jsonl"
    sft = tmp_path / "student.jsonl"
    assert prepare_records(FIXTURES / "source.jsonl", prepared) == 1
    assert validate_records(prepared, "prepared") == 1
    assert validate_records(FIXTURES / "teacher.jsonl", "teacher") == 1
    assert convert_teacher_records(FIXTURES / "teacher.jsonl", sft, "teacher-fixture-v1") == 1
    assert validate_records(sft, "sft") == 1
