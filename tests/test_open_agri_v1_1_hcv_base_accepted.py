import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/ingest_open_agri_v2_accepted.py"
SPEC = importlib.util.spec_from_file_location("open_agri_v2_accepted", SCRIPT)
assert SPEC and SPEC.loader
ingest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ingest)


def test_supervision_fingerprint_normalizes_whitespace() -> None:
    first = {"images": ["images/train_candidate/disease/N04001/a.jpg"], "messages": [{"role": "user", "content": "  identify  this\nimage "}], "metadata": {"route": "direct", "language": "en", "question_type": "open"}}
    second = {"images": ["images/train_candidate/disease/N04001/a.jpg"], "messages": [{"role": "user", "content": "identify this image"}], "metadata": {"route": "direct", "language": "en", "question_type": "open"}}
    assert ingest.fingerprint(first) == ingest.fingerprint(second)


def test_ms_swift_agent_storage_contract_requires_json_strings() -> None:
    row = {
        "tools": "[]",
        "messages": [
            {"role": "user", "content": "<image>\nidentify"},
            {"role": "tool_call", "content": '{"name":"lookup","arguments":{}}'},
            {"role": "tool_response", "content": '{"status":"success"}'},
            {"role": "assistant", "content": "answer"},
        ],
    }
    ingest.validate_ms_swift_agent_record(row)
    row["messages"][1]["content"] = "not JSON"
    try:
        ingest.validate_ms_swift_agent_record(row)
    except ValueError as exc:
        assert "must be valid JSON" in str(exc)
    else:  # pragma: no cover - makes the rejection explicit
        raise AssertionError("invalid tool-call payload was accepted")


def test_multi_source_import_records_priority_and_deduplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "open_agri_v2"
    candidates = root / "vlm_data/candidates"
    candidates.mkdir(parents=True)
    image_a = "images/train_candidate/disease/N04001/a.jpg"
    image_b = "images/train_candidate/pest/N05001/b.jpg"
    pool = [
        {"image_path": image_a, "sft_eligible": True},
        {"image_path": image_b, "sft_eligible": True},
    ]
    (candidates / "image_pool.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pool), encoding="utf-8")

    def record(sample_id: str, image: str, task_domain: str) -> dict:
        return {
            "sample_id": sample_id, "images": [image], "tools": "[]",
            "messages": [{"role": "user", "content": "<image> identify"}, {"role": "assistant", "content": "answer"}],
            "metadata": {"route": "direct", "language": "en", "question_type": "open", "task_domain": task_domain},
        }

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(record("first", image_a, "disease")) + "\n", encoding="utf-8")
    second.write_text("".join(json.dumps(row) + "\n" for row in (record("duplicate", image_a, "disease"), record("second", image_b, "pest"))), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["ingest", "--dataset-root", str(root), "--source", str(first), "--source", str(second)])
    assert ingest.main() == 0

    train_root = root / "vlm_data/accepted/train"
    summary = json.loads((root / "vlm_data/audits/accepted_train_import_summary.json").read_text())
    rejected = [json.loads(line) for line in (root / "vlm_data/audits/accepted_train_dedup.jsonl").read_text().splitlines()]
    assert summary["accepted"] == 2 and summary["rejected"] == 1
    assert [source["accepted"] for source in summary["sources"]] == [1, 1]
    views = json.loads((train_root / "summary.json").read_text())
    assert views["total_rows"] == 2
    assert views["views"]["disease_direct"]["rows"] == 1
    assert views["views"]["pest_direct"]["rows"] == 1
    assert views["views"]["disease_rag"]["rows"] == 0
    assert views["views"]["pest_rag"]["rows"] == 0
    assert rejected[0]["sample_id"] == "duplicate"
    assert rejected[0]["reason"] == "duplicate_supervision_fingerprint"
