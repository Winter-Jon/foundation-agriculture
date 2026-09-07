import json
from pathlib import Path

from agrinet.vlm.long_rag_preflight import build_long_tail


def test_long_tail_is_deterministic_and_preserves_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "x" * 40}]},
        {"messages": [{"role": "user", "content": "y" * 20}]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = tmp_path / "long.jsonl"
    report = build_long_tail(source, destination, 2)
    selected = [json.loads(line) for line in destination.read_text().splitlines()]
    assert selected == [
        {"messages": rows[1]["messages"]},
        {"messages": rows[2]["messages"]},
    ]
    assert report["rows"] == 2
    assert report["derived_view_only"] is True
    assert destination.with_suffix(".report.json").is_file()
