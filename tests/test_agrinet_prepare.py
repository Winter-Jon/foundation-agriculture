import json
from pathlib import Path

from agrinet.data.agrinet import prepare_bounded_contrast, validate_bounded_contrast


def _write_wiki(path: Path, codes: list[str]) -> None:
    payload = {
        "description": {
            f"wiki::{code}": {
                "code": code,
                "english_name": f"Class {code}",
                "chinese_name": f"类别 {code}",
                "description": {"content_1": f"Evidence for {code}"},
            }
            for code in codes
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_bounded_prepare_is_deterministic_and_uses_real_images(tmp_path: Path) -> None:
    codes = [f"N04{i:03d}" for i in range(1, 5)] + [f"N05{i:03d}" for i in range(1, 5)]
    data_root = tmp_path / "data"
    for code in codes:
        class_dir = data_root / "all" / code
        class_dir.mkdir(parents=True)
        for index in range(3):
            (class_dir / f"{code}_{index}.jpg").write_bytes(b"image")
    wiki = tmp_path / "wiki.json"
    _write_wiki(wiki, codes)
    first, second = tmp_path / "first", tmp_path / "second"
    stats = prepare_bounded_contrast(data_root, wiki, first, class_count=8, seed=7)
    prepare_bounded_contrast(data_root, wiki, second, class_count=8, seed=7)
    assert stats == {"classes": 8, "pairs": 8, "samples": 8}
    assert (first / "samples.jsonl").read_bytes() == (second / "samples.jsonl").read_bytes()
    for raw in (first / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        assert Path(row["query_image"]).is_file()
        assert len(row["candidate_labels"]) == 4
        assert row["final_label"] in [candidate["code"] for candidate in row["candidate_labels"]]
        assert all(negative["code"][:3] == row["final_label"][:3] for negative in row["negative_reference_images"])
    assert validate_bounded_contrast(first) == stats
