from pathlib import Path

from agrinet.research.hcv.v13_source import build_source, sha256_file


def _classes():
    return [
        {"code": "N04001", "task_domain": "disease", "english_name": "Spot", "chinese_name": "斑点"},
        {"code": "N04002", "task_domain": "disease", "english_name": "Rust", "chinese_name": "锈病"},
        {"code": "N04003", "task_domain": "disease", "english_name": "Blight", "chinese_name": "枯萎病"},
        {"code": "N04004", "task_domain": "disease", "english_name": "Mildew", "chinese_name": "霉病"},
        {"code": "N05001", "task_domain": "pest", "english_name": "Aphid", "chinese_name": "蚜虫"},
        {"code": "N05002", "task_domain": "pest", "english_name": "Moth", "chinese_name": "飞蛾"},
        {"code": "N05003", "task_domain": "pest", "english_name": "Beetle", "chinese_name": "甲虫"},
        {"code": "N05004", "task_domain": "pest", "english_name": "Thrips", "chinese_name": "蓟马"},
    ]


def test_source_preparation_balances_cells_hashes_file_bytes_and_separates_truth(tmp_path: Path):
    root = tmp_path / "all"
    for class_index, row in enumerate(_classes()):
        folder = root / row["code"]; folder.mkdir(parents=True)
        for image_index in range(5):
            (folder / f"{image_index}.jpg").write_bytes(f"{class_index}:{image_index}".encode())
    rows, report = build_source(_classes(), excluded_hashes=set(), images_root=root)
    assert report["ready"] and len(rows) == 32
    assert all(len(row["image_sha256"]) == 64 for row in rows)
    assert all("audit_truth_code" in row for row in rows)
    assert sum(row["question_type"] == "option" for row in rows) == 16
    assert all(len(row["public_option_labels"]) == 4 for row in rows if row["question_type"] == "option")


def test_source_preparation_never_selects_an_excluded_content_hash(tmp_path: Path):
    root = tmp_path / "all"
    for class_index, row in enumerate(_classes()):
        folder = root / row["code"]; folder.mkdir(parents=True)
        for image_index in range(5):
            (folder / f"{image_index}.jpg").write_bytes(f"{class_index}:{image_index}".encode())
    excluded = sha256_file(root / "N04001" / "0.jpg")
    rows, report = build_source(_classes(), excluded_hashes={excluded}, images_root=root)
    assert report["ready"]
    assert excluded not in {row["image_sha256"] for row in rows}
