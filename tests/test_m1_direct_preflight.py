import hashlib
import json
from pathlib import Path

from agrinet.research.m1.preflight import canonical_classes, similar_rows


def test_real_catalog_is_exactly_the_217_training_classes():
    classes = canonical_classes()
    assert len(classes) == 217
    assert len({row["code"] for row in classes}) == 217
    assert sum(row["task_domain"] == "disease" for row in classes) == 145
    assert sum(row["task_domain"] == "pest" for row in classes) == 72
    assert all(row["english_name"] and row["chinese_name"] for row in classes)


def test_joint_similarity_rows_are_same_domain_unique_and_complete():
    classes = canonical_classes()
    rows = similar_rows(classes)
    domains = {row["code"]: row["task_domain"] for row in classes}
    assert len(rows) == 217
    for row in rows:
        negatives = row["hard_negative_codes"]
        assert row["class_code"] not in negatives
        assert len(negatives) == len(set(negatives))
        assert all(domains[code] == domains[row["class_code"]] for code in negatives)
        names = [next(item["chinese_name"] for item in classes if item["code"] == code) for code in [row["class_code"], *negatives]]
        assert len(names) == len(set(names))
    assert all(len(row["hard_negative_codes"]) == 3 for row in rows)
    assert all(row["source"] == "wiki_siglip2_joint_text_image_top12_zh_unique" for row in rows)


def test_n05053_supplement_has_five_unique_licensed_occurrences_and_valid_hashes():
    path = Path("outputs/artifacts/datasets/m1-direct-current-hcv-v1/n05053_candidates/approved_manifest.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({row["occurrence_id"] for row in rows}) == 5
    assert all(row["class_code"] == "N05053" and row["license"].startswith("cc-") and row["visual_review"] for row in rows)
    assert all(hashlib.sha256(Path(row["query_image"]).read_bytes()).hexdigest() == row["image_sha256"] for row in rows)
