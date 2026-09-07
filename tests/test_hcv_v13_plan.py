from agrinet.research.hcv.v13_plan import build_plan


def _rows():
    rows = []
    for question_type in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                for index in range(5):
                    rows.append({
                        "sample_id": f"{question_type}-{language}-{domain}-{index}",
                        "query_image": f"/images/{question_type}-{language}-{domain}-{index}.jpg",
                        "image_sha256": f"hash-{question_type}-{language}-{domain}-{index}",
                        "question_type": question_type, "language": language, "task_domain": domain,
                        "audit_truth_code": "N00001", "audit_truth_name": "Leaf spot",
                        "audit_truth_name_zh": "叶斑病", "audit_correct_option": "A",
                    })
    return rows


def test_plan_is_exactly_32_balanced_isolated_and_public_private_separated():
    public, private, report = build_plan(_rows())
    assert report["ready"]
    assert len(public) == len(private) == 32
    assert all(count == 4 for count in report["per_cell"].values())
    assert all("audit_truth_code" not in row for row in public)
    assert all(row["audit_truth_code"] == "N00001" for row in private)


def test_plan_excludes_prior_hash_and_reports_shortage():
    rows = _rows()
    excluded = {"hash-open-en-disease-0", "hash-open-en-disease-1"}
    _public, _private, report = build_plan(rows, excluded_hashes=excluded)
    assert report["shortages"] == {"open/en/disease": 1}
    assert not report["ready"]
