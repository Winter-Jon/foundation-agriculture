import json
from pathlib import Path

from scripts.vision.build_p6_holdout_manifests import select_groups


def test_select_groups_are_disjoint_balanced_and_deterministic() -> None:
    rows = []
    for domain, prefix in (("disease", "D"), ("pest", "P")):
        rows.extend({"canonical_class_code": f"{prefix}{index:03d}", "class_role": "known",
                     "domain": domain, "train_candidate_images": 100 + index}
                    for index in range(20))
    first = select_groups(rows, seed="fixed")
    second = select_groups(rows, seed="fixed")
    assert first == second
    assert [len(group) for group in first] == [8, 8, 8]
    assert len(set().union(*map(set, first))) == 24
    for group in first:
        assert sum(code.startswith("D") for code in group) == 4
        assert sum(code.startswith("P") for code in group) == 4
