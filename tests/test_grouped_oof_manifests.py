import json
import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "build_grouped_oof_manifests",
    Path("scripts/vision/build_grouped_oof_manifests.py"),
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_fold_assignment_is_deterministic() -> None:
    assert _MODULE.fold_for("phash:abc", 3) == _MODULE.fold_for("phash:abc", 3)


def test_materialized_oof_summary_is_known_only_and_group_closed() -> None:
    summary = json.loads(Path("outputs/artifacts/vision/openagri-v3-known-vitl-oof-v2/manifests-attempt3/summary.json").read_text())
    assert summary["known_classes"] == 107
    assert summary["rows"] == 142726
    assert summary["fold_group_rule"] == "near_duplicate_group_id"
    assert summary["training_eligible"] is False

    groups: dict[str, set[int]] = {}
    root = Path("outputs/artifacts/vision/openagri-v3-known-vitl-oof-v2/manifests-attempt3")
    for fold in range(3):
        for line in (root / f"fold-{fold}/heldout.jsonl").read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row["near_duplicate_group_id"], set()).add(fold)
    assert all(len(folds) == 1 for folds in groups.values())
