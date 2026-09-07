import importlib.util
from pathlib import Path


_path = Path(__file__).parents[1] / "src/agrinet/research/hcv/promotion_gate.py"
_spec = importlib.util.spec_from_file_location("hcv_promotion_gate", _path)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate = _module.validate


def _review(lower: float, groups: bool = False) -> dict:
    value = {"paired_delta_pp": lower + 1.0, "paired_bootstrap_95pct_ci_pp": [lower, lower + 2.0]}
    if groups:
        value["groups"] = {name: {"delta_pp": -2.0} for name in _module.MAJOR_DIRECT_GROUPS}
    return value


def test_hcv_promotion_gate_requires_direct_and_rag_evidence() -> None:
    report = validate(_review(-2.0, groups=True), _review(0.1), direct_overall_floor=-3.0, direct_group_floor=-8.0, rag_lower_floor=0.0)
    assert report["promotion_authorized"] is True

    rejected = validate(_review(-3.1, groups=True), _review(0.1), direct_overall_floor=-3.0, direct_group_floor=-8.0, rag_lower_floor=0.0)
    assert rejected["promotion_authorized"] is False
    assert rejected["checks"]["direct_overall_noninferiority"] is False


def test_hcv_promotion_gate_rejects_subgroup_or_nonpositive_rag() -> None:
    direct = _review(-2.0, groups=True)
    direct["groups"]["question_type=option"]["delta_pp"] = -8.1
    report = validate(direct, _review(0.0), direct_overall_floor=-3.0, direct_group_floor=-8.0, rag_lower_floor=0.0)
    assert report["promotion_authorized"] is False
    assert report["checks"]["direct_major_subgroups"] is False
    assert report["checks"]["rag_positive_paired_effect"] is False
