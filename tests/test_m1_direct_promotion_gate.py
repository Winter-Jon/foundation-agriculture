import importlib.util
from pathlib import Path


_path = Path(__file__).parents[1] / "src/agrinet/research/m1/promotion_gate.py"
_spec = importlib.util.spec_from_file_location("m1_direct_promotion_gate", _path)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def review(delta: float, lower: float, groups: bool = True) -> dict:
    result = {"paired_delta_pp": delta, "paired_bootstrap_95pct_ci_pp": [lower, 2.0]}
    if groups:
        result["groups"] = {name: {"delta_pp": delta} for name in gate.REQUIRED_GROUPS}
    return result


def test_direct_promotion_requires_raw_base_and_m1_evidence() -> None:
    report = gate.validate(review(0.0, -3.0), review(-1.0, -2.0, groups=False), overall_floor=0.0, bootstrap_floor=-3.0, group_floor=-3.0)
    assert report["promotion_authorized"] is True


def test_direct_promotion_rejects_unknown_or_subgroup_regression() -> None:
    raw = review(0.5, -2.0)
    raw["groups"]["known_bucket=unknown"]["delta_pp"] = -3.1
    report = gate.validate(raw, review(1.0, -1.0, groups=False), overall_floor=0.0, bootstrap_floor=-3.0, group_floor=-3.0)
    assert report["promotion_authorized"] is False
    assert report["checks"]["raw_base_major_subgroups"] is False
