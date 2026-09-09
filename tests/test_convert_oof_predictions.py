import importlib.util
from pathlib import Path


def test_converter_has_explicit_oof_contract_fields() -> None:
    spec = importlib.util.spec_from_file_location(
        "convert_oof_predictions", Path("scripts/vision/convert_oof_predictions.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.sha256(Path("scripts/vision/convert_oof_predictions.py"))
    assert module.REGISTRY_SHA.startswith("4fa642")
