import importlib.util
from pathlib import Path


def test_report_script_imports() -> None:
    spec = importlib.util.spec_from_file_location(
        "summarize_micu_classifier_hcv_v2", Path("scripts/report/summarize_micu_classifier_hcv_v2.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
