from __future__ import annotations

from pathlib import Path

from agrinet.common.test_taxonomy import INTEGRATION_MODULES, UNIT_MODULES


def test_test_taxonomy_covers_each_module_once() -> None:
    modules = {path.name for path in Path(__file__).parent.glob("test_*.py")}
    assert not (UNIT_MODULES & INTEGRATION_MODULES)
    assert UNIT_MODULES | INTEGRATION_MODULES <= modules
    assert modules - UNIT_MODULES - INTEGRATION_MODULES
