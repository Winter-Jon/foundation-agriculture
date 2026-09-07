"""Stable pytest classification without changing the existing flat test paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from agrinet.common.test_taxonomy import INTEGRATION_MODULES, UNIT_MODULES


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        module = Path(str(item.fspath)).name
        if module in UNIT_MODULES:
            item.add_marker(pytest.mark.unit)
        elif module in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.regression)
