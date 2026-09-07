import importlib.util
from pathlib import Path

import pytest


def load_plugin(monkeypatch):
    monkeypatch.setenv("AGRINET_WEIGHTED_LIGER_PLUGIN_NO_INSTALL", "1")
    path = Path("src/agrinet/vlm/weighted_liger_plugin.py")
    spec = importlib.util.spec_from_file_location("weighted_liger_plugin_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_guard_rejects_unreviewed_runtime(monkeypatch):
    plugin = load_plugin(monkeypatch)
    monkeypatch.setattr(plugin, "version", lambda name: "0")
    with pytest.raises(RuntimeError, match="supports only"):
        plugin._require_versions()


def test_plugin_declares_unreduced_liger_contract(monkeypatch):
    plugin = load_plugin(monkeypatch)
    assert "unreduced" in plugin.liger_per_token_loss.__doc__
