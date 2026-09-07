import os
from types import SimpleNamespace

import pytest

from agrinet.common.network import NetworkConfigError, local_proxy_environment


def _clear_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(key, raising=False)


def test_existing_proxy_environment_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_proxy(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1234")
    assert local_proxy_environment() == {"HTTPS_PROXY": "http://127.0.0.1:1234"}


def test_proxy_helper_is_strictly_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_proxy(monkeypatch)
    monkeypatch.setattr(
        "agrinet.common.network.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="http://127.0.0.1:7890\n"),
    )
    assert local_proxy_environment() == {
        "ALL_PROXY": "http://127.0.0.1:7890",
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        "HTTP_PROXY": "http://127.0.0.1:7890",
    }


def test_proxy_helper_rejects_shell_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_proxy(monkeypatch)
    monkeypatch.setattr(
        "agrinet.common.network.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="rm -rf /\n"),
    )
    with pytest.raises(NetworkConfigError, match="invalid proxy URL"):
        local_proxy_environment()
