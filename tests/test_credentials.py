from pathlib import Path

import pytest

from agrinet.common.credentials import CredentialError, yunwu_environment


def _helper(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_parse_encrypted_helper_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YUNWU_API_KEY", raising=False)
    helper = _helper(
        tmp_path / "apikey",
        "printf \"export YUNWU_API_BASE_URL='https://example.test/v1'\\nexport YUNWU_API_KEY='secret-value'\\n\"\n",
    )
    values = yunwu_environment(helper)
    assert values["YUNWU_API_BASE_URL"] == "https://example.test/v1"
    assert values["YUNWU_API_KEY"] == "secret-value"


def test_reject_helper_shell_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YUNWU_API_KEY", raising=False)
    helper = _helper(tmp_path / "apikey", "printf 'rm -rf /\n'\n")
    with pytest.raises(CredentialError, match="unsupported command"):
        yunwu_environment(helper)
