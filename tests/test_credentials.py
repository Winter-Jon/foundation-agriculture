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


def test_e2_micu_runtime_requires_validated_slb_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from agrinet.cli.rag import MICU_SLB_BASE_URL, _micu_runtime_environment

    monkeypatch.setattr("agrinet.cli.rag.yunwu_environment", lambda **_kwargs: {
        "YUNWU_API_KEY": "test-key", "YUNWU_API_BASE_URL": "https://www.micuapi.ai/v1"})
    monkeypatch.setattr("agrinet.cli.rag.local_proxy_environment", lambda: {})
    environment = _micu_runtime_environment({"teacher_base_url": MICU_SLB_BASE_URL}, dry_run=False)
    assert environment["YUNWU_API_BASE_URL"] == MICU_SLB_BASE_URL
    with pytest.raises(ValueError, match="validated SLB"):
        _micu_runtime_environment({"teacher_base_url": "https://www.micuapi.ai/v1"}, dry_run=False)


def test_slb_canary_stops_after_a_failed_first_round(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import run_canary

    response = {"choices": [{"message": {"content": "READY"}}]}
    calls = []
    def invoke(_payload, *, timeout):
        calls.append(timeout)
        if len(calls) == 2:
            raise RuntimeError("provider unavailable")
        return response
    report = run_canary(output_root=tmp_path / "canary", model="gpt-5.6-terra", rounds=2, requests_per_round=10, timeout=1, invoke=invoke)
    assert report["completed_rounds"] == 1
    assert report["generation_ready_for_e2_retry_v2"] is False
    assert len(calls) == 10
    assert not (tmp_path / "canary/round-2").exists()


def test_slb_canary_retains_requested_model_in_report(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import run_canary

    response = {"choices": [{"message": {"content": "READY"}}]}
    report = run_canary(output_root=tmp_path / "canary-sol", model="gpt-5.6-sol",
                         rounds=2, requests_per_round=10, timeout=1, invoke=lambda *_args, **_kwargs: response)
    assert report["model"] == "gpt-5.6-sol"
    assert report["generation_ready_for_e2_retry_v2"] is True
