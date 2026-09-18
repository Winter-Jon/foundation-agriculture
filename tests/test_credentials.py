import json
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


def test_parse_micu_main_helper_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YUNWU_API_KEY", raising=False)
    helper = _helper(
        tmp_path / "apikey",
        "printf \"export MICU_MAIN_API_BASE_URL='https://www.micuapi.ai/v1'\\nexport MICU_MAIN_API_KEY='secret-value'\\n\"\n",
    )
    values = yunwu_environment(helper, profile="micu_main")
    assert values == {"YUNWU_API_BASE_URL": "https://www.micuapi.ai/v1", "YUNWU_API_KEY": "secret-value"}


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
    assert {"127.0.0.1", "localhost", "::1"}.issubset(set(environment["NO_PROXY"].split(",")))
    assert environment["no_proxy"] == environment["NO_PROXY"]
    with pytest.raises(ValueError, match="allowed endpoint"):
        _micu_runtime_environment({"teacher_base_url": "https://www.micuapi.ai/v1"}, dry_run=False)


def test_micu_direct_node_is_canary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from agrinet.cli.rag import MICU_DIRECT_BASE_URL, _micu_runtime_environment

    calls = []
    def load_profile(*, profile):
        calls.append(profile)
        return {"YUNWU_API_KEY": "test-key", "YUNWU_API_BASE_URL": "https://api-slb.micuapi.ai/v1"}
    monkeypatch.setattr("agrinet.cli.rag.yunwu_environment", load_profile)
    monkeypatch.setattr("agrinet.cli.rag.local_proxy_environment", lambda: {})
    environment = _micu_runtime_environment({"teacher_base_url": MICU_DIRECT_BASE_URL},
                                            dry_run=False, allow_direct_node=True)
    assert environment["YUNWU_API_BASE_URL"] == MICU_DIRECT_BASE_URL
    assert calls == ["micu_main"]
    with pytest.raises(ValueError, match="allowed endpoint"):
        _micu_runtime_environment({"teacher_base_url": MICU_DIRECT_BASE_URL}, dry_run=False)


def test_micu_runtime_omits_unreachable_loopback_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from agrinet.cli.rag import _micu_runtime_environment

    monkeypatch.setattr("agrinet.cli.rag.yunwu_environment", lambda **_kwargs: {
        "YUNWU_API_KEY": "test-key", "YUNWU_API_BASE_URL": "https://api-slb.micuapi.ai/v1"})
    monkeypatch.setattr("agrinet.cli.rag.local_proxy_environment", lambda: {"ALL_PROXY": "http://127.0.0.1:7899"})
    monkeypatch.setattr("agrinet.cli.rag.socket.create_connection", lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionRefusedError()))
    environment = _micu_runtime_environment({"teacher_base_url": "https://api-slb.micuapi.ai/v1"}, dry_run=False)
    assert "ALL_PROXY" not in environment


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


def test_slb_canary_records_only_safe_failure_type(tmp_path: Path) -> None:
    from urllib.error import HTTPError

    from agrinet.rag.micu_slb_canary import run_canary

    def denied(_payload, *, timeout):
        raise HTTPError("https://provider.example/v1/chat/completions?secret=do-not-record", 403,
                        "Forbidden", hdrs=None, fp=None)

    report = run_canary(output_root=tmp_path / "canary-denied", model="gpt-5.6-sol",
                         rounds=2, requests_per_round=10, timeout=1, invoke=denied)
    summary = json.loads((tmp_path / "canary-denied/round-1/summary.json").read_text())
    assert report["completed_rounds"] == 1
    assert all(row["reason"] == "DeliveryUnresolved" for row in summary["rows"])
    assert all(row["failure_type"] == "HTTPError" for row in summary["rows"])
    assert "provider.example" not in (tmp_path / "canary-denied/round-1/summary.json").read_text()
    assert "do-not-record" not in (tmp_path / "canary-denied/round-1/summary.json").read_text()


def test_direct_canary_denied_preflight_creates_no_chat_intent_or_output(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import DIRECT_BASE_URL, run_canary

    calls = []
    with pytest.raises(RuntimeError, match="HTTP 403"):
        run_canary(
            output_root=tmp_path / "direct-denied", endpoint=DIRECT_BASE_URL, model="gpt-5.6-sol",
            rounds=2, requests_per_round=10, timeout=1,
            invoke=lambda *_args, **_kwargs: calls.append("chat"),
            direct_preflight=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("HTTP 403")),
        )
    assert calls == []
    assert not (tmp_path / "direct-denied").exists()


def test_direct_canary_runs_only_after_model_preflight(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import DIRECT_BASE_URL, run_canary

    checked, calls = [], []
    response = {"choices": [{"message": {"content": "READY"}}]}
    report = run_canary(
        output_root=tmp_path / "direct-ready", endpoint=DIRECT_BASE_URL, model="gpt-5.6-sol",
        rounds=2, requests_per_round=10, timeout=1,
        direct_preflight=lambda **kwargs: checked.append(kwargs),
        invoke=lambda *_args, **_kwargs: (calls.append("chat"), response)[1],
    )
    assert checked == [{"endpoint": DIRECT_BASE_URL, "model": "gpt-5.6-sol", "timeout": 1}]
    assert len(calls) == 20
    assert report["generation_ready_for_e2_retry_v2"] is True


def test_slb_recovery_canary_uses_new_predecessor_bound_r1_request(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import run_recovery_canary

    response = {"choices": [{"message": {"content": "READY"}}]}
    calls = {}
    def invoke(_payload, *, timeout):
        probe = len(calls) + 1
        calls[probe] = True
        if probe == 1:
            raise TimeoutError("unconfirmed")
        return response

    report = run_recovery_canary(output_root=tmp_path / "recovery", model="gpt-5.6-sol",
                                 probes=10, timeout=1, invoke=invoke)
    r0, r1, r2 = report["attempts"]["R0"], report["attempts"]["R1"], report["attempts"]["R2"]
    assert len(r0) == 10 and len(r1) == 1 and r2 == []
    assert r1[0]["probe"] == 1 and r1[0]["predecessor_request_id"] == r0[0]["request_id"]
    assert r1[0]["request_id"] != r0[0]["request_id"]
    assert report["generation_ready_for_full_classifier"] is True


def test_slb_recovery_canary_stops_after_terminal_r2_unknown_delivery(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import run_recovery_canary

    report = run_recovery_canary(
        output_root=tmp_path / "recovery-fail", model="gpt-5.6-sol", probes=10, timeout=1,
        invoke=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("unconfirmed")),
    )
    assert [len(report["attempts"][attempt]) for attempt in ("R0", "R1", "R2")] == [10, 10, 10]
    assert report["generation_ready_for_full_classifier"] is False
    for index in range(10):
        assert report["attempts"]["R1"][index]["predecessor_request_id"] == report["attempts"]["R0"][index]["request_id"]
        assert report["attempts"]["R2"][index]["predecessor_request_id"] == report["attempts"]["R1"][index]["request_id"]


def test_slb_canary_retains_requested_model_in_report(tmp_path: Path) -> None:
    from agrinet.rag.micu_slb_canary import run_canary

    response = {"choices": [{"message": {"content": "READY"}}]}
    report = run_canary(output_root=tmp_path / "canary-sol", model="gpt-5.6-sol",
                         rounds=2, requests_per_round=10, timeout=1, invoke=lambda *_args, **_kwargs: response)
    assert report["model"] == "gpt-5.6-sol"
    assert report["generation_ready_for_e2_retry_v2"] is True
