from typer.testing import CliRunner

from agrinet.cli.app import app
from agrinet.common.credentials import CredentialError

runner = CliRunner()


def test_root_help_lists_domains() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert all(domain in result.stdout for domain in ("data", "rag", "vlm"))


def test_domain_list_is_bilingual() -> None:
    result = runner.invoke(app, ["vlm", "list"])
    assert result.exit_code == 0
    assert "vlm-sft-rag-qwen3vl4b-full-v1" in result.stdout
    assert "基于已验证 RAG 轨迹" in result.stdout


def test_show_rejects_cross_domain_experiment() -> None:
    result = runner.invoke(app, ["rag", "show", "vlm-sft-rag-qwen3vl4b-full-v1"])
    assert result.exit_code == 2
    assert "belongs to vlm" in result.stderr


def test_data_submit_dry_run_is_deterministic() -> None:
    experiment_id = "data-generate-agrinet-vloom-teacher-v1"
    result = runner.invoke(app, ["data", "submit", experiment_id, "--dry-run"])
    assert result.exit_code == 0
    assert "-m agrinet.cli.app data generate" in result.stdout
    assert "sbatch" not in result.stdout


def test_local_vloom_pilot_is_registered() -> None:
    result = runner.invoke(app, ["data", "show", "data-generate-agrinet-vloom-local-pilot-v1"])
    assert result.exit_code == 0
    assert "runtime/local-data" in result.stdout
    assert "单样本本地 VLOOM 迁移试运行" in result.stdout


def test_hcv_dp8_queue_dry_run_carries_freeze_gate() -> None:
    result = runner.invoke(
        app, ["vlm", "submit", "vlm-sft-qwen3vl4b-hcv-manual-json-8gpu-v1", "--operation", "manual-json-checkpoint-queue", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "HCV_FREEZE_AUDIT=outputs/artifacts/datasets/agrinet-hcv-manual-json-v1/validation.json" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in result.stdout


def test_generate_preflight_happens_before_run_allocation(monkeypatch) -> None:
    monkeypatch.setattr(
        "agrinet.cli.data.yunwu_environment", lambda: (_ for _ in ()).throw(CredentialError("locked"))
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("agrinet.cli.data.run_foreground", forbidden)
    result = runner.invoke(
        app, ["data", "submit", "data-generate-agrinet-vloom-local-pilot-v1", "--operation", "generate"]
    )
    assert result.exit_code == 1
    assert "preflight failed" in result.stderr
    assert called is False
