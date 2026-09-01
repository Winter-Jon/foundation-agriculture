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


def test_m1_direct_v8_pilot_operations_are_registered() -> None:
    result = runner.invoke(
        app, [
            "data", "submit", "data-m1-direct-current-hcv-v1",
            "--operation", "m1-direct-pilot-v8-collect", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "m1-direct-pilot-v8-collect" in result.stdout


def test_m1_direct_v9_pilot_operations_are_registered() -> None:
    result = runner.invoke(
        app, [
            "data", "submit", "data-m1-direct-current-hcv-v1",
            "--operation", "m1-direct-pilot-v9-collect", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "m1-direct-pilot-v9-collect" in result.stdout


def test_m1_direct_v8_screen_operations_are_registered() -> None:
    result = runner.invoke(
        app, [
            "data", "submit", "data-m1-direct-current-hcv-v1",
            "--operation", "m1-direct-pilot-screen-v8-collect", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "m1-direct-pilot-screen-v8-collect" in result.stdout


def test_m1_direct_v9_recovery_chain_operations_are_registered() -> None:
    for operation in ("m1-direct-pilot-screen-v9-collect", "m1-direct-pilot-v10-collect"):
        result = runner.invoke(
            app, [
                "data", "submit", "data-m1-direct-current-hcv-v1",
                "--operation", operation, "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert operation in result.stdout


def test_m1_direct_full_audit_operation_is_registered() -> None:
    result = runner.invoke(
        app, [
            "data", "submit", "data-m1-direct-current-hcv-v1",
            "--operation", "m1-direct-full-audit", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "m1-direct-full-audit" in result.stdout


def test_m1_direct_replenishment_audit_operation_is_registered() -> None:
    result = runner.invoke(
        app, [
            "data", "submit", "data-m1-direct-current-hcv-v1",
            "--operation", "m1-direct-replenishment-v1-audit-merge", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "m1-direct-replenishment-v1-audit-merge" in result.stdout


def test_m1_direct_next_replenishment_operations_are_registered() -> None:
    for operation in (
        "m1-direct-replenishment-next-plan",
        "m1-direct-replenishment-next-collect",
        "m1-direct-replenishment-next-audit-merge",
    ):
        result = runner.invoke(
            app, [
                "data", "submit", "data-m1-direct-current-hcv-v1",
                "--operation", operation, "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert operation in result.stdout


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
    assert "HCV_FREEZE_AUDIT=outputs/artifacts/datasets/agrinet-hcv-manual-json-v3-prompt-contract/validation.json" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in result.stdout
    assert "RAG_API=http://127.0.0.1:8078" in result.stdout


def test_hcv_five_turn_queue_dry_run_carries_declared_master_port() -> None:
    result = runner.invoke(
        app, [
            "vlm", "submit",
            "vlm-sft-qwen3vl4b-hcv-manual-json-five-turn-terminal-rejection-v6-8gpu-v1",
            "--operation", "manual-json-checkpoint-queue", "--dry-run",
        ]
    )
    assert result.exit_code == 0
    assert "MASTER_PORT=29684" in result.stdout


def test_hcv_five_turn_recovery_queue_reuses_declared_artifact_root() -> None:
    result = runner.invoke(
        app, [
            "vlm", "submit",
            "vlm-sft-qwen3vl4b-hcv-manual-json-five-turn-terminal-rejection-v6-8gpu-eval-v1",
            "--operation", "manual-json-checkpoint-queue", "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "QUEUE_ARTIFACT_ROOT=outputs/runs/vlm/vlm-sft-qwen3vl4b-hcv-manual-json-five-turn-terminal-rejection-v6-8gpu-eval-v1/20260823T224528-2ef993b1-a01/artifacts" in result.stdout


def test_formal_direct_recovery_dry_run_preserves_explicit_root() -> None:
    result = runner.invoke(
        app, [
            "vlm", "submit",
            "vlm-hcv-five-turn-v6-epoch2-direct-formal-recovery-v1",
            "--operation", "formal-direct-native", "--dry-run",
        ]
    )
    assert result.exit_code == 0
    assert "FORMAL_ROOT=outputs/runs/vlm/vlm-sft-qwen3vl4b-hcv-manual-json-five-turn-terminal-rejection-v6-8gpu-eval-v1/" in result.stdout


def test_hcv_teacher_collection_dry_run_preserves_two_turn_contract() -> None:
    result = runner.invoke(
        app, ["rag", "submit", "rag-hcv-visual-expand-teacher-collection-v1", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "--max-tool-turns 2" in result.stdout
    assert "--top-k 3" in result.stdout
    assert "--max-concurrent 1" in result.stdout


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
