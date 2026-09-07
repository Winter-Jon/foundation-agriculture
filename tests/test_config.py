import pytest

from agrinet.common.config import ConfigError, load_experiment, resolve_config, validate_experiment_id


def test_load_and_override_experiment() -> None:
    spec = load_experiment("vlm-sft-rag-qwen3vl4b-full-v1")
    resolved = resolve_config(spec, ["parameters.epochs=2", "parameters.enabled=true"])
    assert resolved["parameters"]["epochs"] == 2
    assert resolved["parameters"]["enabled"] is True


@pytest.mark.parametrize("value", ["VLM-test-x-y-v1", "vlm-test-v1", "vlm-test-x-y-latest"])
def test_invalid_experiment_id(value: str) -> None:
    with pytest.raises(ConfigError):
        validate_experiment_id(value)
