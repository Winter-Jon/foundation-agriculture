import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py"
    spec = importlib.util.spec_from_file_location("rag_eval_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_eval_tool_message_never_serializes_manifest_target() -> None:
    runner = _module()
    sample = {
        "language": "en", "question_type": "open",
        "final_label": "secret-code", "final_label_name": "Secret ground truth",
        "final_label_zh": "私有标签", "label_aliases": ["Secret alias"],
    }
    message = runner._tool_response_message(sample, {"status": "success", "results": [{"class_name": "Public class"}]}, [])
    text = message["content"][0]["text"]
    for forbidden in ("Private teacher-forcing", "Secret ground truth", "私有标签", "Secret alias", "secret-code"):
        assert forbidden not in text
    assert "Public class" in text
