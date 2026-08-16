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


def test_eval_sample_preserves_manifest_public_option_order(tmp_path: Path) -> None:
    runner = _module()
    row = {
        "source_sample_id": "s", "image_path": "image.jpg", "language": "en", "task_domain": "disease",
        "question_type": "option", "label_code": "N2", "label_name": "Secret target", "option_answer": "B",
        "option_codes": ["N1", "N2", "N3", "N4"], "option_names": ["Public A", "Public B", "Public C", "Public D"],
    }
    sample = runner._eval_sample(row, tmp_path / "image.jpg")
    assert sample["candidate_labels"] == [
        {"code": "N1", "name": "Public A"}, {"code": "N2", "name": "Public B"},
        {"code": "N3", "name": "Public C"}, {"code": "N4", "name": "Public D"},
    ]
