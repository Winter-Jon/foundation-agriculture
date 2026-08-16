import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "vlm/eval/tools/run_qwen3_vl_direct_eval.py"
    spec = importlib.util.spec_from_file_location("direct_eval_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_messages_do_not_expose_manifest_label_or_tools(tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"not-a-real-image")
    sample = {"language": "en", "question_type": "open", "final_label": "N99999", "final_label_name": "Secret class"}
    messages = runner.public_direct_messages(sample, image)
    text = str(messages)
    assert "Secret class" not in text
    assert "N99999" not in text
    assert "agrinet_rag_search" not in text
