import json
from pathlib import Path


def test_prompt_repair_preserves_evidence_and_sets_task_specific_contract(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).parents[1] / "tools/rag_distill/repair_hcv_prompt_contract.py"
    spec = importlib.util.spec_from_file_location("hcv_prompt_repair", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {
        "sample_id": "s", "images": ["image.jpg"],
        "messages": [
            {"role": "user", "content": "<image> Identify this."},
            {"role": "assistant", "content": "<answer>B</answer>"},
        ],
        "metadata": {
            "language": "en", "task_domain": "disease", "question_type": "option",
            "candidate_labels": [
                {"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}
            ],
        },
    }
    derived = module.derive([source])[0]
    assert "Answer with only the option letter" in derived["messages"][0]["content"]
    assert "A. A" in derived["messages"][0]["content"]
    assert derived["messages"][1:] == source["messages"][1:]
    assert derived["images"] == source["images"]
    assert derived["metadata"]["student_user_contract"] == "agrinet.current-task-contract/v2"
