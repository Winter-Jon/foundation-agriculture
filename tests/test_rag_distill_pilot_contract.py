from types import SimpleNamespace

from tools.rag_distill import run_pilot


def test_execute_rag_call_omits_image_for_text_only_modes(monkeypatch) -> None:
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append((url, body))
        return {"hybrid": []}

    monkeypatch.setattr(run_pilot, "post_json", fake_post)
    sample = {"query_image": "datasets/example.jpg"}
    for retrieval_type in ("semantic", "name"):
        response, ledger = run_pilot.execute_rag_call(
            "http://rag", sample,
            {"retrieval_type": retrieval_type, "image": "none", "top_k": 3, "query": "public class name"},
        )
        assert response["status"] == "success"
        assert ledger["ok"]
    assert all("image_path" not in body for _, body in calls)


def test_execute_rag_call_keeps_image_for_visual_modes(monkeypatch) -> None:
    bodies = []
    monkeypatch.setattr(run_pilot, "post_json", lambda url, body, headers, timeout: bodies.append(body) or {"hybrid": []})
    run_pilot.execute_rag_call(
        "http://rag", {"query_image": "datasets/example.jpg"},
        {"retrieval_type": "visual", "image": "query_image", "top_k": 3, "query": "leaf symptoms"},
    )
    assert bodies[0]["image_path"] == "datasets/example.jpg"


def test_hcv_strategy_uses_distinct_top_k_per_turn() -> None:
    sample = {"strategy_id": "hcv_visual_expand", "preferred_sequence": ["visual", "visual"]}
    assert run_pilot.strategy_top_k_for_turn(sample, 3, 0) == 3
    assert run_pilot.strategy_top_k_for_turn(sample, 3, 1) == 10
    call = {"arguments": {"query": "brown leaf lesions", "top_k": 3}}
    expanded = run_pilot.hcv_visual_expand_args(call, sample, 3)
    assert expanded["retrieval_type"] == "visual"
    assert expanded["image"] == "query_image"
    assert expanded["query"] == "brown leaf lesions"
    assert expanded["top_k"] == 10


def test_public_option_question_uses_blind_public_order() -> None:
    sample = {
        "language": "zh",
        "public_option_labels": [
            {"name": "One", "chinese_name": "一"}, {"name": "Two", "chinese_name": "二"},
            {"name": "Three", "chinese_name": "三"}, {"name": "Four", "chinese_name": "四"},
        ],
    }
    prompt = run_pilot.public_option_question(sample)
    assert "A. 一" in prompt and "D. 四" in prompt
