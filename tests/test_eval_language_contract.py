from vlm.eval.tools.run_qwen3_vl_rag_sglang_eval import _grounded_chinese_answer


def test_grounded_chinese_answer_uses_only_retrieved_mapping():
    text = '<think>证据</think>\n<answer>Corn Northern Leaf Blight</answer>'
    history = [{'tool_response': {'results': [
        {'class_name': 'Corn Northern Leaf Blight', 'chinese_name': '玉米北方叶枯病'},
    ]}}]
    assert '玉米北方叶枯病' in _grounded_chinese_answer(text, history)


def test_grounded_chinese_answer_does_not_invent_translation():
    text = '<think>证据</think>\n<answer>Unknown English Class</answer>'
    assert _grounded_chinese_answer(text, []) == text
