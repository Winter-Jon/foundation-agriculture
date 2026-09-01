import json

from tools.rag_distill.augment_hcv_invalid_terminal import build


def _row(sample_id: str, language: str, question_type: str, domain: str) -> dict:
    return {
        'sample_id': sample_id, 'images': ['query.jpg'],
        'metadata': {'language': language, 'question_type': question_type, 'task_domain': domain},
        'messages': [
            {'role': 'system', 'content': 'bare JSON tool contract'},
            {'role': 'user', 'content': 'query'},
            {'role': 'assistant', 'content': '<think>plan</think>'},
            {'role': 'tool_call', 'content': json.dumps({'name': 'agrinet_rag_search', 'arguments': {'query': 'x', 'retrieval_type': 'visual', 'image': 'query_image', 'top_k': 3}})},
            {'role': 'tool', 'content': '{"status":"success","results":[{"name":"Target"}]}'},
            {'role': 'assistant', 'content': '<think>evidence</think><answer>Target</answer>'},
        ],
    }


def test_v11_balanced_closure_addition_has_no_invalid_assistant_targets():
    rows = []
    for question_type in ('open', 'option'):
        for language in ('en', 'zh'):
            for domain in ('disease', 'pest'):
                for index in range(2):
                    rows.append(_row(f'{question_type}-{language}-{domain}-{index}', language, question_type, domain))
    output, report = build(rows, per_cell=2)
    assert report['training_authorized'] is True
    assert len(output) == 32
    assert output[:16] == rows
    assert all(m['role'] != 'assistant' or '<tool_call>' not in str(m.get('content', '')) for r in output for m in r['messages'])
    assert all([m['role'] for m in r['messages'][-4:]] == ['assistant', 'tool_call', 'tool', 'assistant'] for r in output[8:])
