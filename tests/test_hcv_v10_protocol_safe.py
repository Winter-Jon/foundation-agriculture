from tools.rag_distill.build_hcv_v9_protocol_retention import build


def test_safe_terminal_rows_end_in_assistant_and_use_no_malformed_json() -> None:
    row = {
        'sample_id': 'x', 'images': ['x.jpg'],
        'metadata': {'question_type': 'open', 'language': 'en', 'task_domain': 'disease'},
        'messages': [
            {'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'x'},
            {'role': 'tool_call', 'content': '{}'}, {'role': 'tool', 'content': '{}'},
            {'role': 'assistant', 'content': '<answer>Public class</answer>'},
        ],
    }
    rows = [row | {'sample_id': f'x-{q}-{lang}-{domain}-{i}', 'metadata': row['metadata'] | {'task_domain': domain, 'question_type': q, 'language': lang}}
            for q in ('open', 'option') for lang in ('en', 'zh') for domain in ('disease', 'pest') for i in range(8)]
    rows.extend({
        'sample_id': f'direct-{i}', 'images': [f'direct-{i}.jpg'],
        'metadata': {'question_type': 'open', 'language': 'en', 'task_domain': 'disease'},
        'messages': [{'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '<answer>Public class</answer>'}],
    } for i in range(8))
    output, report = build(rows, direct_replay_factor=1, min_direct_token_fraction=0.0)
    assert report['training_authorized']
    for item in output[len(rows):]:
        if item['metadata'].get('route') != 'terminal_context_v3':
            continue
        assert item['messages'][-1]['role'] == 'assistant'
        assert [m['role'] for m in item['messages'][-4:]] == ['assistant', 'tool_call', 'tool', 'assistant']
        assert 'invalid_tool_call' not in ''.join(str(m.get('content', '')) for m in item['messages'])
