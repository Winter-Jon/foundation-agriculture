import json

import pytest

from agrinet.vlm.full_tool_inference import parse_completion, wire_messages, evaluate
from agrinet.vlm.full_tool import CLASSIFIER_PREDICT, RAG_SEARCH


def call(name, arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def test_generation_stops_at_complete_call_without_trimming():
    from agrinet.vlm.full_tool_evaluation import generation_payload
    messages = [{'role': 'user', 'content': 'Observe.'}]
    payload = generation_payload('model', messages, 4096)
    assert payload['stop'] == ['</tool_call>']
    assert payload['no_stop_trim'] is True
    assert payload['messages'] == messages
    first = call(CLASSIFIER_PREDICT, {})
    decoded = (first + call(RAG_SEARCH, {})).split(payload['stop'][0], 1)[0] + payload['stop'][0]
    assert parse_completion(decoded, 0)['tool_calls'][0]['function']['name'] == CLASSIFIER_PREDICT
    from agrinet.vlm.full_tool_inference import FINALIZE
    final = generation_payload('model', [{'role': 'user', 'content': FINALIZE}], 4096)
    assert 'regex' in final and 'stop' not in final
    import re
    assert re.fullmatch(final['regex'], '<think>Evidence remains ambiguous.</think><answer>INSUFFICIENT_EVIDENCE</answer>')
    assert not re.fullmatch(final['regex'], first)


def test_parser_preserves_planning_and_rejects_multiple():
    value = '<think>Inspect lesions.</think>' + call(CLASSIFIER_PREDICT, {})
    assert parse_completion(value, 1)['content'] == '<think>Inspect lesions.</think>'
    with pytest.raises(ValueError, match='exactly_one'):
        parse_completion(value + call(CLASSIFIER_PREDICT, {}), 2)
    with pytest.raises(ValueError):
        parse_completion(call(RAG_SEARCH, {'retrieval_type': 'name', 'image': 'query_image'}), 2)


def test_reference_is_inside_tool_wrapper():
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AA=='}}
    result = wire_messages([{'role': 'tool', 'tool_call_id': 'x', 'content': '{"top3": []}'},
                            {'role': 'user', 'content': [{'type': 'text', 'text': 'REF1'}, image]}])
    assert len(result) == 1
    assert result[0]['role'] == 'user'
    parts = result[0]['content']
    assert parts[0]['text'] == '<tool_response>\n'
    assert parts[-1]['text'] == '\n</tool_response>'
    assert image in parts


def test_observation_tools_and_terminal(monkeypatch):
    import agrinet.vlm.full_tool_inference as module
    monkeypatch.setattr(module, 'transport_image', lambda *a, **kw: ({'type': 'image_url', 'image_url': {'url': 'query'}}, {}))
    monkeypatch.setattr(module, 'reference_attachments', lambda *a, **kw: (None, []))
    responses = iter(['Visible lesions affect a leaf, while the reverse side and stem cannot be assessed.',
                      call(CLASSIFIER_PREDICT, {}), call(RAG_SEARCH, {
                          'retrieval_type': 'semantic', 'query': 'lesion border', 'top_k': 3, 'rationale': 'Resolve morphology'}),
                      '<think>Visible lesions support the retrieved candidate.</think><answer>Example</answer>'])
    histories = []
    def generate(messages):
        histories.append(messages)
        return next(responses)
    result = evaluate({'image_path': 'unused', 'image_sha256': 'a', 'question': 'Identify.'},
                      generate=generate, classifier=lambda: {'top3': []}, retrieve=lambda args: {'evidence': []})
    assert result['protocol_error'] is None
    assert result['tool_history'] == [CLASSIFIER_PREDICT, RAG_SEARCH]
    assert result['retrieval_modes'] == ['semantic']
    assert 'Observation stage complete' in histories[1][-1]['content']


def test_premature_final_never_accepted(monkeypatch):
    import agrinet.vlm.full_tool_inference as module
    monkeypatch.setattr(module, 'transport_image', lambda *a, **kw: ({'type': 'image_url', 'image_url': {'url': 'query'}}, {}))
    responses = iter(['Visible lesions affect a leaf, while the reverse side and stem cannot be assessed.',
                      '<think>Reasoning.</think><answer>Example</answer>',
                      '<think>Reasoning.</think><answer>Example</answer>'])
    result = evaluate({'image_path': 'unused', 'image_sha256': 'a', 'question': 'Identify.'},
                      generate=lambda _: next(responses), classifier=lambda: {}, retrieve=lambda _: {})
    assert result['protocol_error']
    assert not result['prediction']


def test_third_search_requires_finalization(monkeypatch):
    import agrinet.vlm.full_tool_inference as module
    monkeypatch.setattr(module, 'transport_image', lambda *a, **kw: ({'type': 'image_url', 'image_url': {'url': 'query'}}, {}))
    monkeypatch.setattr(module, 'reference_attachments', lambda *a, **kw: (None, []))
    responses = iter(['Visible lesions affect a leaf, while the reverse side and stem cannot be assessed.',
                      call(CLASSIFIER_PREDICT, {}),
                      *[call(RAG_SEARCH, {'retrieval_type': 'semantic', 'query': str(i), 'top_k': 3, 'rationale': 'Resolve evidence gap'}) for i in range(3)],
                      '<think>Evidence is insufficient for a confident diagnosis.</think><answer>INSUFFICIENT_EVIDENCE</answer>'])
    histories = []
    def generate(messages):
        histories.append(messages)
        return next(responses)
    result = evaluate({'image_path': 'unused', 'image_sha256': 'a', 'question': 'Identify.'},
                      generate=generate, classifier=lambda: {'top3': []}, retrieve=lambda _: {'evidence': []})
    assert result['protocol_error'] is None
    assert result['rag_calls'] == 3
    assert module.FINALIZE in str(histories[-1])
    assert result['insufficient_evidence']
