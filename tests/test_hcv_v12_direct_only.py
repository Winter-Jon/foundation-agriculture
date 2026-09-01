from tools.rag_distill.build_hcv_v12_direct_only import build


def _row(sample_id: str, tool: bool, replay: bool = False) -> dict:
    messages = [{'role': 'user', 'content': 'identify'}]
    if tool:
        messages += [{'role': 'tool_call', 'content': {'name': 'agrinet_rag_search', 'arguments': {'query': 'leaf'}}}]
    messages += [{'role': 'assistant', 'content': 'answer'}]
    metadata = {'route': 'v12_direct_anchor_replay'} if replay else {}
    return {'sample_id': sample_id, 'images': ['image.jpg'], 'messages': messages, 'metadata': metadata}


def test_direct_only_retains_unique_original_zero_tool_rows() -> None:
    rows, report = build([_row('direct', False), _row('rag', True), _row('direct--v12-direct-anchor-2', False, replay=True)])
    assert [row['sample_id'] for row in rows] == ['direct']
    assert report['training_authorized']
    assert report['invariants']['zero_tool_calls']
