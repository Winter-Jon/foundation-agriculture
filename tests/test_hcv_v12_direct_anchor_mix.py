from agrinet.research.hcv.build_v12_direct_anchor_mix import build


def _row(sample_id: str, tool: bool) -> dict:
    messages = [{'role': 'user', 'content': 'identify'}]
    if tool:
        messages += [{'role': 'assistant', 'content': 'think'}, {'role': 'tool_call', 'content': {'name': 'agrinet_rag_search', 'arguments': {'query': 'leaf'}}}, {'role': 'tool', 'content': 'evidence'}]
    messages += [{'role': 'assistant', 'content': 'answer'}]
    return {'sample_id': sample_id, 'images': ['image.jpg'], 'messages': messages}


def test_direct_anchor_mix_replays_only_zero_tool_rows() -> None:
    rows, report = build([_row('direct', False), _row('rag', True)], direct_replay_factor=3, min_direct_token_fraction=0.1)
    assert len(rows) == 4
    assert [row['sample_id'] for row in rows] == ['direct', 'rag', 'direct--v12-direct-anchor-2', 'direct--v12-direct-anchor-3']
    assert report['training_authorized']
    assert report['invariants']['no_new_tool_trajectory']
