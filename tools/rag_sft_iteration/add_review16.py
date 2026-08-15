import json
from pathlib import Path
src = Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/dataset/accepted_base_plus_round1.jsonl')
row = None
for line in src.open():
    if line.strip():
        item = json.loads(line)
        meta = item.get('metadata', {})
        if meta.get('target_id') == 'rag_open-zh-pest_N05071_N05071_P00002' and meta.get('generation_route') == 'oracle_grounded':
            row = item
            break
if row is None:
    raise SystemExit('row not found')
messages = row['messages']
first = next(i for i, message in enumerate(messages) if message.get('role') == 'tool_call')
new_messages = []
for i, message in enumerate(messages):
    if i == first - 1 and message.get('role') == 'assistant' and str(message.get('content', '')).strip().startswith('<think>'):
        continue
    if i == first:
        message = dict(message)
        message['role'] = 'assistant'
    new_messages.append(message)
row['messages'] = new_messages
row['metadata']['strict_first_tool_call'] = True
row['metadata']['canonicalized_from_legacy'] = True
out = Path('outputs/experiments/rag_sft_iteration/candidates/round-001-strict-review-16/train')
out.mkdir(parents=True, exist_ok=True)
for name in ('agent_sft.accepted.jsonl', 'agent_sft.oracle_grounded.jsonl'):
    with out.joinpath(name).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
print(row['sample_id'])
