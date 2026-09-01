#!/usr/bin/env python3
"""Freeze a v12 HCV dataset by token-weighted replay of v11 Direct anchors.

This intentionally changes only the training mixture: v11 rows are retained in
their original order and every zero-tool Direct anchor is repeated with distinct
provenance.  No new loss, model-side constraint, or synthetic trajectory is
introduced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.data.rebuild_sft import canonical_json_hash
from agrinet.rag.distill.catalog_and_isolation import read_jsonl, write_jsonl
from agrinet.rag.distill.freeze_hcv_sft import token_proxy, tool_turns


def build(rows: list[dict], direct_replay_factor: int, min_direct_token_fraction: float) -> tuple[list[dict], dict]:
    if direct_replay_factor < 1:
        raise ValueError('direct_replay_factor must be positive')
    if not 0.0 < min_direct_token_fraction < 1.0:
        raise ValueError('min_direct_token_fraction must be in (0, 1)')
    direct = [row for row in rows if tool_turns(row) == 0]
    if not direct:
        raise ValueError('source contains no zero-tool Direct anchors')
    replay: list[dict] = []
    for replica in range(2, direct_replay_factor + 1):
        for row in direct:
            item = json.loads(json.dumps(row, ensure_ascii=False))
            source_id = str(item.get('sample_id') or '')
            item['sample_id'] = f'{source_id}--v12-direct-anchor-{replica}'
            metadata = dict(item.get('metadata') or {})
            metadata.update({'route': 'v12_direct_anchor_replay', 'direct_anchor_source': source_id, 'direct_anchor_replica': replica})
            item['metadata'] = metadata
            replay.append(item)
    output = [*rows, *replay]
    sample_ids = [str(row.get('sample_id') or '') for row in output]
    total_tokens = sum(token_proxy(row) for row in output)
    direct_tokens = sum(token_proxy(row) for row in output if tool_turns(row) == 0)
    direct_fraction = direct_tokens / total_tokens if total_tokens else 0.0
    report = {
        'schema_version': 'agrinet.hcv-v12-direct-anchor-mix/v1',
        'source_rows': len(rows),
        'rows': len(output),
        'direct_anchor_rows': len(direct),
        'direct_replay_rows': len(replay),
        'direct_replay_factor': direct_replay_factor,
        'route_token_proxy': {'total': total_tokens, 'direct': direct_tokens, 'direct_fraction': direct_fraction, 'direct_fraction_floor': min_direct_token_fraction},
        'invariants': {
            'source_unchanged_in_order': output[:len(rows)] == rows,
            'unique_sample_ids': len(sample_ids) == len(set(sample_ids)),
            'query_image_only': all(len(row.get('images') or []) == 1 for row in output),
            'no_new_tool_trajectory': all(tool_turns(row) == 0 for row in replay),
            'direct_token_weight_floored': direct_fraction >= min_direct_token_fraction,
        },
    }
    report['training_authorized'] = all(report['invariants'].values())
    if not report['training_authorized']:
        raise ValueError('v12 Direct-anchor invariants failed')
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--direct-replay-factor', type=int, default=3)
    parser.add_argument('--min-direct-token-fraction', type=float, default=0.55)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f'destination already exists and is non-empty: {args.destination}')
    rows, report = build(read_jsonl(args.source), args.direct_replay_factor, args.min_direct_token_fraction)
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / 'data.jsonl', rows)
    data_sha256 = canonical_json_hash(rows)
    report['data_sha256'] = data_sha256
    (args.destination / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.destination / 'manifest.yaml').write_text(yaml.safe_dump({'schema_version': report['schema_version'], 'artifact_id': args.destination.name, 'immutable': True, 'data_sha256': data_sha256, 'validation': report}, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps({'artifact': str(args.destination), 'rows': len(rows), 'direct_token_fraction': report['route_token_proxy']['direct_fraction'], 'data_sha256': data_sha256, 'training_authorized': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
