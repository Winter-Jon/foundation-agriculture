#!/usr/bin/env python3
"""Freeze the original zero-tool Direct anchors from the v12 SFT artifact."""
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
from agrinet.research.shared.catalog import read_jsonl, write_jsonl
from agrinet.research.hcv.freeze import tool_turns


def build(rows: list[dict]) -> tuple[list[dict], dict]:
    """Retain each original zero-tool sample once, in stable source order."""
    direct = [
        row for row in rows
        if tool_turns(row) == 0
        and not str((row.get('metadata') or {}).get('route', '')).startswith('v12_direct_anchor_replay')
    ]
    ids = [str(row.get('sample_id') or '') for row in direct]
    report = {
        'schema_version': 'agrinet.hcv-v12-direct-only/v1',
        'source_rows': len(rows),
        'rows': len(direct),
        'route_rows': {'direct_only': len(direct)},
        'invariants': {
            'non_empty': bool(direct),
            'unique_sample_ids': len(ids) == len(set(ids)),
            'zero_tool_calls': all(tool_turns(row) == 0 for row in direct),
            'no_v12_replay_duplicates': all('--v12-direct-anchor-' not in sample_id for sample_id in ids),
            'query_image_only': all(len(row.get('images') or []) == 1 for row in direct),
        },
    }
    report['training_authorized'] = all(report['invariants'].values())
    if not report['training_authorized']:
        raise ValueError('Direct-only v12 invariants failed')
    return direct, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f'destination already exists and is non-empty: {args.destination}')
    rows, report = build(read_jsonl(args.source))
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / 'data.jsonl', rows)
    report['data_sha256'] = canonical_json_hash(rows)
    (args.destination / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.destination / 'manifest.yaml').write_text(yaml.safe_dump({'schema_version': report['schema_version'], 'artifact_id': args.destination.name, 'immutable': True, 'data_sha256': report['data_sha256'], 'validation': report}, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps({'artifact': str(args.destination), 'rows': len(rows), 'data_sha256': report['data_sha256'], 'training_authorized': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
