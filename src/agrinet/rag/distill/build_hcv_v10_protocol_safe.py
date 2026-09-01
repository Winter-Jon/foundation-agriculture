#!/usr/bin/env python3
"""Build the v10 safe HCV freeze from the immutable v7 source."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.rag.distill.catalog_and_isolation import read_jsonl, write_jsonl
from agrinet.rag.distill.build_hcv_v9_protocol_retention import build
from agrinet.data.rebuild_sft import canonical_json_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--direct-replay-factor', type=int, default=1)
    parser.add_argument('--min-direct-token-fraction', type=float, default=0.35)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f'destination already exists and is non-empty: {args.destination}')
    rows, report = build(read_jsonl(args.source), args.direct_replay_factor, args.min_direct_token_fraction)
    report['schema_version'] = 'agrinet.hcv-v10-protocol-safe/v1'
    report['source_artifact'] = str(args.source)
    report['training_authorized'] = all(report['invariants'].values())
    if not report['training_authorized']:
        raise ValueError('v10 protocol-safe invariants failed')
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / 'data.jsonl', rows)
    data_sha256 = canonical_json_hash(rows)
    (args.destination / 'validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.destination / 'manifest.yaml').write_text(
        yaml.safe_dump({
            'schema_version': 'agrinet.hcv-v10-protocol-safe/v1',
            'artifact_id': 'agrinet-hcv-manual-json-v10-protocol-safe',
            'immutable': True, 'data_sha256': data_sha256,
            'validation': report,
        }, sort_keys=False, allow_unicode=True), encoding='utf-8'
    )
    print(json.dumps({'artifact': str(args.destination), 'data_sha256': data_sha256, 'training_authorized': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
