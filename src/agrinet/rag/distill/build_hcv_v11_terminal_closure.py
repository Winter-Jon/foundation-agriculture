#!/usr/bin/env python3
"""Build v11 by expanding audited native terminal-closure examples."""
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
from agrinet.rag.distill.augment_hcv_invalid_terminal import build as add_terminal_context
from agrinet.rag.distill.catalog_and_isolation import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--per-cell', type=int, default=70)
    args = parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError(f'destination already exists and is non-empty: {args.destination}')
    source = read_jsonl(args.source)
    # The v10 artifact already contains v3 closure rows.  Keep the immutable
    # source rows for provenance, but never use prior closure rows as new
    # anchors; otherwise rebuilding would duplicate their IDs and curriculum.
    source_for_augmentation = [
        row for row in source
        if (row.get('metadata') or {}).get('route') not in {'terminal_context_v3', 'terminal_context_v4'}
    ]
    augmented, closure_report = add_terminal_context(
        source_for_augmentation, per_cell=args.per_cell,
        route_name="terminal_context_v4", sample_suffix="--terminal-context-v4",
    )
    rows = [*source, *augmented[len(source_for_augmentation):]]
    report = {
        'schema_version': 'agrinet.hcv-v11-terminal-closure/v1',
        'source_artifact': str(args.source),
        'source_rows': len(source), 'rows': len(rows),
        'added_rows': len(rows) - len(source), 'added_per_cell': args.per_cell,
        'terminal_closure_report': closure_report,
        'invariants': {
            'source_unchanged_in_order': rows[:len(source)] == source,
            'unique_sample_ids': len({str(r.get('sample_id') or '') for r in rows}) == len(rows),
            'query_image_only': all(len(r.get('images') or []) == 1 for r in rows),
            'no_malformed_assistant_supervision': all(
                not (m.get('role') == 'assistant' and any(x in str(m.get('content') or '') for x in ('<tool_call>', 'invalid_tool_call')))
                for r in rows for m in r.get('messages') or []
            ),
            'all_added_end_in_assistant': all((r.get('messages') or [])[-1].get('role') == 'assistant' for r in rows[len(source):]),
            'closure_authorized': closure_report.get('training_authorized') is True,
        },
    }
    report['training_authorized'] = all(report['invariants'].values())
    if not report['training_authorized']:
        raise ValueError('v11 terminal-closure invariants failed')
    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.destination / 'data.jsonl', rows)
    data_sha256 = canonical_json_hash(rows)
    report['data_sha256'] = data_sha256
    (args.destination / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.destination / 'manifest.yaml').write_text(yaml.safe_dump({
        'schema_version': 'agrinet.hcv-v11-terminal-closure/v1',
        'artifact_id': 'agrinet-hcv-manual-json-v11-terminal-closure',
        'immutable': True, 'data_sha256': data_sha256, 'validation': report,
    }, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps({'artifact': str(args.destination), 'rows': len(rows), 'added_rows': len(rows) - len(source), 'data_sha256': data_sha256, 'training_authorized': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
