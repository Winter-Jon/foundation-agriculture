"""Bounded, read-only environment/data checks for the M1 Transformers-5.9.0 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise SystemExit('dataset is empty')
    row = rows[0]
    messages = row.get('messages')
    if not isinstance(messages, list) or not messages:
        raise SystemExit('first row has no messages')
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = processor(text=[text], return_tensors='pt', padding=True)
    token_count = int(encoded['input_ids'].shape[-1])
    report = {
        'dataset': str(args.dataset), 'dataset_sha256': digest(args.dataset), 'rows': len(rows),
        'sample_id': row.get('id') or row.get('metadata', {}).get('sample_id'),
        'messages': len(messages), 'token_count_without_image_payload': token_count,
        'transformers': transformers.__version__, 'cuda_available': torch.cuda.is_available(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
