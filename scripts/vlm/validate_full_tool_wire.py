"""Compare real training encoding to the public inference history encoding."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MAX_PIXELS', '1048576')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from agrinet.vlm.full_tool_inference import wire_messages
from agrinet.data.full_tool_sft import sha
from swift.model import get_model_processor
from swift.template import get_template


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    _, processor = get_model_processor('models/Qwen3-VL-4B-Instruct', load_model=False, model_type='qwen3_vl')
    template = get_template(processor, template_type='qwen3_vl', agent_template='hermes',
                            loss_scale='hermes', max_length=16384, truncation_strategy='raise')
    template.set_mode('train')
    results = []
    for line in Path(args.data).read_text().splitlines():
        row = json.loads(line)
        messages = wire_messages(row['messages'])
        # Wire inputs contain the tools prompt already; no second tools expansion.
        encoded = template.encode(copy.deepcopy(row))
        wire = template.encode({'messages': messages, 'images': row['images']})
        results.append({'input_ids_equal': encoded['input_ids'] == wire['input_ids'],
                        'labels_equal': encoded['labels'] == wire['labels'],
                        'native_length': len(encoded['input_ids']), 'wire_length': len(wire['input_ids'])})
    passed = bool(results) and all(r['input_ids_equal'] and r['labels_equal'] for r in results)
    Path(args.output).write_text(json.dumps({'passed': passed, 'rows': results, 'data_sha256': sha(args.data)}, indent=2))
    print(json.dumps({'passed': passed, 'rows': len(results)}))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
