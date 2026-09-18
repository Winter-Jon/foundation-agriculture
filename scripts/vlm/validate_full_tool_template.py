"""Run with .venv_test: inspect actual qwen3_vl/Hermes supervised targets."""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MAX_PIXELS", "1048576")
from swift.model import get_model_processor
from swift.template import get_template
from swift.template.base import MaxLengthError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    args = p.parse_args()
    _, processor = get_model_processor(args.model, load_model=False, model_type="qwen3_vl")
    template = get_template(processor, template_type="qwen3_vl", agent_template="hermes",
                            loss_scale="hermes", max_length=16384, truncation_strategy="raise")
    template.set_mode("train")
    tokenizer = processor.tokenizer
    checks = []
    for row_index, line in enumerate(args.data.read_text().splitlines()):
        row = json.loads(line)
        try:
            encoded = template.encode(copy.deepcopy(row))
        except MaxLengthError as exc:
            checks.append({"row_index": row_index, "error": "max_length_exceeded", "detail": str(exc)})
            continue
        ids, labels = encoded["input_ids"], encoded["labels"]
        rendered = tokenizer.decode(ids)
        supervised = tokenizer.decode([label for label in labels if label != -100])
        calls = [call for message in row["messages"] for call in message.get("tool_calls", [])]
        rendered_calls = [json.loads(value) for value in re.findall(r'<tool_call>(.*?)</tool_call>', supervised, re.S)]
        expected_calls = [{"name": call["function"]["name"],
                           "arguments": json.loads(call["function"]["arguments"])} for call in calls]
        final = row["messages"][-1]["content"]
        evidence_parts = [part for message in row["messages"] if message["role"] == "tool"
                          for part in message["content"].split("<image>") if part.strip()]
        image_tokens = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_spans = rendered.count("<|vision_start|>")
        weights = encoded.get('loss_scale')
        if weights is None:
            weights = [1.0] * len(labels)
        checks.append({"row_index": row_index, "length": len(ids), "supervised_tokens": sum(x != -100 for x in labels),
                       'effective_supervised_tokens': sum(float(w) for label, w in zip(labels, weights) if label != -100),
                       'loss_weight_counts': {str(w): sum(label != -100 and weight == w for label, weight in zip(labels, weights)) for w in sorted(set(weights))},
                       "final_present": final in rendered, "final_supervised": final in supervised,
                       "all_calls_supervised": rendered_calls == expected_calls,
                       "all_evidence_present": all(part in rendered for part in evidence_parts),
                       "evidence_masked": all(part not in supervised for part in evidence_parts),
                       "all_images_present": image_spans == len(row["images"]),
                       "image_tokens_masked": all(label == -100 for token, label in zip(ids, labels) if token == image_tokens)})
    passed = bool(checks) and all(all(row.get(key, False) for key in ("final_present", "final_supervised", "all_calls_supervised", "all_evidence_present", "evidence_masked", "all_images_present", "image_tokens_masked")) for row in checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": passed, "rows": checks, "max_length": 16384,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "template": "qwen3_vl", "agent_template": "hermes", "loss_scale": "hermes", "truncation": "raise"}, indent=2))
    print(json.dumps({"passed": passed, "rows": len(checks)}))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
