#!/usr/bin/env python3
"""Run an image-only raw-Qwen diagnostic over the fixed v2 smoke source.

This is deliberately an execution diagnostic, not an evaluation or SFT
converter.  It reads only the public source fields, loads the raw model once,
and commits one result per image so interruption cannot erase completed work.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def read_source(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 32:
        raise ValueError("v2 raw diagnostic requires exactly 32 smoke-source rows")
    required = ("sample_id", "image_path", "image_sha256", "question")
    if len({str(row.get("sample_id") or "") for row in rows}) != 32:
        raise ValueError("source has non-unique sample identities")
    if any(not isinstance(row.get(key), str) or not row[key] for row in rows for key in required):
        raise ValueError("source has incomplete public diagnostic fields")
    return rows


def completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") in {"complete", "failed"}:
                result.add(str(row.get("sample_id") or ""))
    return result - {""}


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 8192:
        raise ValueError("max-new-tokens must be in [1, 8192]")
    rows = read_source(args.source)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    existing = completed(results_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started = time.perf_counter()
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}, local_files_only=True,
    ).eval()
    load_seconds = round(time.perf_counter() - load_started, 3)
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in existing:
            continue
        record: dict[str, object] = {
            "schema_version": "agrinet.qwen3-vl4b-source-diagnostic/v1",
            "sample_id": sample_id, "image_sha256": row["image_sha256"],
            "model": str(args.model), "device": args.device,
            "max_new_tokens": args.max_new_tokens, "training_eligible": False, "status": "failed",
        }
        one_started = time.perf_counter()
        try:
            image_path = Path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with Image.open(image_path) as image:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": row["question"]},
                ]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                record["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
            text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            record.update({"status": "complete", "generation_seconds": round(time.perf_counter() - generation_started, 3),
                           "input_tokens": int(inputs["input_ids"].shape[-1]),
                           "output_tokens": int(trimmed[0].shape[-1]), "output_preview": text[:2000]})
        except Exception as exc:
            record["error_type"] = type(exc).__name__
        record["elapsed_seconds"] = round(time.perf_counter() - one_started, 3)
        append(results_path, record)
    recorded = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["sample_id"]): row for row in recorded}
    if len(by_id) != len(recorded):
        raise ValueError("diagnostic result ledger has duplicate sample identities")
    states = Counter(str(row.get("status") or "unknown") for row in recorded)
    summary = {"schema_version": "agrinet.qwen3-vl4b-source-diagnostic-summary/v1",
               "source": str(args.source), "model": str(args.model), "device": args.device,
               "load_seconds": load_seconds, "elapsed_seconds": round(time.perf_counter() - started, 3),
               "rows_expected": 32, "rows_recorded": len(recorded), "statuses": dict(sorted(states.items())),
               "training_eligible": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if len(recorded) == 32 else 2


if __name__ == "__main__":
    raise SystemExit(main())
