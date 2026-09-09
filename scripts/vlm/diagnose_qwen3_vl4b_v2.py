#!/usr/bin/env python3
"""Run one bounded raw Qwen3-VL-4B execution diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    started = time.perf_counter()
    result: dict[str, object] = {
        "schema_version": "agrinet.qwen3-vl4b-diagnostic/v1",
        "model": str(args.model), "image": str(args.image),
        "device": args.device, "max_new_tokens": args.max_new_tokens,
        "status": "failed", "training_eligible": False,
    }
    try:
        if not args.model.is_dir():
            raise FileNotFoundError(args.model)
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device_index) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": device}, local_files_only=True,
        ).eval()
        result["load_seconds"] = round(time.perf_counter() - load_started, 3)
        with Image.open(args.image) as image:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": "Describe the visible plant symptom briefly."},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result.update({
            "status": "complete", "generation_seconds": round(time.perf_counter() - generation_started, 3),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(trimmed[0].shape[-1]), "output_preview": text[:2000],
        })
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
