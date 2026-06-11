#!/usr/bin/env python3
"""Run local Qwen3-VL inference on an AgriNet eval manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def _resolve(path: str, repo_root: Path) -> Path:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    return image_path


def _build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct", help="Model or merged checkpoint path.")
    parser.add_argument("--manifest", required=True, help="Evaluation manifest JSONL.")
    parser.add_argument("--output", required=True, help="Prediction JSONL.")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving relative image paths.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    rows = _load_jsonl(Path(args.manifest))
    if args.limit:
        rows = rows[: args.limit]

    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            image_path = _resolve(str(row["image"]), repo_root)
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
            messages = _build_messages(image_path, str(row["prompt"]))
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            out = dict(row)
            out["prediction"] = response.strip()
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] {row['id']} -> {response.strip()}", flush=True)


if __name__ == "__main__":
    main()
