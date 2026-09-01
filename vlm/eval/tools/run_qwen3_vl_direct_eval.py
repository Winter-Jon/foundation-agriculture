#!/usr/bin/env python3
"""Run a public-image-only, no-tool Direct retention evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# This directory is an executable-tools directory rather than a Python
# package.  Make the shared public RAG evaluation helpers importable both when
# launched as a script and when loaded by its focused safety test.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_vl_rag_sglang_eval import (
    _chat_completion, _eval_sample, _load_jsonl, _resolve,
)
from agrinet.rag.distill.run_pilot import image_url_content, public_option_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=1200)
    return parser.parse_args()


def public_direct_messages(sample: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
    chinese = sample.get("language") == "zh"
    option = sample.get("question_type") == "option"
    if chinese:
        system = (
            "你是农业图像识别助手。这是无检索、无工具的 Direct 评测：绝不输出 JSON、工具调用或检索请求。"
            "只根据用户图片和公开题面回答。输出 <think>...</think><answer>...</answer>。"
            + ("<answer> 只能包含 A、B、C 或 D 中的一个公开选项字母。" if option else "<answer> 只能包含预测的中文规范类别名称。")
        )
        question = "请识别图中的农业病虫害，并输出其规范名称。"
    else:
        system = (
            "You are an agricultural image recognition assistant. This is a Direct no-retrieval, no-tool evaluation: never output JSON, a tool call, or a retrieval request. "
            "Use only the user image and public question. Output <think>...</think><answer>...</answer>. "
            + ("The <answer> must contain exactly one public option letter A, B, C, or D." if option else "The <answer> must contain only the predicted canonical English class name.")
        )
        question = "Identify the agricultural object or disease in the image and output its canonical name."
    if option:
        question = public_option_question(sample)
    return [{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": question}, image_url_content(image_path)]}]


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(Path(args.manifest))
    rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    root = Path(args.repo_root).resolve()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            image_path = _resolve(str(row["image_path"]), root)
            sample = _eval_sample(row, image_path)
            response = _chat_completion(args.api_base, args.api_key, args.model, public_direct_messages(sample, image_path), max_new_tokens=args.max_new_tokens, temperature=0.0, timeout=args.request_timeout)
            choices = response.get("choices") or []
            message = choices[0].get("message") if choices else {}
            content = message.get("content") if isinstance(message, dict) else ""
            out = dict(row)
            out.update({"prediction": str(content or "").strip(), "direct_no_tool": True, "tool_turns": 0, "forced_tool_turns": 0})
            stream.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row['id']} direct_no_tool", flush=True)


if __name__ == "__main__":
    main()
