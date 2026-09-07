#!/usr/bin/env python3
"""Create immutable OpenAgri v2 SFT ablation views and prompt artifacts.

The formal v2 accepted training data is the only source.  This script changes
rendering prompts only; it never reads dev/test/public/private inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
TRAIN_ROOT = V2_ROOT / "vlm_data/accepted/train"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/artifacts/open-agri-v2-sft-ablation-v1"

V4_SYSTEM = (
    "You are an agricultural visual diagnosis assistant. Reason from the image, "
    "then respond exactly as: <think>concise diagnostic reasoning</think>"
    "<answer>final answer</answer>"
)
OPEN_USER = {
    "en": "<image> Task: identify the image with one canonical disease or pest name. Answer with only the canonical name.",
    "zh": "<image> 任务：根据图像给出一个规范的病害或虫害名称。请只输出规范名称。",
}
OPTION_PREFIX = {
    "en": "<image> Task: choose one of the candidates given below. Answer with only one option letter.",
    "zh": "<image> 任务：在下列候选项中选择一个。请只输出一个选项字母。",
}
SOURCE_FILES = (
    ("disease", "direct"), ("pest", "direct"),
    ("disease", "rag"), ("pest", "rag"),
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def extract_option_lines(content: str, sample_id: str) -> list[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    options = [line for line in lines if len(line) >= 3 and line[0] in "ABCD" and line[1] == "."]
    if len(options) != 4 or [line[0] for line in options] != list("ABCD"):
        raise ValueError(f"{sample_id}: option user message lacks exactly A-D candidates")
    return options


def rewritten_user(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    language = str(metadata.get("language") or "")
    question_type = str(metadata.get("question_type") or "")
    if language not in OPEN_USER or question_type not in {"open", "option"}:
        raise ValueError(f"{record.get('sample_id')}: unsupported language/question type {language}/{question_type}")
    if question_type == "open":
        return OPEN_USER[language]
    messages = record.get("messages") or []
    current = next((str(message.get("content") or "") for message in messages if message.get("role") == "user"), "")
    return OPTION_PREFIX[language] + "\n" + "\n".join(extract_option_lines(current, str(record.get("sample_id"))))


def validate_record(record: dict[str, Any], expected_route: str) -> None:
    metadata = record.get("metadata") or {}
    if metadata.get("route") != expected_route:
        raise ValueError(f"{record.get('sample_id')}: route differs from source view")
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{record.get('sample_id')}: expected exactly one image")
    if metadata.get("class_role") != "known" or metadata.get("v2_image_split") != "train_candidate":
        raise ValueError(f"{record.get('sample_id')}: not a formal v2 Known train_candidate row")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record.get('sample_id')}: missing messages")
    final = messages[-1]
    if final.get("role") != "assistant" or not str(final.get("content") or "").lstrip().startswith("<think>") or "<answer>" not in str(final.get("content") or ""):
        raise ValueError(f"{record.get('sample_id')}: final target is not think/answer formatted")
    has_tool_call = any(message.get("role") == "tool_call" for message in messages)
    if expected_route == "rag" and not has_tool_call:
        raise ValueError(f"{record.get('sample_id')}: RAG row has no tool call")
    if expected_route == "direct" and has_tool_call:
        raise ValueError(f"{record.get('sample_id')}: Direct row unexpectedly has tool call")


def render_record(record: dict[str, Any], route: str) -> dict[str, Any]:
    validate_record(record, route)
    result = dict(record)
    messages: list[dict[str, Any]] = []
    for message in record["messages"]:
        role = message.get("role")
        if role == "system":
            messages.append({"role": "system", "content": V4_SYSTEM})
        elif role == "user":
            messages.append({"role": "user", "content": rewritten_user(record)})
        else:
            messages.append(dict(message))
    result["messages"] = messages
    metadata = dict(record.get("metadata") or {})
    metadata.update({
        "system_prompt_version": "m1-system-format-v4",
        "user_prompt_version": "m1-user-guidance-v3",
        "open_agri_v2_ablation_route": route,
    })
    result["metadata"] = metadata
    return result


def source_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    direct: list[dict[str, Any]] = []
    rag: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for domain, route in SOURCE_FILES:
        path = TRAIN_ROOT / f"{domain}_{route}.jsonl"
        rows = read_jsonl(path)
        sources[f"{domain}_{route}"] = {"path": str(path.relative_to(REPO_ROOT)), "rows": len(rows), "sha256": digest_file(path)}
        target = direct if route == "direct" else rag
        target.extend(render_record(row, route) for row in rows)
    if len(direct) != 708 or len(rag) != 1456:
        raise ValueError(f"unexpected formal v2 row counts: Direct={len(direct)}, RAG={len(rag)}")
    return direct, rag, sources


def route_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str((row.get("metadata") or {}).get("route")) for row in rows))


def write_view(root: Path, name: str, rows: list[dict[str, Any]], source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    view_root = root / "views" / name
    data_path = view_root / "data.jsonl"
    write_jsonl(data_path, rows)
    ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not item for item in ids):
        raise ValueError(f"{name}: missing sample ID")
    manifest = {
        "schema_version": "agrinet.open-agri-v2-sft-ablation-view/v1",
        "name": name,
        "formal_dataset": "datasets/AgriNet-1K/open_agri_v2",
        "source_views": source,
        "rows": len(rows),
        "route_counts": route_counts(rows),
        "unique_source_sample_ids": len(set(ids)),
        "data_sha256": digest_file(data_path),
        "system_prompt_sha256": digest_bytes(V4_SYSTEM.encode()),
        "open_user_prompt_sha256": {language: digest_bytes(value.encode()) for language, value in OPEN_USER.items()},
        "option_user_prefix_sha256": {language: digest_bytes(value.encode()) for language, value in OPTION_PREFIX.items()},
        "training_authorized": True,
        "authorization_type": "open-agri-v2-sft-ablation-view",
    }
    (view_root / "authorization.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true", help="Replace a previous generated artifact root.")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists():
        if not args.replace:
            raise SystemExit(f"output already exists: {output}; use --replace only after verifying it is disposable")
        shutil.rmtree(output)
    direct, rag, sources = source_rows()
    output.mkdir(parents=True)
    (output / "prompts").mkdir()
    (output / "prompts" / "direct_system_v4.txt").write_text(V4_SYSTEM + "\n", encoding="utf-8")
    for language, prompt in OPEN_USER.items():
        (output / "prompts" / f"open_user_v3_{language}.txt").write_text(prompt + "\n", encoding="utf-8")
    views = {
        "direct-only": write_view(output, "direct-only", direct, sources),
        "direct-rag": write_view(output, "direct-rag", direct + rag, sources),
        "direct3-rag": write_view(output, "direct3-rag", direct * 3 + rag, sources),
    }
    root_manifest = {
        "schema_version": "agrinet.open-agri-v2-sft-ablation/v1",
        "dataset": str(V2_ROOT.relative_to(REPO_ROOT)),
        "views": {name: {"rows": value["rows"], "route_counts": value["route_counts"], "data_sha256": value["data_sha256"]} for name, value in views.items()},
        "prompts": {"v4_system": V4_SYSTEM, "open_user_v3": OPEN_USER},
    }
    (output / "manifest.json").write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "views": root_manifest["views"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
