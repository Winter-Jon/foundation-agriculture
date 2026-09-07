#!/usr/bin/env python3
"""Durable public-only Direct/RAG OpenAgri v2 inference with v4/Hermes prompts."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vlm/eval/tools"))
from agrinet.research.hcv.collector import compact_hit, image_url_content
from agrinet.rag.hermes_protocol import (
    canonicalize_hermes_tool_turn,
    parse_hermes_tool_calls,
    swift_hermes_system_prompt,
    swift_hermes_tool_response_message,
)
from agrinet.rag.tool_schema import TOOL_NAME, tool_schema, validate_tool_arguments
from eval_runner_common import SnapshotStore, load_jsonl, request_fingerprint, validate_manifest


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("direct", "rag"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--rag-api", default="")
    parser.add_argument("--max-tool-turns", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-concurrent", type=int, default=24)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Optional bounded prefix for a service smoke.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def chat(a: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    response = requests.post(
        f"{a.api_base.rstrip('/')}/chat/completions",
        json={"model": a.model, "messages": messages, "max_tokens": a.max_new_tokens, "temperature": 0.0},
        timeout=a.request_timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"].get("content") or "")


def retrieve(a: argparse.Namespace, image: Path, call: dict[str, Any]) -> dict[str, Any]:
    arguments = call["arguments"]
    body = {"top_k": int(arguments["top_k"]), "preset": str(arguments["retrieval_type"])}
    if str(arguments.get("query") or "").strip():
        body["text"] = str(arguments["query"]).strip()
    if arguments.get("image") == "query_image":
        body["image_path"] = str(image)
    endpoint = f"{a.rag_api.rstrip('/')}/search/{arguments['retrieval_type']}"
    response = requests.post(endpoint, json=body, timeout=a.request_timeout)
    response.raise_for_status()
    raw = response.json()
    hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or []
    refs: list[str] = []
    return {
        "status": "success", "retrieval_type": arguments["retrieval_type"], "query": arguments.get("query") or "",
        "results": [compact_hit(hit, refs) for hit in hits[:int(arguments["top_k"])]],
    }


def parse_call(content: str) -> list[dict[str, Any]]:
    """Accept the trained pure XML call and the served think-then-call variant.

    Hermes training records have a pure planning turn followed by a tool-call
    turn.  SGLang can concatenate those two adjacent assistant emissions into
    one response.  Recover exactly one schema-valid XML call without inventing
    any retrieval request; reject multiple or malformed XML envelopes.
    """
    strict = parse_hermes_tool_calls(content, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments)
    if strict:
        return strict
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
    if len(matches) != 1:
        return []
    try:
        call = json.loads(matches[0])
    except json.JSONDecodeError:
        return []
    if not isinstance(call, dict) or call.get("name") != TOOL_NAME or not isinstance(call.get("arguments"), dict):
        return []
    return [{"name": TOOL_NAME, "arguments": call["arguments"]}] if not validate_tool_arguments(call["arguments"]) else []


def one(a: argparse.Namespace, row: dict[str, Any], root: Path, system: str) -> dict[str, Any]:
    image = Path(str(row["image_path"]))
    image = image if image.is_absolute() else root / image
    if not image.is_file():
        raise FileNotFoundError(image)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": str(row["question"])}, image_url_content(image)]},
    ]
    history: list[dict[str, Any]] = []
    tool_turns = 0
    while True:
        prediction = chat(a, messages)
        if a.route == "direct":
            break
        calls = parse_call(prediction)
        if not calls:
            break
        if tool_turns >= a.max_tool_turns:
            prediction = "<answer></answer>"
            break
        canonical_prediction, canonical_calls = canonicalize_hermes_tool_turn(
            prediction, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments)
        # ``parse_call`` accepts a served think-then-call completion.  Its
        # accepted call must be renderable by the canonical Swift Hermes
        # formatter before it is allowed into future assistant history.
        if len(canonical_calls) != 1:
            prediction = "<answer></answer>"
            break
        call = canonical_calls[0]
        tool = retrieve(a, image, call)
        history.append({"tool_call": call, "tool_response": tool})
        messages.append({"role": "assistant", "content": canonical_prediction})
        messages.append(swift_hermes_tool_response_message(tool))
        tool_turns += 1
    return {
        "id": str(row["id"]), "language": row["language"], "route": a.route,
        "prediction": prediction, "tool_turns": tool_turns, "tool_history": history,
    }


async def run(a: argparse.Namespace) -> None:
    if a.route == "rag" and not a.rag_api:
        raise ValueError("--rag-api is required for RAG")
    manifest = Path(a.manifest)
    rows = load_jsonl(manifest)
    if a.limit:
        rows = rows[:a.limit]
    ids = validate_manifest(rows)
    base_system = Path(a.system_prompt_file).read_text(encoding="utf-8").strip()
    system = swift_hermes_system_prompt(base_system, [tool_schema()]) if a.route == "rag" else base_system
    fingerprint = request_fingerprint(
        manifest=manifest, protocol="agrinet.open-agri-v2-v4-hermes-eval/v1", model=a.model,
        parameters={"route": a.route, "system": system, "top_k": a.top_k, "max_tool_turns": a.max_tool_turns, "limit": a.limit,
                    "max_new_tokens": a.max_new_tokens, "temperature": 0.0},
    )
    store = SnapshotStore(Path(a.output), fingerprint, resume=a.resume)
    completed = store.completed(set(ids))
    lock = asyncio.Lock(); semaphore = asyncio.Semaphore(a.max_concurrent); count = len(completed)
    root = Path(a.repo_root).resolve()

    async def task(row: dict[str, Any]) -> None:
        nonlocal count
        item_id = str(row["id"])
        if item_id in completed:
            return
        result: dict[str, Any]
        async with semaphore:
            try:
                error = None
                for attempt in range(a.request_retries + 1):
                    try:
                        result = await asyncio.to_thread(one, a, row, root, system)
                        result["request_attempts"] = attempt + 1
                        break
                    except Exception as exc:
                        error = exc
                else:
                    raise RuntimeError(f"request failed after {a.request_retries + 1} attempts: {error}")
            except Exception as exc:
                result = {"id": item_id, "language": row["language"], "route": a.route, "prediction": "", "error": str(exc), "tool_turns": 0, "tool_history": []}
        async with lock:
            completed[item_id] = result; store.append(result); count += 1
            print(f"[{count}/{len(rows)}] {item_id}", flush=True)

    await asyncio.gather(*(task(row) for row in rows))
    store.finalize(ids, completed)


if __name__ == "__main__":
    asyncio.run(run(args()))
