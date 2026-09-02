#!/usr/bin/env python3
"""Prepare and validate the fixed three-seed RAG evidence protocol for open_agri_v3.

The manifest deliberately joins only v2 public test input with public class
names. Private truth is never passed to the evaluator; it is used later by the
offline aggregator. One evaluator output and durable snapshot is reserved per
seed, making interruption/recovery explicit instead of silently resampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/artifacts/datasets/open-agri-v3-rag-role-evidence-v1"
EVALUATOR = REPO_ROOT / "vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py"
SEEDS = (20260921, 20260922, 20260923)
MODEL = "models/Qwen3-VL-4B-Instruct"
PROTOCOL = "agrinet.hermes-rag-sglang-async/v5-native-json-multi-query-recovery-strict-similar-classes"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must provide exactly three unique fixed seeds")
    return seeds


def build_manifest(v2_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = read_jsonl(v2_root / "vlm_data/accepted/test_public.jsonl")
    image_rows = {str(row["image_sha256"]): row for row in read_jsonl(v2_root / "manifests/images.jsonl")}
    if len(public) != 1019 or len({str(row.get("id") or "") for row in public}) != len(public):
        raise ValueError("expected a unique 1,019-row v2 public test manifest")
    rows: list[dict[str, Any]] = []
    for item in public:
        item_id = str(item["id"])
        image = image_rows.get(str(item["image_sha256"]))
        if image is None or image.get("image_split") != "test":
            raise ValueError(f"missing public test image entry for {item_id}")
        # All fields in this row are public evaluator input. In particular do
        # not add code, answer, candidates, aliases, or test role/truth here.
        rows.append({
            "id": item_id,
            "image_path": str(item["images"][0]),
            "image_sha256": str(item["image_sha256"]),
            "task_domain": str((item.get("metadata") or {}).get("domain") or image["domain"]),
            "language": "en",
            "question_type": "open",
            "question": "Identify the agricultural disease or pest shown in this image. Use retrieval evidence before giving the final answer.",
            "public_input_schema": "open-agri-v3-rag-evidence-public/v1",
        })
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("evidence manifest IDs must be unique")
    inventory = {
        "v2_test_public_sha256": sha256(v2_root / "vlm_data/accepted/test_public.jsonl"),
        "v2_images_sha256": sha256(v2_root / "manifests/images.jsonl"),
        "test_rows": len(rows),
    }
    return rows, inventory


def prepare(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    manifest_path = output / "manifests/rag_evidence_public_test.jsonl"
    rows, inventory = build_manifest(args.v2_root.resolve())
    if manifest_path.exists() and not args.replace:
        if sha256(manifest_path) != hashlib.sha256("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()).hexdigest():
            raise ValueError(f"existing manifest differs: {manifest_path}; use --replace after review")
    else:
        write_jsonl(manifest_path, rows)
    run_plan = {
        "schema_version": "agrinet.open-agri-v3.rag-role-evidence/v1",
        "protocol_version": PROTOCOL,
        "model": MODEL,
        "fixed_generation": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512},
        "fixed_rag": {"top_k": 3, "max_tool_turns": 5, "invalid_tool_call_policy": "strict", "forced_first_visual_retrieval": True},
        "seeds": list(args.seeds),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "inventory": inventory,
        "privacy": "public input manifest excludes class_code, answer, candidates, aliases, and all private truth",
        "test_informed_protocol": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_plan.json").write_text(json.dumps(run_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def execute(args: argparse.Namespace, manifest: Path) -> None:
    for seed in args.seeds:
        seed_root = args.output / "predictions" / f"seed-{seed}"
        command = [
            str(args.python), str(EVALUATOR), "--manifest", str(manifest), "--output", str(seed_root / "predictions.jsonl"),
            "--repo-root", str(REPO_ROOT), "--model", args.model, "--api-base", args.api_base, "--rag-api", args.rag_api,
            "--seed", str(seed), "--temperature", "0.7", "--top-p", "0.9", "--max-new-tokens", "512",
            "--top-k", "3", "--max-tool-turns", "5", "--invalid-tool-call-policy", "strict", "--terminal-errors", "--request-retries", "0",
        ]
        if args.resume:
            command.append("--resume")
        if args.max_concurrent:
            command.extend(["--max-concurrent", str(args.max_concurrent)])
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run all three seeds after preparing the public manifest.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--rag-api", default="http://127.0.0.1:8077")
    parser.add_argument("--max-concurrent", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(args)
    if args.run:
        execute(args, manifest)
    print(json.dumps({"manifest": str(manifest), "run": args.run, "seeds": list(args.seeds)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
