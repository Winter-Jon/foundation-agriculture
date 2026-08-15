"""Shared catalog and image-isolation helpers for active RAG SFT builders.

The helpers deliberately read historic *artifacts* as audit evidence, but no
active builder imports a historical round-specific implementation.  They do
not call a teacher, Milvus, or mutate a corpus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ROOT / "outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(ROOT))
        except ValueError:
            return str(candidate)
    return str(candidate)


def evaluation_images() -> set[str]:
    images: set[str] = set()
    for path in (ROOT / "outputs").rglob("manifest.jsonl"):
        if "vlm_eval" not in path.parts and "evaluation" not in path.parts:
            continue
        for row in read_jsonl(path):
            image = normalized(row.get("image_path") or row.get("query_image"))
            if image:
                images.add(image)
    return images


def audited_catalog() -> dict[str, dict[str, str]]:
    """Read class metadata from bounded data and existing audited plans."""
    catalog: dict[str, dict[str, str]] = {}
    for entry in read_jsonl(CLASSES):
        catalog[entry["code"]] = {
            "code": entry["code"], "english_name": entry["english_name"],
            "chinese_name": entry["chinese_name"], "task_domain": entry["task_domain"],
        }
    plans = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan"
    for path in plans.rglob("*.jsonl"):
        for row in read_jsonl(path):
            code = row.get("class_code")
            if code and row.get("class_name") and row.get("class_name_zh") and row.get("task_domain"):
                catalog.setdefault(code, {"code": code, "english_name": row["class_name"], "chinese_name": row["class_name_zh"], "task_domain": row["task_domain"]})
            for label in row.get("candidate_labels") or []:
                code = label.get("code")
                english, chinese, domain = label.get("name") or label.get("english_name"), label.get("chinese_name"), label.get("task_domain")
                if code and english and chinese and domain:
                    catalog.setdefault(code, {"code": code, "english_name": english, "chinese_name": chinese, "task_domain": domain})
    return catalog


def candidate_labels(classes: dict[str, dict[str, str]], code: str, domain: str, correct_option: str) -> list[dict[str, str]]:
    pool = sorted((entry for entry in classes.values() if entry["task_domain"] == domain and entry["code"] != code), key=lambda entry: entry["code"])
    labels = pool[:3]
    labels.insert(ord(correct_option) - ord("A"), classes[code])
    return [{"code": entry["code"], "name": entry["english_name"], "chinese_name": entry["chinese_name"], "task_domain": entry["task_domain"]} for entry in labels]


def sample_image(row: dict[str, Any]) -> str | None:
    sample = row.get("sample") or row.get("trace", {}).get("sample") or row
    metadata, images = row.get("metadata") or {}, row.get("images") or []
    return normalized(sample.get("query_image") or metadata.get("query_image") or (images[0] if images else None))


def exposed_images() -> set[str]:
    images = set(evaluation_images())
    for path in (ROOT / "outputs/experiments/rag_sft_iteration/rounds").rglob("plan.jsonl"):
        for row in read_jsonl(path):
            if image := sample_image(row):
                images.add(image)
    for path in (ROOT / "outputs/experiments/rag_sft_iteration/candidates").rglob("*.jsonl"):
        if path.name in {"raw_trajectories.jsonl", "agent_sft.accepted.jsonl"}:
            for row in read_jsonl(path):
                if image := sample_image(row):
                    images.add(image)
    for path in (ROOT / "outputs/artifacts/datasets").glob("agrinet-rag-*/data.jsonl"):
        for row in read_jsonl(path):
            if image := sample_image(row):
                images.add(image)
    return images


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
