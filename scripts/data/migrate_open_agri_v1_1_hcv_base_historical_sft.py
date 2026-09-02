#!/usr/bin/env python3
"""Migrate audited historical HCV SFT into canonical ms-swift Agent records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
SCHEMA_VERSION = "agrinet.open-agri-v2.ms-swift-agent/v1"
CANONICAL_SYSTEM = (
    "You identify agricultural diseases and pests from images. Follow the user "
    "task constraints. Return the final result as <think>...</think><answer>...</answer>."
)

# Source priority also gives the deterministic duplicate winner. v11 and
# v12-direct-only are lineage ancestors fully subsumed by the v12 anchor.
SOURCES: tuple[dict[str, str], ...] = (
    {
        "artifact_id": "agrinet-hcv-manual-json-v12-direct-anchor-mix",
        "path": "outputs/artifacts/agrinet-hcv-manual-json-v12-direct-anchor-mix/data.jsonl",
        "tier": "mainline", "disposition": "migrate",
        "notes": "Representative of the v11/v12 Direct lineage.",
    },
    {
        "artifact_id": "agrinet-hermes-current-contract-v3",
        "path": "outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/sft-hermes-current-contract-v3/data.jsonl",
        "tier": "mainline", "disposition": "migrate",
        "notes": "Historical Hermes rendering; normalize to framework-neutral messages.",
    },
    {
        "artifact_id": "agrinet-hermes-native-json-current-contract-v4",
        "path": "outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/sft-hermes-native-json-current-contract-v4/data.jsonl",
        "tier": "mainline", "disposition": "migrate",
        "notes": "Historical bare-JSON rendering; normalize to framework-neutral messages.",
    },
    {
        "artifact_id": "agrinet-reconstructive-direct-blind-intermediate-v1",
        "path": "outputs/artifacts/datasets/agrinet-reconstructive-direct-blind-intermediate-v1/sft-hermes/data.jsonl",
        "tier": "supplementary", "disposition": "migrate",
        "notes": "Audited reconstructive material; not automatically promoted to a mainline view.",
    },
    {
        "artifact_id": "agrinet-hcv-manual-json-v12-direct-only",
        "path": "outputs/artifacts/agrinet-hcv-manual-json-v12-direct-only/data.jsonl",
        "tier": "lineage_only", "disposition": "lineage_only",
        "notes": "All eligible supervision is represented by the v12 anchor mix.",
    },
    {
        "artifact_id": "agrinet-hcv-manual-json-v11-terminal-closure",
        "path": "outputs/artifacts/agrinet-hcv-manual-json-v11-terminal-closure/data.jsonl",
        "tier": "lineage_only", "disposition": "lineage_only",
        "notes": "All eligible supervision is represented by the v12 anchor mix.",
    },
)


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def compact_json(value: Any) -> str:
    return json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json_string(value: Any, *, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field}_invalid_json") from exc
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record is not an object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def source_path_keys(value: str) -> set[str]:
    """Return lookup forms for old relative and current logical image paths."""
    path = Path(value)
    keys = {value}
    if path.is_absolute():
        keys.add(str(path.resolve()))
    else:
        keys.add(str((REPO_ROOT / path).resolve()))
    return keys


def load_image_pool(root: Path) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(root / "vlm_data/candidates/image_pool.jsonl"):
        for value in (row.get("source_path"), row.get("image_path")):
            if isinstance(value, str):
                for key in source_path_keys(value):
                    pool[key] = row
    return pool


def normalize_tools(value: Any, *, has_tool_calls: bool) -> str:
    parsed = load_json_string(value if value is not None else "[]", field="tools")
    if not isinstance(parsed, list):
        raise ValueError("tools_not_list")
    if has_tool_calls and not parsed:
        raise ValueError("tool_calls_missing_tools")
    return compact_json(parsed)


def normalize_tool_response_content(content: str) -> str:
    """Convert an unambiguous legacy JSON-plus-terminal-instruction response.

    Historical terminal-rejection rows encoded a valid JSON tool state followed
    by a plain-language instruction. ms-swift requires the complete tool
    response content to be JSON, so retain the instruction in a named field.
    Other malformed content remains rejected rather than guessed.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        try:
            parsed, end = decoder.raw_decode(content.lstrip())
        except json.JSONDecodeError as exc:
            raise ValueError("tool_response_invalid_json") from exc
        trailing = content.lstrip()[end:].strip()
        if not isinstance(parsed, dict) or not trailing:
            raise ValueError("tool_response_invalid_json")
        if "terminal_instruction" in parsed:
            raise ValueError("tool_response_ambiguous_terminal_instruction")
        parsed = {**parsed, "terminal_instruction": trailing}
    if not isinstance(parsed, dict):
        raise ValueError("tool_response_content_not_object")
    return compact_json(parsed)


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Drop source rendering system prompts and normalize structural turns."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages_missing")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message_not_object")
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            role = "tool_response"
        if role not in {"user", "assistant", "tool_call", "tool_response"}:
            raise ValueError(f"unsupported_role:{role}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{role}_content_not_string")
        if role == "tool_call":
            parsed = load_json_string(content, field=role)
            if not isinstance(parsed, dict):
                raise ValueError(f"{role}_content_not_object")
            content = compact_json(parsed)
        elif role == "tool_response":
            content = normalize_tool_response_content(content)
        normalized.append({"role": role, "content": content})
    if not normalized or normalized[0]["role"] != "user":
        raise ValueError("first_message_not_user")
    if normalized[-1]["role"] != "assistant" or not normalized[-1]["content"].strip():
        raise ValueError("final_message_not_assistant")
    for index, message in enumerate(normalized):
        if message["role"] == "tool_response" and (
            index == 0 or normalized[index - 1]["role"] != "tool_call"
        ):
            raise ValueError("tool_response_not_adjacent_to_tool_call")
    return normalized


def ensure_query_image(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Make the single image placeholder explicit for all historic prompt styles."""
    first = dict(messages[0])
    content = first["content"]
    count = content.count("<image>")
    if count == 0:
        first["content"] = "<image>\n" + content
    elif count != 1:
        raise ValueError("query_image_placeholder_count")
    return [first, *messages[1:]]


def infer_task_metadata(
    row: dict[str, Any], messages: list[dict[str, str]], image: dict[str, Any]
) -> dict[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    user_message = next((message for message in messages if message["role"] == "user"), None)
    if user_message is None:
        raise ValueError("messages_missing_user")
    user_content = user_message["content"]
    task_domain = str(metadata.get("task_domain") or image["domain"])
    question_type = str(
        metadata.get("question_type")
        or ("option" if re.search(r"(?m)^A\. ", user_content) else "open")
    )
    language = str(
        metadata.get("language")
        or ("zh" if re.search(r"[\u4e00-\u9fff]", user_content) else "en")
    )
    route = "rag" if any(message["role"] == "tool_call" for message in messages) else "direct"
    return {
        "route": route,
        "question_type": question_type,
        "language": language,
        "task_domain": task_domain,
        "source_sample_id": str(row.get("sample_id") or ""),
    }


def supervision_fingerprint(
    image_path: str, tools: str, messages: list[dict[str, str]], task: dict[str, str]
) -> str:
    payload = {
        "query_image": image_path,
        "tools": tools,
        "messages": messages,
        "task": {key: task[key] for key in ("route", "question_type", "language", "task_domain")},
    }
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def source_row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(compact_json(row).encode("utf-8")).hexdigest()


def normalize_record(
    row: dict[str, Any], *, source: dict[str, str], source_sha256: str,
    pool: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("not_single_query_image")
    source_image = images[0]
    image = next((pool[key] for key in source_path_keys(source_image) if key in pool), None)
    if image is None:
        raise ValueError("unmapped_image")
    if image.get("class_role") != "known":
        raise ValueError("unknown_class")
    if image.get("image_split") != "train_candidate" or not image.get("sft_eligible"):
        raise ValueError("v2_non_train_candidate_image")
    messages = [
        {"role": "system", "content": CANONICAL_SYSTEM},
        *ensure_query_image(normalize_messages(row.get("messages"))),
    ]
    has_tool_calls = any(message["role"] == "tool_call" for message in messages)
    tools = normalize_tools(row.get("tools"), has_tool_calls=has_tool_calls)
    details = infer_task_metadata(row, messages, image)
    fingerprint = supervision_fingerprint(str(image["image_path"]), tools, messages, details)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"historical-{fingerprint[:20]}",
        "images": [image["image_path"]],
        "tools": tools,
        "messages": messages,
        "metadata": {
            **details,
            "class_code": image["class_code"],
            "class_role": image["class_role"],
            "v2_image_sha256": image["image_sha256"],
            "v2_image_split": image["image_split"],
            "open_agri_v2_supervision_fingerprint": fingerprint,
            "normalization": {
                "schema": "ms-swift-agent-support/v1",
                "source_artifact_id": source["source_artifact_id"],
                "source_jsonl": source["source_jsonl"],
                "source_data_sha256": source_sha256,
                "source_row_sha256": source_row_digest(row),
                "source_image_path": source_image,
            },
        },
    }


def registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for priority, source in enumerate(SOURCES):
        path = REPO_ROOT / source["path"]
        rows.append({
            "source_artifact_id": source["artifact_id"],
            "source_jsonl": source["path"],
            "tier": source["tier"],
            "disposition": source["disposition"],
            "priority": priority,
            "notes": source["notes"],
            "source_exists": path.is_file(),
            "source_data_sha256": sha256_file(path) if path.is_file() else None,
        })
    return rows


def migrate(root: Path, *, replace: bool) -> dict[str, Any]:
    output = root / "vlm_data/historical"
    if output.exists() and any(output.iterdir()) and not replace:
        raise ValueError(f"historical output already exists: {output}; use --replace after review")
    pool = load_image_pool(root)
    registry = registry_rows()
    missing = [row["source_artifact_id"] for row in registry if not row["source_exists"]]
    if missing:
        raise FileNotFoundError(f"historical source missing: {', '.join(missing)}")

    winners: dict[str, dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for source in registry:
        path = REPO_ROOT / str(source["source_jsonl"])
        source_id = str(source["source_artifact_id"])
        for line_no, row in enumerate(read_jsonl(path), start=1):
            source_counts[source_id] += 1
            audit_base = {
                "source_artifact_id": source_id,
                "source_jsonl": source["source_jsonl"],
                "source_line": line_no,
                "source_sample_id": str(row.get("sample_id") or ""),
                "source_row_sha256": source_row_digest(row),
            }
            if source["disposition"] != "migrate":
                audit.append({**audit_base, "status": "not_imported", "reason": "lineage_only_source"})
                continue
            try:
                item = normalize_record(
                    row, source=source, source_sha256=str(source["source_data_sha256"]), pool=pool
                )
            except ValueError as exc:
                audit.append({**audit_base, "status": "excluded", "reason": str(exc)})
                continue
            fingerprint = item["metadata"]["open_agri_v2_supervision_fingerprint"]
            prior = winners.get(fingerprint)
            if prior is not None:
                audit.append({
                    **audit_base,
                    "status": "excluded",
                    "reason": "duplicate_supervision_fingerprint",
                    "fingerprint": fingerprint,
                    "kept_sample_id": prior["sample_id"],
                    "kept_source_artifact_id": prior["metadata"]["normalization"]["source_artifact_id"],
                })
                continue
            winners[fingerprint] = item
            accepted.append(item)
            audit.append({
                **audit_base,
                "status": "accepted",
                "reason": "canonical_ms_swift_agent_record",
                "fingerprint": fingerprint,
                "canonical_sample_id": item["sample_id"],
                "route": item["metadata"]["route"],
            })

    accepted.sort(key=lambda item: item["sample_id"])
    direct = [item for item in accepted if item["metadata"]["route"] == "direct"]
    rag = [item for item in accepted if item["metadata"]["route"] == "rag"]
    if len(direct) + len(rag) != len(accepted):
        raise AssertionError("canonical records have an unknown route")

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "canonical").mkdir(parents=True)
    (staging / "audits").mkdir()
    write_jsonl(staging / "canonical/direct.jsonl", direct)
    write_jsonl(staging / "canonical/rag.jsonl", rag)
    write_jsonl(staging / "registry.jsonl", registry)
    write_jsonl(staging / "audits/migration.jsonl", audit)
    summary = {
        "schema_version": "agrinet.open-agri-v2.historical-migration/v1",
        "selection_protocol": (
            "direct raw-source canonicalization against the final dataset image pool; "
            "do not inherit a prior version's Known/Unknown-filtered canonical view"
        ),
        "canonical_message_schema": "ms-swift-agent-support/v1",
        "canonical_message_contract": {
            "tools": "JSON string",
            "tool_call": "role with JSON-string content",
            "tool_response": "role with JSON-string content",
            "agent_template": "selected only at training/inference rendering time",
        },
        "sources": {source_id: source_counts[source_id] for source_id in sorted(source_counts)},
        "accepted": len(accepted),
        "direct": len(direct),
        "rag": len(rag),
        "excluded": sum(row["status"] == "excluded" for row in audit),
        "lineage_only": sum(row["reason"] == "lineage_only_source" for row in audit),
        "files": {
            "direct": "vlm_data/historical/canonical/direct.jsonl",
            "rag": "vlm_data/historical/canonical/rag.jsonl",
            "registry": "vlm_data/historical/registry.jsonl",
            "audit": "vlm_data/historical/audits/migration.jsonl",
        },
    }
    (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    formal_summary_path = root / "manifests/summary.json"
    if formal_summary_path.is_file():
        formal_summary = json.loads(formal_summary_path.read_text(encoding="utf-8"))
        formal_summary.setdefault("counts", {})["historical_canonical"] = {
            "direct": len(direct), "rag": len(rag),
        }
        formal_summary["historical_supervision"] = {
            "selection_protocol": summary["selection_protocol"],
            "accepted": len(accepted),
            "direct": len(direct),
            "rag": len(rag),
            "historical_summary": "vlm_data/historical/summary.json",
        }
        temporary_summary = formal_summary_path.with_name(formal_summary_path.name + ".tmp")
        temporary_summary.write_text(
            json.dumps(formal_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_summary, formal_summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = migrate(args.dataset_root.resolve(), replace=args.replace)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
