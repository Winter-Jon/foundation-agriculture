#!/usr/bin/env python3
"""Create canonical-v1 Direct-only, Direct+RAG, and Directx3+RAG SFT views."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/artifacts/open-agri-v2-canonical-v1-sft"
sys.path.insert(0, str(REPO_ROOT / "src"))
from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, tool_schema, validate_tool_arguments
from agrinet.research.open_agri_v2_canonical.cards import CANDIDATE_CARD_SCHEMA, card_from_legacy_hit
from agrinet.research.open_agri_v2_canonical.registry import CANONICAL_VERSION, Registry, load_registry

SYSTEM = "You identify agricultural diseases and pests from images. Return <think>...</think><answer>the requested canonical name only</answer>."
OPEN = {
    "en": "<image> Identify the image. Answer with only the canonical English name; include the explicit life stage when it is part of the label.",
    "zh": "<image> 识别图中病虫害。只输出规范中文名称；标签包含生命周期时必须明确写出生命周期。",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_options(registry: Registry, code: str, language: str) -> list[tuple[str, str]]:
    domain = registry.rows_by_code[code]["domain"]
    pool = [key for key, value in sorted(registry.rows_by_code.items()) if key != code and value["domain"] == domain]
    ranked = sorted(pool, key=lambda item: hashlib.sha256(f"{code}:{language}:{item}".encode()).hexdigest())[:3]
    options = [code, *ranked]
    options.sort(key=lambda item: hashlib.sha256(f"option-order:{code}:{language}:{item}".encode()).hexdigest())
    return [(item, registry.display_name(item, language)) for item in options]


def direct_rows(registry: Registry, limit_per_class: int) -> list[dict[str, Any]]:
    selected: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for image in read_jsonl(ROOT / "vlm_data/candidates/image_pool.jsonl"):
        if not image["sft_eligible"] or image["image_split"] != "train_candidate":
            continue
        code = str(image["canonical_class_code"])
        if selected[code] >= limit_per_class:
            continue
        selected[code] += 1
        for language in ("en", "zh"):
            answer = registry.display_name(code, language)
            sample_id = hashlib.sha256(f"direct:{image['image_sha256']}:{language}".encode()).hexdigest()[:24]
            rows.append({"schema_version": "agrinet.open-agri-v2-canonical-v1.ms-swift-agent/v1", "sample_id": "canonical-direct-" + sample_id, "images": [image["image_path"]], "tools": "[]", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": OPEN[language]}, {"role": "assistant", "content": f"<think>Identify the visible agricultural class.</think><answer>{answer}</answer>"}], "metadata": {"route": "direct", "question_type": "open", "language": language, "source_class_code": image["source_class_code"], "canonical_class_code": code, "image_sha256": image["image_sha256"], "taxonomy_version": CANONICAL_VERSION}})
            options = stable_options(registry, code, language)
            letters = "ABCD"
            correct_letter = letters[next(index for index, (option_code, _) in enumerate(options) if option_code == code)]
            rendered_options = "\n".join(f"{letter}. {name}" for letter, (_, name) in zip(letters, options, strict=True))
            option_prompt = ("<image> Select the canonical English name. Reply with only the option letter.\n" if language == "en" else "<image> 请选择规范中文名称。只输出选项字母。\n") + rendered_options
            option_id = hashlib.sha256(f"option:{image['image_sha256']}:{language}".encode()).hexdigest()[:24]
            rows.append({"schema_version": "agrinet.open-agri-v2-canonical-v1.ms-swift-agent/v1", "sample_id": "canonical-option-" + option_id, "images": [image["image_path"]], "tools": "[]", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": option_prompt}, {"role": "assistant", "content": f"<think>Compare the four canonical candidates.</think><answer>{correct_letter}</answer>"}], "metadata": {"route": "direct", "question_type": "option", "language": language, "source_class_code": image["source_class_code"], "canonical_class_code": code, "image_sha256": image["image_sha256"], "option_codes": [option_code for option_code, _ in options], "option_names": [name for _, name in options], "correct_option": correct_letter, "taxonomy_version": CANONICAL_VERSION}})
    return rows


def canonical_rag_rows(registry: Registry, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize retained historical RAG messages to current candidate cards.

    The source row must already be a canonical-v1 Known train image. Old tool
    response candidate names are resolved through the approved registry; rows
    with no public resolvable candidate are rejected rather than reconstructed.
    """
    pool = {str(row["image_sha256"]): row for row in read_jsonl(ROOT / "vlm_data/candidates/image_pool.jsonl")}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    sources = (
        REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2/vlm_data/accepted/train/disease_rag.jsonl",
        REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2/vlm_data/accepted/train/pest_rag.jsonl",
    )
    for source_path in sources:
      for source in read_jsonl(source_path):
        if limit and len(accepted) >= limit:
            break
        metadata = source.get("metadata") or {}
        image_sha = str(metadata.get("v2_image_sha256") or "")
        image = pool.get(image_sha)
        if not image or not image["sft_eligible"]:
            continue
        code = str(image["canonical_class_code"])
        language = str(metadata.get("language") or "en")
        messages = source.get("messages")
        cards = []
        if isinstance(messages, list):
            for message in messages:
                if message.get("role") not in {"tool", "tool_response"}:
                    continue
                try:
                    payload = json.loads(str(message.get("content") or "{}"))
                except json.JSONDecodeError:
                    continue
                for rank, hit in enumerate(payload.get("results") or [], 1):
                    if isinstance(hit, dict):
                        card = card_from_legacy_hit(registry, hit, rank)
                        if card and card["canonical_class_code"] not in {item["canonical_class_code"] for item in cards}:
                            cards.append(card)
        if not cards:
            audit.append({"sample_id": source.get("sample_id"), "status": "excluded", "reason": "no_resolvable_public_candidate_card"})
            continue
        call = {"name": TOOL_NAME, "arguments": {"query": "identify the visible agricultural disease or pest", "retrieval_type": "visual", "image": "query_image", "top_k": min(10, len(cards)), "rationale": "compare canonical public candidates against visible image evidence"}}
        response = {"schema_version": CANDIDATE_CARD_SCHEMA, "status": "success", "results": cards}
        answer = registry.display_name(code, language)
        rewritten = [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": OPEN[language]},
            {"role": "assistant", "content": "<think>Use public candidate cards to compare the visible evidence.</think>"},
            {"role": "tool_call", "content": json.dumps(call, ensure_ascii=False, separators=(",", ":"))},
            {"role": "tool_response", "content": json.dumps(response, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": f"<think>Choose the canonical class supported by the image and public candidates.</think><answer>{answer}</answer>"},
        ]
        try:
            normalized = normalize_training_messages(rewritten, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments, require_tool_calls=True)
        except ValueError as exc:
            audit.append({"sample_id": source.get("sample_id"), "status": "excluded", "reason": str(exc)})
            continue
        sample_id = hashlib.sha256(f"rag:{image_sha}:{language}:{source.get('sample_id')}".encode()).hexdigest()[:24]
        accepted.append({"schema_version": "agrinet.open-agri-v2-canonical-v1.ms-swift-agent/v1", "sample_id": "canonical-rag-" + sample_id, "images": [image["image_path"]], "tools": json.dumps([tool_schema()], ensure_ascii=False, separators=(",", ":")), "messages": normalized, "metadata": {"route": "rag", "question_type": "open", "language": language, "source_class_code": image["source_class_code"], "canonical_class_code": code, "image_sha256": image_sha, "taxonomy_version": CANONICAL_VERSION, "rag_candidate_card_schema": CANDIDATE_CARD_SCHEMA, "source_sample_id": source.get("sample_id")}})
        audit.append({"sample_id": source.get("sample_id"), "status": "accepted", "candidate_cards": len(cards), "canonical_class_code": code})
      if limit and len(accepted) >= limit:
          break
    return accepted, audit


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--review-preview", action="store_true", help="Render an explicitly non-trainable preview while manual taxonomy review is pending.")
    parser.add_argument("--direct-per-class", type=int, default=8)
    parser.add_argument("--rag-limit", type=int, default=0, help="0 keeps every canonical-v1-eligible retained RAG row.")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise ValueError(f"output exists: {args.output}; use --replace")
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    registry_path = ROOT / "taxonomy/canonical_label_registry.jsonl"
    approval_path = ROOT / "taxonomy/approval.json"
    registry = load_registry(registry_path, approval_path, require_approval=not args.review_preview)
    direct = direct_rows(registry, args.direct_per_class)
    rag, audit = canonical_rag_rows(registry, args.rag_limit)
    write_jsonl(args.output / "audits/rag_normalization.jsonl", audit)
    views = {
        "direct-only": direct,
        "direct-rag": direct + rag,
        "direct3-rag": direct * 3 + rag,
    }
    rendered = {}
    for name, rows in views.items():
        data = args.output / "views" / name / "data.jsonl"
        count = write_jsonl(data, rows)
        authorization = {"schema_version": "agrinet.open-agri-v2-canonical-v1-sft-view/v1", "authorization_type": "open-agri-v2-canonical-v1-approved-view", "taxonomy_version": CANONICAL_VERSION, "registry_sha256": registry.digest, "rows": count, "data_sha256": digest(data), "training_authorized": not args.review_preview, "release_status": "review_preview_only" if args.review_preview else "approved", "requires_registry_approval": True}
        (data.parent / "authorization.json").write_text(json.dumps(authorization, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rendered[name] = authorization
    manifest = {"schema_version": "agrinet.open-agri-v2-canonical-v1-sft/v1", "taxonomy_version": CANONICAL_VERSION, "registry_sha256": registry.digest, "registry": str(registry_path), "approval": str(approval_path), "release_status": "review_preview_only" if args.review_preview else "approved", "direct_rows": len(direct), "rag_rows": len(rag), "rag_audit": {"accepted": sum(row["status"] == "accepted" for row in audit), "excluded": sum(row["status"] == "excluded" for row in audit)}, "views": rendered}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
