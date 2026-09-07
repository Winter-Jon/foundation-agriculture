#!/usr/bin/env python3
"""Build a repaired OpenAgri v2 SFT successor without changing formal data.

This successor keeps both Open and Option examples. It freezes Open answers to
the catalog contract, normalizes RAG tool schemas/calls, and excludes any RAG
Open row that lacks public evidence for its supervised class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from agrinet.rag.tool_schema import TOOL_NAME, tool_schema, tools_json, validate_tool_arguments


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"
TRAIN_ROOT = V2_ROOT / "vlm_data/accepted/train"
CATALOG_PATH = REPO_ROOT / "datasets/AgriNet-1K/vlm_data/disease_pest/classes_all_disease_pest.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/artifacts/open-agri-v2-sft-ablation-v4-canonical-registry"

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
SOURCE_FILES = (("disease", "direct"), ("pest", "direct"), ("disease", "rag"), ("pest", "rag"))
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TOOL_ARGUMENT_KEYS = set(tool_schema()["function"]["parameters"]["properties"])

# This is a derived SFT/evaluation answer taxonomy only.  It does not edit the
# formal v2 catalog, public manifests, or private truth.  Chinese display-name
# collisions are intentionally represented by the lower source code.
MERGED_CLASS_CODES = {
    "N04031": "N04029",  # 樱桃健康叶片
    "N04065": "N04058",  # 葡萄褐斑病
    "N04082": "N04080",  # 石榴炭疽病
    "N04116": "N04111",  # 番茄早疫病
    "N05045": "N05017",  # 玉米螟
    "N05037": "N05022",  # 豆荚螟
}
# These pairs describe adult/larval stages.  They remain distinct canonical
# classes, while their historical duplicate English surfaces remain aliases.
LARVA_ENGLISH_NAMES = {
    "N05020": "cydia pomonella larva",
    "N05064": "spodoptera frugiperda larva",
}
# Public historical retrieval surfaces not present verbatim in the frozen
# catalog.  These resolve only to the stated canonical code.
PUBLIC_LEGACY_ALIASES = {
    "Grape Leaf blight": "N04058",
}


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


def normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[\u3000\s]+", " ", text)
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_final_answer(content: str, sample_id: str) -> tuple[str, tuple[int, int]]:
    matches = list(ANSWER_RE.finditer(content))
    if not matches:
        raise ValueError(f"{sample_id}: missing final answer tag")
    final = matches[-1]
    return final.group(1).strip(), final.span(1)


def replace_final_answer(content: str, replacement: str, sample_id: str) -> str:
    _, (start, end) = extract_final_answer(content, sample_id)
    return content[:start] + replacement + content[end:]


def extract_option_lines(content: str, sample_id: str) -> list[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    options = [line for line in lines if len(line) >= 3 and line[0] in "ABCD" and line[1] == "."]
    if len(options) != 4 or [line[0] for line in options] != list("ABCD"):
        raise ValueError(f"{sample_id}: option user message lacks exactly A-D candidates")
    return options


def rewritten_user(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    language, question_type = str(metadata.get("language") or ""), str(metadata.get("question_type") or "")
    if language not in OPEN_USER or question_type not in {"open", "option"}:
        raise ValueError(f"{record.get('sample_id')}: unsupported language/question type {language}/{question_type}")
    if question_type == "open":
        return OPEN_USER[language]
    old_user = next((str(m.get("content") or "") for m in record.get("messages") or [] if m.get("role") == "user"), "")
    return OPTION_PREFIX[language] + "\n" + "\n".join(extract_option_lines(old_user, str(record.get("sample_id"))))


def validate_source_record(record: dict[str, Any], expected_route: str) -> None:
    metadata = record.get("metadata") or {}
    sample_id = str(record.get("sample_id") or "<missing>")
    if metadata.get("route") != expected_route:
        raise ValueError(f"{sample_id}: route differs from source view")
    if metadata.get("class_role") != "known" or metadata.get("v2_image_split") != "train_candidate":
        raise ValueError(f"{sample_id}: not a formal v2 Known train_candidate row")
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{sample_id}: expected exactly one image")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{sample_id}: malformed message sequence")
    extract_final_answer(str(messages[-1].get("content") or ""), sample_id)
    has_tool_call = any(message.get("role") == "tool_call" for message in messages)
    if (expected_route == "rag") != has_tool_call:
        raise ValueError(f"{sample_id}: tool-call route contract violation")


def canonical_code(code: str) -> str:
    return MERGED_CLASS_CODES.get(code, code)


def build_registry() -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Build the user-approved derived answer taxonomy and public aliases."""
    catalog = {str(row["code"]): row for row in read_jsonl(CATALOG_PATH)}
    if set(MERGED_CLASS_CODES) - set(catalog) or set(MERGED_CLASS_CODES.values()) - set(catalog):
        raise ValueError("merged class code is missing from the frozen catalog")
    if set(LARVA_ENGLISH_NAMES) - set(catalog):
        raise ValueError("larval class code is missing from the frozen catalog")

    source_codes: dict[str, list[str]] = defaultdict(list)
    for code in catalog:
        source_codes[canonical_code(code)].append(code)
    registry: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for code, row in sorted(catalog.items()):
        if canonical_code(code) != code:
            continue
        sources = sorted(source_codes[code])
        canonical_en = LARVA_ENGLISH_NAMES.get(code, str(row["english_name"]).replace("_", " "))
        canonical_zh = str(row["chinese_name"])
        english_aliases = {canonical_en}
        chinese_aliases = {canonical_zh}
        for source_code in sources:
            source = catalog[source_code]
            english_aliases.add(str(source["english_name"]).replace("_", " "))
            chinese_aliases.add(str(source["chinese_name"]))
        registry[code] = {
            "code": code,
            "source_codes": sources,
            "english_name": str(row["english_name"]).replace("_", " "),
            "chinese_name": canonical_zh,
            "canonical_en": canonical_en,
            "canonical_zh": canonical_zh,
            "english_aliases": sorted(english_aliases),
            "chinese_aliases": sorted(chinese_aliases),
            "task_domain": str(row.get("task_domain") or ""),
        }
        for surface in english_aliases | chinese_aliases:
            aliases[normalized_name(surface)].add(code)
    for surface, target in PUBLIC_LEGACY_ALIASES.items():
        aliases[normalized_name(surface)].add(canonical_code(target))
    # A historical English duplicate may be a scorer alias for both an adult
    # and larval class.  Canonical rendering itself must remain one-to-one;
    # force these surfaces back to their declared canonical code for training
    # validation and evidence resolution.
    for code, item in registry.items():
        for language in ("en", "zh"):
            aliases[normalized_name(item[f"canonical_{language}"])] = {code}
    for code, item in registry.items():
        for language in ("en", "zh"):
            surface = item[f"canonical_{language}"]
            if aliases[normalized_name(surface)] != {code}:
                raise ValueError(f"{code}: canonical {language} surface is not unique")
    return registry, aliases


def resolve_answer(value: str, aliases: dict[str, set[str]], target_code: str) -> dict[str, Any]:
    codes = sorted(aliases.get(normalized_name(value), set()))
    if codes == [target_code]:
        status = "target"
    elif len(codes) == 1:
        status = "other"
    elif codes:
        status = "ambiguous"
    else:
        status = "unresolved"
    return {"text": value, "normalized": normalized_name(value), "codes": codes, "status": status}


def canonicalize_tool_calls(messages: list[dict[str, Any]], sample_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_messages: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        copied = dict(message)
        if copied.get("role") != "tool_call":
            canonical_messages.append(copied)
            continue
        try:
            payload = json.loads(str(copied.get("content") or ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{sample_id}: tool call {message_index} is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{sample_id}: tool call {message_index} is not an object")
        arguments = payload.get("arguments")
        errors: list[str] = []
        if payload.get("name") != TOOL_NAME:
            errors.append(f"name must be {TOOL_NAME}")
        if not isinstance(arguments, dict):
            errors.append("arguments must be an object")
        else:
            extra = sorted(set(arguments) - TOOL_ARGUMENT_KEYS)
            if extra:
                errors.append(f"unknown argument keys: {extra}")
            errors.extend(validate_tool_arguments(arguments))
        audit.append({"message_index": message_index, "valid": not errors, "errors": errors})
        if errors:
            raise ValueError(f"{sample_id}: invalid tool call {message_index}: {'; '.join(errors)}")
        copied["content"] = json.dumps(
            {"name": TOOL_NAME, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical_messages.append(copied)
    if not audit:
        raise ValueError(f"{sample_id}: RAG row lacks a tool call")
    return canonical_messages, audit


def evidence_codes(messages: list[dict[str, Any]], aliases: dict[str, set[str]]) -> tuple[set[str], list[str], int]:
    codes: set[str] = set()
    schema_versions: set[str] = set()
    unresolved_or_ambiguous_names = 0
    for message in messages:
        if message.get("role") != "tool_response":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool response is not JSON: {exc}") from exc
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("tool response results must be a list")
        result_keys = {key for result in results if isinstance(result, dict) for key in result}
        if {"class_name", "chinese_name"} & result_keys:
            schema_versions.add("legacy-candidate-card")
        if {"name", "name_zh"} & result_keys:
            schema_versions.add("current-candidate-card")
        if not result_keys:
            schema_versions.add("empty-or-error")
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("tool response result must be an object")
            for key in ("name", "name_zh", "class_name", "chinese_name"):
                value = result.get(key)
                if isinstance(value, str):
                    resolved = aliases.get(normalized_name(value), set())
                    if len(resolved) == 1:
                        codes.update(resolved)
                    elif value.strip():
                        unresolved_or_ambiguous_names += 1
    return codes, sorted(schema_versions), unresolved_or_ambiguous_names


def contract_language_ok(surface: str, language: str) -> bool:
    return bool(surface) and ((language == "zh" and bool(re.match(r"^[㐀-鿿]", surface))) or (language == "en" and bool(re.match(r"^[A-Za-z]", surface))))


def repair_record(
    source: dict[str, Any],
    route: str,
    registry: dict[str, dict[str, str]],
    aliases: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    validate_source_record(source, route)
    sample_id = str(source["sample_id"])
    metadata = dict(source.get("metadata") or {})
    question_type, language, source_code = (str(metadata.get(key) or "") for key in ("question_type", "language", "class_code"))
    target_code = canonical_code(source_code)
    if target_code not in registry:
        raise ValueError(f"{sample_id}: class code missing from catalog")
    original_answer, _ = extract_final_answer(str(source["messages"][-1].get("content") or ""), sample_id)
    original_resolution = resolve_answer(original_answer, aliases, target_code)
    result = dict(source)
    result_messages: list[dict[str, Any]] = []
    for message in source["messages"]:
        if message.get("role") == "system":
            result_messages.append({"role": "system", "content": V4_SYSTEM})
        elif message.get("role") == "user":
            result_messages.append({"role": "user", "content": rewritten_user(source)})
        else:
            result_messages.append(dict(message))

    tool_audit: list[dict[str, Any]] = []
    response_versions: list[str] = []
    supported = None
    exclusion_reasons: list[str] = []
    if route == "rag":
        result_messages, tool_audit = canonicalize_tool_calls(result_messages, sample_id)
        public_codes, response_versions, unresolved_evidence_names = evidence_codes(result_messages, aliases)
        if question_type == "open":
            supported = target_code in public_codes
            if not supported:
                exclusion_reasons.append("target_code_absent_from_public_retrieval")

    rewritten_answer = original_answer
    rewritten_resolution = original_resolution
    if question_type == "open":
        rewritten_answer = registry[target_code][f"canonical_{language}"]
        if not contract_language_ok(rewritten_answer, language):
            raise ValueError(f"{sample_id}: canonical answer language contract failed")
        rewritten_resolution = resolve_answer(rewritten_answer, aliases, target_code)
        if rewritten_resolution["status"] != "target":
            raise ValueError(f"{sample_id}: canonical answer does not resolve uniquely to target")
        result_messages[-1]["content"] = replace_final_answer(str(result_messages[-1].get("content") or ""), rewritten_answer, sample_id)

    metadata.update({
        "source_class_code": source_code,
        "class_code": target_code,
        "system_prompt_version": "m1-system-format-v4",
        "user_prompt_version": "m1-user-guidance-v3",
        "open_agri_v2_ablation_route": route,
        "answer_contract_version": "open-agri-v2-canonical-registry/v2-merged-zh-larva-en",
        "rag_tool_contract_version": "agrinet-rag-search/current-v1" if route == "rag" else None,
    })
    result["metadata"] = metadata
    result["messages"] = result_messages
    if route == "rag":
        result["tools"] = tools_json()

    action = "exclude" if exclusion_reasons else ("rewrite_answer" if rewritten_answer != original_answer else "keep")
    audit = {
        "sample_id": sample_id,
        "route": route,
        "question_type": question_type,
        "language": language,
        "source_class_code": source_code,
        "class_code": target_code,
        "original_answer": original_answer,
        "rewritten_answer": rewritten_answer,
        "original_answer_resolution": original_resolution,
        "rewritten_answer_resolution": rewritten_resolution,
        "tool_schema_hash": digest_bytes(tools_json().encode()) if route == "rag" else None,
        "tool_call_validation": tool_audit,
        "tool_response_schema_version": response_versions,
        "unresolved_or_ambiguous_evidence_names": unresolved_evidence_names if route == "rag" else 0,
        "evidence_support_pass": supported,
        "tool_contract_rewritten": route == "rag",
        "action": action,
        "exclusion_reasons": exclusion_reasons,
    }
    return (None if exclusion_reasons else result), audit


def source_rows(
    registry: dict[str, dict[str, str]], aliases: dict[str, set[str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    direct: list[dict[str, Any]] = []
    rag: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for domain, route in SOURCE_FILES:
        path = TRAIN_ROOT / f"{domain}_{route}.jsonl"
        rows = read_jsonl(path)
        sources[f"{domain}_{route}"] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": len(rows),
            "sha256": digest_file(path),
        }
        for source in rows:
            repaired, audit = repair_record(source, route, registry, aliases)
            audits.append(audit)
            if repaired is not None:
                (direct if route == "direct" else rag).append(repaired)
    if len(direct) != 708:
        raise ValueError(f"Direct rows changed unexpectedly: {len(direct)}")
    excluded_rag = sum(item["route"] == "rag" and item["action"] == "exclude" for item in audits)
    if len(rag) + excluded_rag != 1456:
        raise ValueError("RAG retention/exclusion accounting is inconsistent")
    return direct, rag, audits, sources


def route_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str((row.get("metadata") or {}).get("route")) for row in rows))


def validate_rows(
    rows: list[dict[str, Any]],
    expected_route_counts: dict[str, int],
    registry: dict[str, dict[str, str]],
    aliases: dict[str, set[str]],
) -> None:
    if route_counts(rows) != expected_route_counts:
        raise ValueError(f"unexpected route counts: {route_counts(rows)} != {expected_route_counts}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if not ids or any(not item for item in ids):
        raise ValueError("view contains missing sample IDs")
    for row in rows:
        metadata = row.get("metadata") or {}
        route = str(metadata.get("route") or "")
        if metadata.get("system_prompt_version") != "m1-system-format-v4":
            raise ValueError(f"{row['sample_id']}: system prompt metadata mismatch")
        if metadata.get("user_prompt_version") != "m1-user-guidance-v3":
            raise ValueError(f"{row['sample_id']}: user prompt metadata mismatch")
        final, _ = extract_final_answer(str(row["messages"][-1].get("content") or ""), str(row["sample_id"]))
        if metadata.get("question_type") == "open":
            resolution = resolve_answer(final, aliases, str(metadata["class_code"]))
            if resolution["status"] != "target":
                raise ValueError(f"{row['sample_id']}: Open answer is not unique target canonical")
            expected = registry[str(metadata["class_code"])][f"canonical_{metadata['language']}"]
            if final != expected:
                raise ValueError(f"{row['sample_id']}: Open answer differs from registry")
        if route == "rag":
            if row.get("tools") != tools_json():
                raise ValueError(f"{row['sample_id']}: noncanonical tools schema")
            calls = [message for message in row["messages"] if message.get("role") == "tool_call"]
            if not calls:
                raise ValueError(f"{row['sample_id']}: RAG row lacks tool call")
            for call in calls:
                payload = json.loads(str(call.get("content") or ""))
                if set(payload) != {"name", "arguments"} or payload["name"] != TOOL_NAME:
                    raise ValueError(f"{row['sample_id']}: tool call not canonical")
                errors = validate_tool_arguments(payload["arguments"])
                if errors or set(payload["arguments"]) - TOOL_ARGUMENT_KEYS:
                    raise ValueError(f"{row['sample_id']}: invalid retained tool arguments: {errors}")
            if metadata.get("question_type") == "open":
                public_codes, _, _ = evidence_codes(row["messages"], aliases)
                if metadata["class_code"] not in public_codes:
                    raise ValueError(f"{row['sample_id']}: RAG Open lacks target public evidence")


def write_view(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
    source: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, str]],
    aliases: dict[str, set[str]],
) -> dict[str, Any]:
    if name == "direct-only":
        expected = {"direct": 708}
    elif name == "direct-rag":
        expected = {"direct": 708, "rag": len(rows) - 708}
    elif name == "direct3-rag":
        expected = {"direct": 2124, "rag": len(rows) - 2124}
    else:
        raise ValueError(f"unknown view {name}")
    validate_rows(rows, expected, registry, aliases)
    view_root = root / "views" / name
    data_path = view_root / "data.jsonl"
    write_jsonl(data_path, rows)
    manifest = {
        "schema_version": "agrinet.open-agri-v2-sft-ablation-view/v3-canonical-registry",
        "name": name,
        "formal_dataset": "datasets/AgriNet-1K/open_agri_v2",
        "source_views": source,
        "rows": len(rows),
        "route_counts": route_counts(rows),
        "unique_source_sample_ids": len(set(str(row["sample_id"]) for row in rows)),
        "data_sha256": digest_file(data_path),
        "system_prompt_sha256": digest_bytes(V4_SYSTEM.encode()),
        "open_user_prompt_sha256": {language: digest_bytes(value.encode()) for language, value in OPEN_USER.items()},
        "option_user_prefix_sha256": {language: digest_bytes(value.encode()) for language, value in OPTION_PREFIX.items()},
        "canonical_registry_sha256": digest_file(root / "canonical_name_registry.jsonl"),
        "rag_tools_sha256": digest_bytes(tools_json().encode()),
        "training_authorized": True,
        "authorization_type": "open-agri-v2-sft-ablation-canonical-registry-view",
    }
    (view_root / "authorization.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists():
        raise SystemExit(f"output already exists and will not be replaced: {output}")

    registry, aliases = build_registry()
    direct, rag, audits, sources = source_rows(registry, aliases)
    output.mkdir(parents=True)
    write_jsonl(output / "canonical_name_registry.jsonl", (registry[code] for code in sorted(registry)))
    (output / "prompts").mkdir()
    (output / "prompts" / "direct_system_v4.txt").write_text(V4_SYSTEM + "\n", encoding="utf-8")
    for language, prompt in OPEN_USER.items():
        (output / "prompts" / f"open_user_v3_{language}.txt").write_text(prompt + "\n", encoding="utf-8")
    (output / "prompts" / "rag_tools_current.json").write_text(tools_json() + "\n", encoding="utf-8")
    write_jsonl(output / "audits" / "answer_contract_audit.jsonl", audits)
    views = {
        "direct-only": write_view(output, "direct-only", direct, sources, registry, aliases),
        "direct-rag": write_view(output, "direct-rag", direct + rag, sources, registry, aliases),
        "direct3-rag": write_view(output, "direct3-rag", direct * 3 + rag, sources, registry, aliases),
    }
    exclusion_counts = Counter(reason for audit in audits for reason in audit["exclusion_reasons"])
    root_manifest = {
        "schema_version": "agrinet.open-agri-v2-sft-ablation/v3-canonical-registry",
        "dataset": str(V2_ROOT.relative_to(REPO_ROOT)),
        "source_rows": {"direct": 708, "rag": 1456},
        "class_code_policy": {
            "version": "merged-zh-collisions-v1",
            "merged_source_to_canonical": MERGED_CLASS_CODES,
            "larva_english_canonical_names": LARVA_ENGLISH_NAMES,
            "legacy_public_aliases": PUBLIC_LEGACY_ALIASES,
        },
        "retained_rows": {"direct": len(direct), "rag": len(rag)},
        "excluded_rows": {"direct": 0, "rag": 1456 - len(rag), "reasons": dict(exclusion_counts)},
        "option_rows_preserved": sum(audit["question_type"] == "option" and audit["action"] != "exclude" for audit in audits),
        "views": {
            name: {"rows": value["rows"], "route_counts": value["route_counts"], "data_sha256": value["data_sha256"]}
            for name, value in views.items()
        },
        "prompts": {"v4_system": V4_SYSTEM, "open_user_v3": OPEN_USER},
        "rag_tools_sha256": digest_bytes(tools_json().encode()),
        "audit": {"path": "audits/answer_contract_audit.jsonl", "rows": len(audits)},
    }
    (output / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "views": root_manifest["views"], "excluded_rows": root_manifest["excluded_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
