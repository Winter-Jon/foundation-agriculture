#!/usr/bin/env python3
"""Build open_agri_v3 from the approved canonical-v1 taxonomy and supplement evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, tools_json, validate_tool_arguments
from agrinet.research.open_agri_v2_canonical.registry import Registry, load_registry, registry_digest

DEFAULT_V1 = ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
DEFAULT_OUTPUT = ROOT / "datasets/AgriNet-1K/open_agri_v3"
DEFAULT_LEGACY = ROOT / "datasets/AgriNet-1K/open_agri_v2/vlm_data/accepted/train"
DEFAULT_SUPPLEMENT = ROOT / "outputs/artifacts/datasets/open-agri-v2-known-supplement-v1"
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
OPTION_RE = re.compile(r"^([ABCD])\.\s*(.+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_span(content: str, sample_id: str) -> tuple[str, tuple[int, int]]:
    matches = list(ANSWER_RE.finditer(content))
    if len(matches) != 1:
        raise ValueError(f"{sample_id}: expected exactly one final answer")
    match = matches[0]
    return match.group(1).strip(), match.span(1)


def replace_answer(content: str, replacement: str, sample_id: str) -> str:
    _, (start, end) = answer_span(content, sample_id)
    return content[:start] + replacement + content[end:]


def metadata_code(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("canonical_class") or metadata.get("label_code") or metadata.get("class_code") or "")


def metadata_sha(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("image_sha256") or metadata.get("v2_image_sha256") or "")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def terminal_kind(row: dict[str, Any]) -> str | None:
    messages = row.get("messages") or []
    if not any("tool_budget_exhausted" in str(item.get("content") or "") for item in messages if isinstance(item, dict)):
        return None
    calls = sum(item.get("role") == "tool_call" for item in messages if isinstance(item, dict))
    return "five_turn" if calls == 6 else "short_budget"


def canonical_code(registry: Registry, source: str) -> str:
    return registry.canonical_for_source(source)


def canonical_name(registry: Registry, code: str, language: str) -> str:
    return registry.display_name(code, language)


def resolve_name(registry: Registry, value: Any) -> str | None:
    resolved = registry.resolve_answer(value)
    if resolved:
        return resolved
    # Historical option lists sometimes append an explanatory synonym in
    # parentheses.  It is not a distinct label; resolve its leading legacy
    # surface and then render the reviewed canonical display name.
    if isinstance(value, str):
        leading = re.sub(r"\s*\([^)]*\)\s*$", "", value.replace("_", " ")).strip()
        if leading and leading != value:
            return registry.resolve_answer(leading)
    return None


def canonicalize_payload(value: Any, registry: Registry) -> Any:
    """Rewrite structured public name fields only; reasoning prose remains verbatim."""
    if isinstance(value, list):
        return [canonicalize_payload(item, registry) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonicalize_payload(item, registry) for key, item in value.items()}
    for key, language in (("name", "en"), ("class_name", "en"), ("english_name", "en"),
                          ("name_zh", "zh"), ("chinese_name", "zh"), ("canonical_chinese_name", "zh")):
        if isinstance(result.get(key), str):
            code = resolve_name(registry, result[key])
            if code:
                result[key] = canonical_name(registry, code, language)
    return result


def rewrite_options(
    content: str, registry: Registry, language: str, sample_id: str,
    *, target_code: str, correct_letter: str,
) -> str:
    lines = content.splitlines()
    options: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        match = OPTION_RE.match(line.strip())
        if match:
            options.append((index, match.group(1), match.group(2)))
    if [letter for _, letter, _ in options] != list("ABCD"):
        raise ValueError(f"{sample_id}: expected exactly A-D options")
    for index, letter, label in options:
        # Historic adult/larva aliases can spell the correct target as its
        # adult surface. The answer letter and private target code determine
        # that position unambiguously; render it with the reviewed target name.
        code = target_code if letter == correct_letter else resolve_name(registry, label)
        if not code:
            raise ValueError(f"{sample_id}: option {letter} is not a canonical alias")
        lines[index] = f"{letter}. {canonical_name(registry, code, language)}"
    return "\n".join(lines)


def normalize_messages(messages: Any, *, rag: bool) -> list[dict[str, Any]]:
    if rag:
        return normalize_training_messages(messages, tool_name=TOOL_NAME,
                                           validate_arguments=validate_tool_arguments,
                                           require_tool_calls=True)
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Direct record must contain exactly system, user, assistant")
    if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("Direct message roles are invalid")
    answer_span(str(messages[-1].get("content") or ""), "direct")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


def convert_record(
    source: dict[str, Any], *, registry: Registry, source_path: Path, source_line: int,
    source_kind: str, allow_short_budget: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Convert one source record, returning an audit record for every decision."""
    sample_id = str(source.get("sample_id") or "<missing>")
    metadata = dict(source.get("metadata") or {})
    route = str(metadata.get("route") or "")
    language = str(metadata.get("language") or "")
    question_type = str(metadata.get("question_type") or "")
    digest = metadata_sha(source)
    source_class = metadata_code(source)
    audit: dict[str, Any] = {
        "sample_id": sample_id, "source_file": display_path(source_path),
        "source_line": source_line, "source_kind": source_kind,
        "source_sha256": sha256(source_path), "image_sha256": digest,
        "route": route, "language": language, "question_type": question_type,
        "source_class_code": source_class, "action": "reject", "reasons": [],
    }
    try:
        if route not in {"direct", "rag"} or language not in {"en", "zh"} or question_type not in {"open", "option"}:
            raise ValueError("unsupported route/language/question_type")
        if not digest or not source_class:
            raise ValueError("missing image sha256 or class code")
        code = canonical_code(registry, source_class)
        if route == "rag" and terminal_kind(source) == "short_budget" and not allow_short_budget:
            audit["reasons"] = ["short_budget_terminal_isolated"]
            audit["canonical_class_code"] = code
            audit["terminal_kind"] = "short_budget"
            audit["action"] = "isolate"
            return None, audit
        messages = normalize_messages(source.get("messages"), rag=route == "rag")
        converted: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if item["role"] == "tool":
                item["content"] = json.dumps(canonicalize_payload(json.loads(item["content"]), registry),
                                             ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            converted.append(item)
        user_index = next(index for index, item in enumerate(converted) if item["role"] == "user")
        final_index = len(converted) - 1
        if question_type == "option":
            answer, _ = answer_span(converted[final_index]["content"], sample_id)
            if answer not in "ABCD" or len(answer) != 1:
                raise ValueError("option final answer is not one letter")
            converted[user_index]["content"] = rewrite_options(
                converted[user_index]["content"], registry, language, sample_id,
                target_code=code, correct_letter=answer,
            )
        else:
            converted[final_index]["content"] = replace_answer(
                converted[final_index]["content"], canonical_name(registry, code, language), sample_id,
            )
        metadata.update({
            "source_class_code": source_class, "canonical_class_code": code,
            "class_code": code, "canonical_english_name": canonical_name(registry, code, "en"),
            "canonical_chinese_name": canonical_name(registry, code, "zh"),
            "canonical_name_conversion_version": "open-agri-v3/v1",
            "canonical_name_registry_sha256": registry.digest,
            "open_agri_v3_source_kind": source_kind,
        })
        result = {**source, "messages": converted, "metadata": metadata}
        if route == "rag":
            result["tools"] = tools_json()
        audit.update({"canonical_class_code": code, "terminal_kind": terminal_kind(source), "action": "include"})
        return result, audit
    except (ValueError, json.JSONDecodeError) as exc:
        audit["reasons"] = [str(exc)]
        return None, audit


def supplement_files(root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(root.glob("**/collection/direct/accepted.jsonl")):
        files.append((path, "supplement_direct"))
    for path in sorted(root.glob("**/collection/rag/accepted.jsonl")):
        kind = "oracle_rag" if "/recoveries/oracle-" in str(path) else "supplement_rag"
        files.append((path, kind))
    files.append((root / "collection/rag-v2/accepted.jsonl", "supplement_rag"))
    return [(path, kind) for path, kind in files if path.is_file() and path.stat().st_size]


def selected_supplement_hashes(root: Path) -> tuple[set[str], set[str]]:
    normal = {
        str(row["image_sha256"]) for row in read_jsonl(root / "review/image_decisions.jsonl")
        if row.get("accepted") is True
    }
    oracle = {
        str(row["image_sha256"]) for row in read_jsonl(root / "oracle_final_export/review/image_decisions.jsonl")
        if row.get("provisional_accepted") is True
    }
    return normal, oracle


def complete_supplement_rows(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]], *, selected_normal: set[str], selected_oracle: set[str],
) -> list[dict[str, Any]]:
    """Require complete 4-view route groups; ordinary images require both routes."""
    rejected = {str(item["sample_id"]) for item in audits if item["action"] != "include"}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(metadata_sha(row), str((row.get("metadata") or {}).get("route")))].append(row)
    accepted: list[dict[str, Any]] = []
    required_cells = {(question, language) for question in ("open", "option") for language in ("en", "zh")}
    by_image: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (digest, route), items in grouped.items():
        by_image[digest][route] = items
    for digest, routes in sorted(by_image.items()):
        normal = digest in selected_normal
        oracle = digest in selected_oracle
        needed = {"direct", "rag"} if normal else ({"rag"} if oracle else set())
        if not needed:
            continue
        if not needed <= set(routes):
            continue
        complete = True
        for route in needed:
            cells = {(str((row.get("metadata") or {}).get("question_type")), str((row.get("metadata") or {}).get("language"))) for row in routes[route]}
            if cells != required_cells or len(routes[route]) != 4 or any(str(row.get("sample_id")) in rejected for row in routes[route]):
                complete = False
        if complete:
            for route in sorted(needed):
                accepted.extend(routes[route])
    return accepted


def copy_v1_base(v1: Path, output: Path) -> None:
    """Copy the small manifests and retain source image links without touching v1."""
    for relative in ("catalog", "manifests", "taxonomy", "vlm_data/candidates", "vlm_data/accepted/dev.jsonl",
                     "vlm_data/accepted/private", "vlm_data/accepted/test_public.jsonl", "vlm_data/audits"):
        source = v1 / relative
        destination = output / relative
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    images = output / "images"
    images.symlink_to((v1 / "images").resolve())


def source_rows(files: list[tuple[Path, str]], registry: Registry) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    included, isolated, audits = [], [], []
    for path, kind in files:
        for line, row in enumerate(read_jsonl(path), start=1):
            converted, audit = convert_record(row, registry=registry, source_path=path, source_line=line, source_kind=kind)
            audits.append(audit)
            if audit["action"] == "isolate":
                isolated.append({"source_record": row, "audit": audit})
            elif converted is not None:
                included.append(converted)
    return included, isolated, audits


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate exact image/route/cell supervision with legacy preceding supplement."""
    chosen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    dropped = []
    for row in rows:
        metadata = row.get("metadata") or {}
        key = (metadata_sha(row), str(metadata["route"]), str(metadata["question_type"]), str(metadata["language"]))
        if key in chosen:
            dropped.append({"sample_id": row.get("sample_id"), "reason": "duplicate_image_route_cell", "kept_sample_id": chosen[key].get("sample_id")})
            continue
        chosen[key] = row
    return sorted(chosen.values(), key=lambda row: str(row.get("sample_id"))), dropped


def train_views(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    views: dict[str, list[dict[str, Any]]] = {f"{domain}_{route}": [] for domain in ("disease", "pest") for route in ("direct", "rag")}
    for row in rows:
        metadata = row["metadata"]
        key = f"{metadata['task_domain']}_{metadata['route']}"
        if key not in views:
            raise ValueError(f"{row.get('sample_id')}: invalid train view {key}")
        views[key].append(row)
    summary = {"schema_version": "agrinet.open-agri-v3.train-views/v1", "total_rows": len(rows), "views": {}}
    for key, items in views.items():
        path = output / "vlm_data/accepted/train" / f"{key}.jsonl"
        write_jsonl(path, items)
        summary["views"][key] = {"rows": len(items), "path": str(path.relative_to(output)), "sha256": sha256(path)}
    write_json(output / "vlm_data/accepted/train/summary.json", summary)
    return summary


def build(v1: Path, output: Path, legacy: Path, supplement: Path, *, replace: bool) -> dict[str, Any]:
    if output.exists():
        if not replace:
            raise ValueError(f"output exists: {output}; pass --replace")
        shutil.rmtree(output)
    registry = load_registry(v1 / "taxonomy/canonical_label_registry.jsonl", v1 / "taxonomy/approval.json")
    copy_v1_base(v1, output)
    legacy_files = [(legacy / f"{domain}_{route}.jsonl", "historical") for domain in ("disease", "pest") for route in ("direct", "rag")]
    historical, isolated, history_audit = source_rows(legacy_files, registry)
    supplement_rows, supplement_isolated, supplement_audit = source_rows(supplement_files(supplement), registry)
    normal, oracle = selected_supplement_hashes(supplement)
    supplement_rows = complete_supplement_rows(supplement_rows, supplement_audit, selected_normal=normal, selected_oracle=oracle)
    all_rows, dedup = deduplicate(historical + supplement_rows)
    all_isolated = isolated + supplement_isolated
    train_summary = train_views(output, all_rows)
    write_jsonl(output / "vlm_data/compatibility/short_budget_terminal/records.jsonl", all_isolated)
    write_jsonl(output / "vlm_data/audits/historical_and_supplement_conversion.jsonl", history_audit + supplement_audit)
    write_jsonl(output / "vlm_data/audits/deduplicated_records.jsonl", dedup)
    provenance = {
        "parent_dataset": display_path(v1), "parent_registry_sha256": registry.digest,
        "supplement_root": display_path(supplement), "selected_normal_images": len(normal),
        "selected_oracle_images": len(oracle), "short_budget_terminal_isolated": len(all_isolated),
        "five_turn_terminal_included": sum(terminal_kind(row) == "five_turn" for row in all_rows),
        "supplement_rows_after_complete_view_gate": len(supplement_rows),
        "supplement_images_after_complete_view_gate": len({metadata_sha(row) for row in supplement_rows}),
        "supplement_images_by_source_kind": dict(sorted(Counter(
            str((row.get("metadata") or {}).get("open_agri_v3_source_kind"))
            for row in supplement_rows
        ).items())),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "vlm_data/audits/open_agri_v3_provenance.json", provenance)
    summary_path = output / "manifests/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "schema_version": "agrinet.open-agri-v3/v1", "taxonomy_version": "open_agri_v3",
        "parent_taxonomy_version": "open_agri_v2_canonical_v1", "release_status": "approved",
        "registry_sha256": registry.digest, "canonical_name_source": display_path(v1 / "taxonomy/canonical_label_registry.jsonl"),
        "supervision": {"train_rows": train_summary["total_rows"], "short_budget_terminal_isolated": len(all_isolated),
                          "five_turn_terminal_included": provenance["five_turn_terminal_included"]},
    })
    write_json(summary_path, summary)
    return {"summary": summary, "train": train_summary, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-train-root", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--supplement-root", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = build(args.v1_root.resolve(), args.output_root.resolve(), args.legacy_train_root.resolve(),
                   args.supplement_root.resolve(), replace=args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
