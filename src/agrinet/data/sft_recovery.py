from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from agrinet.data.io import DataError


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
RECOVERY_PILOT_ARTIFACT_ID_RE = re.compile(r"agrinet-rag-recovery-pilot-v\d+")


def recovery_pilot_artifact_id(pilot_dir: Path) -> str:
    """Return the stable artifact ID encoded by a recovery-pilot output directory."""
    artifact_id = pilot_dir.name
    if not RECOVERY_PILOT_ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise DataError(
            "recovery Pilot output directory must be named "
            f"agrinet-rag-recovery-pilot-v<integer>, got: {pilot_dir}"
        )
    return artifact_id


def recovery_pilot_experiment_id(artifact_id: str) -> str:
    """Map a versioned recovery-pilot artifact ID to its registered experiment ID."""
    if not RECOVERY_PILOT_ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise DataError(f"invalid recovery Pilot artifact ID: {artifact_id!r}")
    return f"data-sft-{artifact_id.removeprefix('agrinet-')}"


def validate_recovery_pilot_destination(
    artifacts_root: Path, pilot_dir: Path, artifact_id: str
) -> None:
    """Fail closed when a versioned recovery-pilot artifact could be misidentified or overwritten."""
    expected_artifact_id = recovery_pilot_artifact_id(pilot_dir)
    if artifact_id != expected_artifact_id:
        raise DataError(
            "recovery Pilot artifact ID does not match its output directory: "
            f"{artifact_id!r} != {expected_artifact_id!r}"
        )
    if pilot_dir.parent.resolve() != artifacts_root.resolve():
        raise DataError(
            "recovery Pilot output directory must be a direct child of artifacts_root: "
            f"{pilot_dir}"
        )
    if pilot_dir.exists():
        raise DataError(
            "recovery Pilot output directory already exists; preserve this immutable "
            f"artifact instead of overwriting it: {pilot_dir}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read JSONL {path}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_sha256(path: Path, root: Path) -> str:
    resolved = path if path.is_absolute() else root / path
    return sha256_file(resolved.resolve())


def publish_frozen_dataset(
    source: Path, destination: Path, artifact_id: str, description: str
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    data_path = destination / "data.jsonl"
    if destination.exists():
        if not data_path.is_file() or sha256_file(data_path) != source_hash:
            raise DataError(f"immutable artifact conflict: {destination}")
    else:
        destination.mkdir(parents=True)
        shutil.copyfile(source, data_path)

    rows = read_jsonl(data_path)
    languages: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    tool_rows = 0
    for row in rows:
        messages = row.get("messages") or []
        user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
        languages["zh" if ZH_RE.search(user) else "en"] += 1
        image = str((row.get("images") or [""])[0])
        code = _class_code_from_path(image)
        if code:
            domains[_domain(code)] += 1
        tool_rows += int(bool(row.get("tools")))
    stats = {
        "rows": len(rows),
        "languages": dict(sorted(languages.items())),
        "domains": dict(sorted(domains.items())),
        "tool_rows": tool_rows,
    }
    manifest = {
        "schema_version": "agrinet.sft.frozen/v1",
        "artifact_id": artifact_id,
        "artifact_type": "datasets",
        "source_path": str(source),
        "data_sha256": source_hash,
        "immutable": True,
        "statistics": stats,
    }
    (destination / "artifact.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (destination / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        f"# {artifact_id}\n\n{description}\n\nFrozen from `{source}`. Do not edit `data.jsonl` in place.\n",
        encoding="utf-8",
    )
    return stats


def _class_code_from_path(path: str) -> str:
    match = re.search(r"/(N0[45]\d{3})/", path.replace("\\", "/"))
    return match.group(1) if match else ""


def _domain(code: str) -> str:
    return "disease" if code.startswith("N04") else "pest"


def _sample_quota(
    pool: list[dict[str, Any]], count: int, rng: random.Random, used: set[str]
) -> list[dict[str, Any]]:
    available = [row for row in pool if row["sample_id"] not in used]
    available.sort(key=lambda row: row["sample_id"])
    rng.shuffle(available)
    if len(available) < count:
        raise DataError(f"quota needs {count} samples, only {len(available)} available")
    selected = available[:count]
    used.update(row["sample_id"] for row in selected)
    return selected


def _eval_exclusions(manifest: Path, root: Path) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for row in read_jsonl(manifest):
        raw = str(row.get("image_path") or "")
        if not raw:
            continue
        path = Path(raw)
        resolved = (path if path.is_absolute() else root / path).resolve()
        paths.add(str(resolved))
        if resolved.is_file():
            hashes.add(sha256_file(resolved))
    return paths, hashes


def build_pilot_plan(
    candidates_path: Path, split_path: Path, eval_manifest: Path, output_dir: Path, root: Path, seed: int,
) -> dict[str, Any]:
    candidates = read_jsonl(candidates_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    unknown = set(split["test_only_unknown_classes"])
    excluded_paths, excluded_hashes = _eval_exclusions(eval_manifest, root)
    eligible: list[dict[str, Any]] = []
    audit_pool: list[dict[str, Any]] = []
    excluded = Counter()
    for row in candidates:
        code = str(row.get("final_label") or "")
        image = Path(str(row.get("query_image") or ""))
        resolved = (image if image.is_absolute() else root / image).resolve()
        if not resolved.is_file():
            excluded["missing_image"] += 1
            continue
        digest = sha256_file(resolved)
        item = {**row, "class_code": code, "image_sha256": digest}
        if code in unknown:
            audit_pool.append(item)
        elif str(resolved) in excluded_paths or digest in excluded_hashes:
            excluded["evaluation_overlap"] += 1
        else:
            eligible.append(item)

    rng = random.Random(seed)
    used: set[str] = set()
    targets: list[dict[str, Any]] = []
    specifications = [
        ("rag_open", "open", "standard"),
        ("rag_option", "option", "standard"),
        ("rag_stop_correction", "open", "stop_correction"),
    ]
    for kind, question_type, trajectory_mode in specifications:
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                pool = [row for row in eligible if _domain(row["class_code"]) == domain]
                for reserve_rank, row in enumerate(_sample_quota(pool, 20, rng, used)):
                    target = {
                        "target_id": f"{kind}-{language}-{row['sample_id']}",
                        "source_sample_id": row["sample_id"],
                        "query_image": row["query_image"],
                        "image_sha256": row["image_sha256"],
                        "class_code": row["class_code"],
                        "class_name": _label_name(row),
                        "class_name_zh": row.get("final_label_zh", ""),
                        "task_domain": domain,
                        "language": language,
                        "question_type": question_type,
                        "trajectory_mode": trajectory_mode,
                        "train_eligible": True,
                        "max_tool_turns": 3,
                        "reserve": reserve_rank >= 4,
                    }
                    if question_type == "option":
                        target["correct_option"] = "ABCD"[reserve_rank % 4]
                    targets.append(target)

    attempts = [
        {**target, "candidate_index": candidate_index}
        for target in targets
        for candidate_index in range(1, 4)
    ]
    audit: list[dict[str, Any]] = []
    audit_used: set[str] = set()
    for language in ("en", "zh"):
        for domain in ("disease", "pest"):
            pool = [row for row in audit_pool if _domain(row["class_code"]) == domain]
            for row in _sample_quota(pool, 4, rng, audit_used):
                audit.append({
                    "audit_id": f"unknown-{language}-{row['sample_id']}",
                    "source_sample_id": row["sample_id"],
                    "query_image": row["query_image"],
                    "image_sha256": row["image_sha256"],
                    "class_code": row["class_code"],
                    "task_domain": domain,
                    "language": language,
                    "known_bucket": "unknown",
                    "train_eligible": False,
                })

    write_jsonl(output_dir / "plan" / "rag_targets.jsonl", targets)
    write_jsonl(output_dir / "plan" / "rag_candidate_attempts.jsonl", attempts)
    write_jsonl(output_dir / "audit_unknown" / "probes.jsonl", audit)
    stats = {
        "seed": seed, "rag_targets": len(targets),
        "rag_candidate_attempts": len(attempts), "unknown_audit": len(audit),
        "excluded": dict(excluded),
    }
    (output_dir / "plan" / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _label_name(row: dict[str, Any]) -> str:
    code = row.get("final_label")
    for candidate in row.get("candidate_labels", []):
        if candidate.get("code") == code:
            return str(candidate.get("name") or "")
    return ""


def _language(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
    return "zh" if ZH_RE.search(user) else "en"

def _normalize_direct_think(content: str, language: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL | re.IGNORECASE)
    body = match.group(1).strip() if match else content.strip()
    stop_patterns = (
        r"(?im)^\s*(Knowledge Verification|Wiki Knowledge|External Knowledge|知识验证|百科知识|论文摘要|分类谱系)\s*[:：]",
        r"(?i)<title>摘要</title>",
    )
    for pattern in stop_patterns:
        found = re.search(pattern, body)
        if found: body = body[:found.start()].strip()
    body = re.sub(r"(?im)^\s*(Visual Observation|视觉观察)\s*[:：]\s*", "", body, count=1)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    # Keep the image-grounded portion only; legacy tails often contain unrelated wiki dumps.
    body = body[:550].rsplit("\n", 1)[0] if len(body) > 550 else body
    if language == "zh":
        headings = ("视觉观察", "候选分析", "证据与排除", "不确定性")
        note = "本 Direct 样本在形成视觉候选后直接判断，不调用 RAG。"
        evidence_note = "以上分析依据可见特征比较并排除了相近候选。"
        uncertainty_note = "沿用原始分析中的不确定性判断。"
    else:
        headings = ("Visual Observation", "Candidate Analysis", "Evidence and Rejections", "Uncertainty")
        note = "This Direct sample answers after forming visual candidates without calling RAG."
        evidence_note = "The analysis above compares visible evidence and rejects nearby alternatives."
        uncertainty_note = "The uncertainty follows the source assessment."
    # Preserve the source reasoning while imposing the same stage boundaries used by RAG rows.
    rendered = (f"<think>\n{headings[0]}:\n{body}\n\n{headings[1]}:\n{note}\n\n"
            f"{headings[2]}:\n{evidence_note}\n\n"
            f"{headings[3]}:\n{uncertainty_note}\n</think>")
    return rendered


def _option_row(
    source: dict[str, Any], sample: dict[str, Any], letter: str, source_index: int
) -> dict[str, Any]:
    choices = [str(item.get("name") or "") for item in sample.get("candidate_labels", [])]
    answer = _label_name(sample)
    if len(choices) != 4 or answer not in choices:
        raise DataError(f"sample cannot form four-way option question: {sample.get('sample_id')}")
    choices.remove(answer)
    position = ord(letter) - ord("A")
    choices.insert(position, answer)
    language = _language(source)
    question = (
        "<image>\n请选择图中病虫害的规范名称。只在 <answer> 中输出选项字母。\n"
        if language == "zh"
        else "<image>\nSelect the canonical name shown in the image. Output only the option letter inside <answer>.\n"
    )
    question += "\n".join(f"{key}. {value}" for key, value in zip("ABCD", choices))
    messages = [dict(message) for message in source["messages"]]
    messages[1] = {**messages[1], "content": question}
    original = str(messages[-1].get("content", ""))
    think = _normalize_direct_think(original, language)
    messages[-1] = {**messages[-1], "content": f"{think}\n\n<answer>{letter}</answer>"}
    return {
        "sample_id": f"direct-option-{language}-{sample['sample_id']}",
        "images": source["images"], "messages": messages,
        "metadata": {
            "source_dataset": "agrinet-disease-pest-direct-sft-v1",
            "source_row_index": source_index, "source_sample_id": sample["sample_id"],
            "language": language, "task_domain": _domain(sample["final_label"]),
            "question_type": "option", "correct_option": letter, "train_eligible": True,
        },
    }


def build_direct_pilot(
    direct_path: Path, candidates_path: Path, eval_manifest: Path, output_dir: Path, root: Path,
    seed: int, excluded_rag_images: set[str] | None = None,
) -> dict[str, Any]:
    direct_rows = read_jsonl(direct_path)
    candidates = read_jsonl(candidates_path)
    by_image = {str(row["query_image"]): row for row in candidates}
    excluded_paths, excluded_hashes = _eval_exclusions(eval_manifest, root)
    pool: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(direct_rows):
        image = str((row.get("images") or [""])[0])
        sample = by_image.get(image)
        if not sample:
            continue
        resolved = (Path(image) if Path(image).is_absolute() else root / image).resolve()
        digest = sha256_file(resolved)
        if str(resolved) not in excluded_paths and digest not in excluded_hashes and image not in (excluded_rag_images or set()):
            pool.append((index, row, sample))

    rng = random.Random(seed + 1)
    selected: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    for question_type in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                available = [
                    item for item in pool
                    if _language(item[1]) == language and _domain(item[2]["final_label"]) == domain
                    and item[2]["sample_id"] not in {chosen[2]["sample_id"] for chosen in selected}
                ]
                available.sort(key=lambda item: item[2]["sample_id"]); rng.shuffle(available)
                if len(available) < 4:
                    raise DataError(f"direct quota unavailable: {question_type}/{language}/{domain}")
                selected.extend((*item, question_type) for item in available[:4])

    open_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    letters = iter([letter for letter in "ABCD" for _ in range(4)])
    for index, row, sample, question_type in selected:
        if question_type == "open":
            copied = dict(row)
            copied["sample_id"] = f"direct-open-{_language(row)}-{sample['sample_id']}"
            copied["metadata"] = {
                "source_dataset": "agrinet-disease-pest-direct-sft-v1",
                "source_row_index": index, "source_sample_id": sample["sample_id"],
                "language": _language(row), "task_domain": _domain(sample["final_label"]),
                "question_type": "open", "train_eligible": True,
            }
            messages = [dict(message) for message in copied["messages"]]
            original = str(messages[-1].get("content", ""))
            answer = ANSWER_RE.search(original)
            messages[-1] = {**messages[-1], "content": _normalize_direct_think(original, _language(row)) +
                            f"\n\n<answer>{answer.group(1).strip() if answer else _label_name(sample)}</answer>"}
            copied["messages"] = messages
            open_rows.append(copied)
        else:
            option_rows.append(_option_row(row, sample, next(letters), index))
    write_jsonl(output_dir / "accepted" / "direct_open.jsonl", open_rows)
    write_jsonl(output_dir / "accepted" / "direct_option.jsonl", option_rows)
    return {"direct_open": len(open_rows), "direct_option": len(option_rows)}


def validate_pilot(output_dir: Path) -> dict[str, Any]:
    targets = read_jsonl(output_dir / "plan" / "rag_targets.jsonl")
    attempts = read_jsonl(output_dir / "plan" / "rag_candidate_attempts.jsonl")
    direct_open = read_jsonl(output_dir / "accepted" / "direct_open.jsonl")
    direct_option = read_jsonl(output_dir / "accepted" / "direct_option.jsonl")
    audit = read_jsonl(output_dir / "audit_unknown" / "probes.jsonl")
    reserve_plan = any("reserve" in row for row in targets)
    expected_targets = len(targets)
    expected = {"rag_targets": expected_targets, "attempts": expected_targets * 3, "direct_open": 16, "direct_option": 16, "audit": 16}
    actual = {
        "rag_targets": len(targets), "attempts": len(attempts),
        "direct_open": len(direct_open), "direct_option": len(direct_option), "audit": len(audit),
    }
    if actual != expected:
        raise DataError(f"pilot count mismatch: expected={expected} actual={actual}")
    if any(row.get("train_eligible") is not False for row in audit):
        raise DataError("unknown audit probe marked train eligible")
    if any(row.get("train_eligible") is not True for row in targets):
        raise DataError("RAG target marked train ineligible")
    attempt_counts = Counter(row["target_id"] for row in attempts)
    if set(attempt_counts.values()) != {3}:
        raise DataError("each RAG target must have exactly three candidate attempts")
    rag_quota = Counter(
        (row["trajectory_mode"], row["question_type"], row["language"], row["task_domain"])
        for row in targets
    )
    if len(set(rag_quota.values())) != 1 or (reserve_plan and next(iter(rag_quota.values())) < 4):
        raise DataError(f"unbalanced RAG quotas: {rag_quota}")
    option_balance = Counter(row["metadata"]["correct_option"] for row in direct_option)
    if option_balance != Counter({letter: 4 for letter in "ABCD"}):
        raise DataError(f"unbalanced Direct options: {option_balance}")
    return {**actual, "rag_quota_cells": len(rag_quota), "direct_option_balance": dict(option_balance)}


def prepare_recovery_pilot(
    direct_source: Path, rag_source: Path, candidates: Path, split: Path, eval_manifest: Path,
    artifacts_root: Path, pilot_dir: Path, repository: Path, artifact_id: str, seed: int,
) -> dict[str, Any]:
    validate_recovery_pilot_destination(artifacts_root, pilot_dir, artifact_id)
    direct_stats = publish_frozen_dataset(
        direct_source, artifacts_root / "agrinet-disease-pest-direct-sft-v1",
        "agrinet-disease-pest-direct-sft-v1", "Existing bilingual non-RAG disease and pest SFT corpus.",
    )
    rag_stats = publish_frozen_dataset(
        rag_source, artifacts_root / "agrinet-rag-toolcall-sft-v1",
        "agrinet-rag-toolcall-sft-v1", "Existing single-image RAG tool-call SFT corpus.",
    )
    pilot_dir.mkdir(parents=True, exist_ok=True)
    for name in ("candidates", "rejected"):
        directory = pilot_dir / name
        directory.mkdir(exist_ok=True)
        (directory / "README.md").write_text(
            "Populated by plan-driven RAG rejection sampling after runtime smoke validation.\n",
            encoding="utf-8",
        )
    plan_stats = build_pilot_plan(candidates, split, eval_manifest, pilot_dir, repository, seed)
    planned = read_jsonl(pilot_dir / "plan" / "rag_targets.jsonl")
    direct_pilot_stats = build_direct_pilot(
        direct_source, candidates, eval_manifest, pilot_dir, repository, seed,
        excluded_rag_images={str(row["query_image"]) for row in planned},
    )
    validation = validate_pilot(pilot_dir)
    summary = {
        "direct_frozen": direct_stats, "rag_frozen": rag_stats,
        "plan": plan_stats, "direct_pilot": direct_pilot_stats, "validation": validation,
        "full_generation_authorized": False, "training_authorized": False,
    }
    (pilot_dir / "review").mkdir(exist_ok=True)
    (pilot_dir / "review" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        ("Frozen Direct rows", direct_stats["rows"]), ("Frozen RAG rows", rag_stats["rows"]),
        ("RAG targets", validation["rag_targets"]), ("RAG candidate attempts", validation["attempts"]),
        ("Direct Open", validation["direct_open"]), ("Direct Option", validation["direct_option"]),
        ("Unknown audit probes", validation["audit"]),
    ]
    table = "".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in rows)
    (pilot_dir / "review" / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AgriNet SFT recovery Pilot</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:40px auto;color:#202124}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #bbb;text-align:left}"
        "h1{font-size:28px;letter-spacing:0}</style><h1>AgriNet SFT recovery Pilot</h1>"
        f"<table>{table}</table><p>Full generation and training remain disabled pending review.</p>", encoding="utf-8"
    )
    manifest = {
        "schema_version": "agrinet.sft.recovery-pilot/v1",
        "artifact_id": artifact_id, "seed": seed,
        "status": "planned", "review_required": True, "statistics": validation,
    }
    (pilot_dir / "artifact.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return summary
