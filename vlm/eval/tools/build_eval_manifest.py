#!/usr/bin/env python3
"""Build AgriNet VLM evaluation manifests and VLMEvalKit TSV files."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


OPEN_PROMPTS = {
    "en": {
        "disease": "What disease is shown in the image? Answer with the disease name only.",
        "pest": "What pest is shown in the image? Answer with the pest name only.",
    },
    "zh": {
        "disease": "图片中显示的是什么植物病害？请只回答病害名称。",
        "pest": "图片中显示的是什么害虫？请只回答害虫名称。",
    },
}

OPTION_PROMPTS = {
    "en": "What category is shown in the image? Choose the single best option. Answer with only the option letter.\n{options}",
    "zh": "图片中显示的类别是哪一个？请选择唯一最佳选项。请只回答选项字母。\n{options}",
}

LETTERS = "ABCD"
CODE_RE = re.compile(r"^N\d{5}$", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def _lang_from_path(path: Path) -> str:
    name = path.name.lower()
    if re.search(r"(^|[_-])zh([_.-]|$)", name):
        return "zh"
    return "en"


def _resolve(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _read_class_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _class_aliases(row: dict[str, Any]) -> list[str]:
    values = []
    for key in ("name", "english_name", "chinese_name", "label", "code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return list(dict.fromkeys(values))


def _class_name(row: dict[str, Any]) -> str:
    return str(row.get("english_name") or row.get("name") or row.get("label") or row.get("code"))


def _candidate_zh_names(row: dict[str, Any]) -> list[str]:
    structured = row.get("structured_result") if isinstance(row.get("structured_result"), dict) else {}
    final_zh = structured.get("final_label_zh")
    return [final_zh.strip()] if isinstance(final_zh, str) and final_zh.strip() else []


def _load_class_index(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    by_code: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, list[str]] = {}
    for path in sorted(data_dir.glob("classes_all*.jsonl")):
        for row in _read_class_rows(path):
            code = str(row.get("code") or row.get("label") or "").strip()
            if not code:
                continue
            by_code[code] = row
            domain = str(row.get("task_domain") or "unknown")
            by_domain.setdefault(domain, [])
            if code not in by_domain[domain]:
                by_domain[domain].append(code)
    return by_code, by_domain


def _load_pair_index(data_dir: Path) -> dict[str, list[str]]:
    pair_index: dict[str, list[str]] = {}
    for path in [data_dir / "finegrained_pairs_vit_base.jsonl", data_dir / "finegrained_pairs.jsonl"]:
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            code = str(row.get("anchor") or row.get("label") or "")
            if not code:
                continue
            negatives: list[str] = []
            for value in row.get("similar_classes") or []:
                if isinstance(value, str):
                    negatives.append(value)
            for value in row.get("hard_negatives") or []:
                if isinstance(value, dict) and value.get("code"):
                    negatives.append(str(value["code"]))
                elif isinstance(value, str):
                    negatives.append(value)
            pair_index[code] = list(dict.fromkeys(x for x in negatives if x and x != code))
        break
    return pair_index


def _option_codes(
    row: dict[str, Any],
    *,
    label_code: str,
    domain: str,
    pair_index: dict[str, list[str]],
    by_domain: dict[str, list[str]],
    rng: random.Random,
) -> list[str]:
    codes = [label_code]
    for value in row.get("candidate_codes") or []:
        code = str(value)
        if code != label_code:
            codes.append(code)
    codes.extend(pair_index.get(label_code, []))
    codes = list(dict.fromkeys(code for code in codes if CODE_RE.match(code)))

    if len(codes) < 4:
        pool = [code for code in by_domain.get(domain, []) if code not in codes]
        rng.shuffle(pool)
        codes.extend(pool[: 4 - len(codes)])

    negatives = [code for code in codes if code != label_code][:3]
    options = [label_code] + negatives
    rng.shuffle(options)
    return options


def _option_prompt(lang: str, option_names: list[str]) -> str:
    lines = [f"{letter}. {name}" for letter, name in zip(LETTERS, option_names)]
    template = OPTION_PROMPTS.get(lang, OPTION_PROMPTS["en"])
    return template.format(options="\n".join(lines))


def _base_record(
    row: dict[str, Any],
    *,
    lang: str,
    image: str,
    sample_id: str,
    label_code: str,
    label_name: str,
    by_code: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    class_row = by_code.get(label_code, {})
    aliases = []
    aliases.extend(_candidate_zh_names(row))
    aliases.extend(_class_aliases(class_row))
    aliases.extend(row.get("candidate_names") or [])
    aliases = list(dict.fromkeys(str(x).strip() for x in aliases if str(x).strip()))
    return {
        "source_sample_id": sample_id,
        "language": lang,
        "task_domain": row.get("task_domain") or class_row.get("task_domain") or "disease",
        "image_path": image,
        "label_code": label_code,
        "label_name": label_name,
        "label_aliases": aliases,
        "candidate_codes": row.get("candidate_codes") or [],
        "candidate_names": row.get("candidate_names") or [],
        "uncertainty": row.get("uncertainty"),
    }


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(args.traces[0]).resolve().parent
    by_code, by_domain = _load_class_index(data_dir)
    pair_index = _load_pair_index(data_dir)
    rng = random.Random(args.seed)

    records: list[dict[str, Any]] = []
    source_count = 0
    for trace_path_raw in args.traces:
        trace_path = Path(trace_path_raw)
        lang = _lang_from_path(trace_path)
        for row in _load_jsonl(trace_path):
            image = row.get("query_image")
            if not isinstance(image, str) or not image:
                continue
            if not _resolve(image, repo_root).exists():
                raise FileNotFoundError(f"image not found for {row.get('sample_id')}: {image}")

            sample_id = str(row.get("sample_id") or f"sample_{source_count:06d}")
            label_code = str(row.get("final_label_code") or row.get("final_label") or "")
            label_name = str(row.get("final_label_name") or row.get("final_label") or label_code)
            if not label_code:
                continue
            base = _base_record(
                row, lang=lang, image=image, sample_id=sample_id, label_code=label_code,
                label_name=label_name, by_code=by_code,
            )
            domain = str(base["task_domain"])

            open_record = dict(base)
            open_record.update({
                "index": f"{lang}_{sample_id}_open",
                "id": f"{lang}_{sample_id}_open",
                "question_type": "open",
                "question": OPEN_PROMPTS.get(lang, OPEN_PROMPTS["en"]).get(domain, OPEN_PROMPTS[lang]["disease"]),
                "answer": label_name,
                "option_answer": "",
                "option_codes": [],
                "option_names": [],
            })
            records.append(open_record)

            option_codes = _option_codes(
                row, label_code=label_code, domain=domain, pair_index=pair_index,
                by_domain=by_domain, rng=rng,
            )
            option_names = [_class_name(by_code.get(code, {"code": code})) for code in option_codes]
            correct_letter = LETTERS[option_codes.index(label_code)]
            option_record = dict(base)
            option_record.update({
                "index": f"{lang}_{sample_id}_option",
                "id": f"{lang}_{sample_id}_option",
                "question_type": "option",
                "question": _option_prompt(lang, option_names),
                "answer": correct_letter,
                "option_answer": correct_letter,
                "option_codes": option_codes,
                "option_names": option_names,
            })
            for letter, code, name in zip(LETTERS, option_codes, option_names):
                option_record[letter] = name
                option_record[f"{letter}_code"] = code
            records.append(option_record)

            source_count += 1
            if args.limit and source_count >= args.limit:
                break
        if args.limit and source_count >= args.limit:
            break
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="Teacher trace JSONL files.")
    parser.add_argument("--output", required=True, help="Output manifest JSONL path.")
    parser.add_argument("--output-tsv", help="Optional VLMEvalKit TSV output path.")
    parser.add_argument("--repo-root", default=".", help="Repository root for image existence checks.")
    parser.add_argument("--data-dir", help="Directory containing classes_all*.jsonl and finegrained pairs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional source-image row limit for smoke eval.")
    parser.add_argument("--seed", type=int, default=20260530, help="Deterministic option shuffle seed.")
    args = parser.parse_args()

    records = build_records(args)
    output = Path(args.output)
    write_jsonl(records, output)
    if args.output_tsv:
        tsv = Path(args.output_tsv)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(tsv, sep="\t", index=False)

    question_counts = pd.Series([r["question_type"] for r in records]).value_counts().to_dict() if records else {}
    print(json.dumps({"output": str(output), "output_tsv": args.output_tsv, "rows": len(records), "question_counts": question_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
