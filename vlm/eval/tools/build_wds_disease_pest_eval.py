#!/usr/bin/env python3
"""Build a deterministic disease/pest RAG evaluation manifest from WDS shards."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from pathlib import Path

LETTERS = "ABCD"
OPEN_PROMPTS = {
    "en": {"disease": "What disease is shown in the image? Answer with the disease name only.", "pest": "What pest is shown in the image? Answer with the pest name only."},
    "zh": {"disease": "图片中显示的是什么植物病害？请只回答病害名称。", "pest": "图片中显示的是什么害虫？请只回答害虫名称。"},
}
OPTION_PROMPTS = {
    "en": "What category is shown in the image? Choose the single best option. Answer with only the option letter.\n{options}",
    "zh": "图片中显示的类别是哪一个？请选择唯一最佳选项。请只回答选项字母。\n{options}",
}


def class_names(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["编码"]: (row["英文"], row["中文"]) for row in csv.DictReader(handle)}


def strict_unknown_classes(all_root: Path, rare_threshold: int = 100) -> set[str]:
    """Match create_agrinet1k_open_domain_split.choose_unknown_classes."""
    return {
        class_dir.name
        for class_dir in all_root.iterdir()
        if class_dir.is_dir() and sum(1 for image in class_dir.iterdir() if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) <= rare_threshold
    }


def candidates(test_root: Path, unknown: set[str]) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]]:
    buckets: tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]] = ([], [])
    for shard in sorted(test_root.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".jpg"):
                    continue
                code = Path(member.name).stem.split("_", 1)[0]
                if code.startswith(("N04", "N05")):
                    buckets[1 if code in unknown else 0].append((shard, member.name, code))
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/AgriNet-1K"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--known-count", type=int, default=172)
    parser.add_argument("--unknown-count", type=int, default=137)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    names = class_names(args.dataset_root / "Agrinet1k中英文名.csv")
    unknown_classes = strict_unknown_classes(args.dataset_root / "all")
    known_rows, unknown_rows = candidates(args.dataset_root / "wds_split/test", unknown_classes)
    rng = random.Random(args.seed)
    rng.shuffle(known_rows)
    rng.shuffle(unknown_rows)
    selected = [("known", row) for row in known_rows[: args.known_count]] + [("unknown", row) for row in unknown_rows[: args.unknown_count]]
    domain_classes = {prefix: sorted(code for code in names if code.startswith(prefix)) for prefix in ("N04", "N05")}
    records = []
    for selected_index, (bucket, (shard, member_name, code)) in enumerate(selected):
        image_path = image_dir / member_name
        if not image_path.exists():
            with tarfile.open(shard) as archive:
                image_path.write_bytes(archive.extractfile(member_name).read())
        domain = "disease" if code.startswith("N04") else "pest"
        en, zh = names[code]
        pool = [value for value in domain_classes[code[:3]] if value != code]
        option_rng = random.Random(f"{args.seed}:{member_name}")
        negatives = option_rng.sample(pool, 3)
        option_codes = [code, *negatives]
        option_rng.shuffle(option_codes)
        # Historical evaluation has 309 source rows total across EN and ZH,
        # with one open and one option question per source row.
        lang = "en" if selected_index % 2 == 0 else "zh"
        label_name = en if lang == "en" else zh
        for lang, label_name in ((lang, label_name),):
            option_names = [names[value][0 if lang == "en" else 1] for value in option_codes]
            base = {
                "source_sample_id": Path(member_name).stem, "language": lang, "task_domain": domain,
                "known_bucket": bucket, "image_path": str(image_path), "label_code": code,
                "label_name": label_name, "label_aliases": [code, en, zh],
            }
            open_row = dict(base, index=f"{lang}_{Path(member_name).stem}_open", id=f"{lang}_{Path(member_name).stem}_open",
                            question_type="open", question=OPEN_PROMPTS[lang][domain], answer=label_name, option_answer="")
            correct_letter = LETTERS[option_codes.index(code)]
            option_row = dict(base, index=f"{lang}_{Path(member_name).stem}_option", id=f"{lang}_{Path(member_name).stem}_option",
                              question_type="option", question=OPTION_PROMPTS[lang].format(options="\n".join(f"{letter}. {name}" for letter, name in zip(LETTERS, option_names))),
                              answer=correct_letter, option_answer=correct_letter, option_codes=option_codes, option_names=option_names)
            records.extend((open_row, option_row))
    manifest = args.output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"source": "AgriNet-1K wds_split train/test", "seed": args.seed, "known_images": args.known_count,
               "unknown_images": args.unknown_count, "records": len(records), "unknown_class_count": len(unknown_classes)}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
