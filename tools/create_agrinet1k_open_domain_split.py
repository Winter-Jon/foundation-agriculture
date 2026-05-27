#!/usr/bin/env python3
"""Create an open-domain AgriNet-1K split from all/ and repack it as WDS."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    class_name: str
    path: Path


@dataclass(frozen=True)
class Action:
    class_name: str
    source_split: str
    target_split: str
    source_path: Path
    target_path: Path
    action: str
    reason: str
    seed: int
    entropy: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive open-domain AgriNet-1K split from "
            "all/<class_name>/*, then repack open_domain/train|val|test "
            "as WebDataset shards."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/AgriNet-1K"))
    parser.add_argument("--rare-threshold", type=int, default=100)
    parser.add_argument("--val-unknown-ratio", type=float, default=0.30)
    parser.add_argument("--val-image-ratio", type=float, default=0.30)
    parser.add_argument("--known-val-ratio", type=float, default=0.10)
    parser.add_argument("--known-test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--split-policy",
        choices=("ratio", "original-double-cross-entropy"),
        default="ratio",
        help="Split policy. The strict policy doubles the original val/test sizes and uses entropy-aware sampling.",
    )
    parser.add_argument(
        "--original-split-info",
        type=Path,
        default=Path("datasets/AgriNet-1K/split_info.json"),
        help="Original AgriNet-1K split_info.json used by the strict policy.",
    )
    parser.add_argument(
        "--cross-entropy-csv",
        type=Path,
        default=Path("datasets/AgriNet-1K/cross_entropy.csv"),
        help="Fallback cross-entropy CSV used by the strict policy when original entropy is missing.",
    )
    parser.add_argument("--target-val-count", type=int, default=100000)
    parser.add_argument("--target-test-count", type=int, default=200000)
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=100,
        help="Number of quantile bins for strict-policy val sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--timestamp", default=None, help="Override artifact timestamp.")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--pack-workers", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--split-info",
        type=Path,
        default=None,
        help="Split info JSON to pack. Defaults to the latest open_domain/manifests/split_info_*.json.",
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help="Only rebuild open_domain/wds_split from an existing open_domain train/val/test view.",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Create split manifests and train/val/test view without packing WebDataset shards.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write plan artifacts but do not link or pack files.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating relative symlinks.")
    parser.add_argument(
        "--replace-open-domain",
        action="store_true",
        help="Remove an existing open_domain directory before rebuilding.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rare_threshold < 1:
        raise ValueError("--rare-threshold must be >= 1")
    if not 0 < args.val_unknown_ratio <= 1:
        raise ValueError("--val-unknown-ratio must be in (0, 1]")
    if not 0 < args.val_image_ratio < 1:
        raise ValueError("--val-image-ratio must be in (0, 1)")
    if not 0 <= args.known_val_ratio < 1:
        raise ValueError("--known-val-ratio must be in [0, 1)")
    if not 0 <= args.known_test_ratio < 1:
        raise ValueError("--known-test-ratio must be in [0, 1)")
    if args.known_val_ratio + args.known_test_ratio >= 1:
        raise ValueError("--known-val-ratio + --known-test-ratio must be < 1")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be >= 1")
    if args.pack_workers < 1:
        raise ValueError("--pack-workers must be >= 1")
    if args.target_size < 1:
        raise ValueError("--target-size must be >= 1")
    if args.target_val_count < 1:
        raise ValueError("--target-val-count must be >= 1")
    if args.target_test_count < 1:
        raise ValueError("--target-test-count must be >= 1")
    if args.entropy_bins < 1:
        raise ValueError("--entropy-bins must be >= 1")
    if args.dry_run and args.replace_open_domain:
        raise ValueError("--dry-run cannot be combined with --replace-open-domain")
    if args.pack_only and args.skip_pack:
        raise ValueError("--pack-only cannot be combined with --skip-pack")


def load_all_records(dataset_root: Path) -> dict[str, list[ImageRecord]]:
    all_root = dataset_root / "all"
    if not all_root.is_dir():
        raise RuntimeError(f"Missing required source directory: {all_root}")

    records: dict[str, list[ImageRecord]] = {}
    for class_dir in sorted(p for p in all_root.iterdir() if p.is_dir()):
        images = [
            ImageRecord(class_name=class_dir.name, path=p)
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            records[class_dir.name] = images

    if not records:
        raise RuntimeError(f"No images found under {all_root}")
    return records


def choose_unknown_classes(
    by_class: dict[str, list[ImageRecord]],
    rare_threshold: int,
    val_unknown_ratio: float,
) -> tuple[set[str], set[str], set[str]]:
    rare_counts = {
        class_name: len(images)
        for class_name, images in by_class.items()
        if len(images) <= rare_threshold
    }
    ordered = sorted(rare_counts, key=lambda name: (rare_counts[name], name))
    if not ordered:
        return set(), set(), set()

    eligible_for_val = [name for name in ordered if rare_counts[name] >= 2]
    val_count = round(len(ordered) * val_unknown_ratio)
    if val_count > 0 and eligible_for_val:
        val_count = max(1, min(val_count, len(eligible_for_val)))
    else:
        val_count = 0

    # Smallest rare classes are test-only; validation unknowns come from the
    # higher-count rare classes so they can still keep samples for test.
    val_unknown = set(eligible_for_val[-val_count:]) if val_count else set()
    unknown = set(ordered)
    test_only = unknown - val_unknown
    return unknown, val_unknown, test_only


def target_path(open_domain_root: Path, target_split: str, record: ImageRecord) -> Path:
    return open_domain_root / target_split / record.class_name / record.path.name


def add_action(
    actions: list[Action],
    *,
    record: ImageRecord,
    target_split: str,
    open_domain_root: Path,
    reason: str,
    seed: int,
    entropy: float | None = None,
) -> None:
    actions.append(
        Action(
            class_name=record.class_name,
            source_split="all",
            target_split=target_split,
            source_path=record.path,
            target_path=target_path(open_domain_root, target_split, record),
            action="copy" if False else "link",
            reason=reason,
            seed=seed,
            entropy=entropy,
        )
    )


def split_count(total: int, ratio: float, *, leave_at_least: int = 0) -> int:
    count = round(total * ratio)
    if total - count < leave_at_least:
        count = max(0, total - leave_at_least)
    return count


def plan_actions(
    by_class: dict[str, list[ImageRecord]],
    open_domain_root: Path,
    unknown_classes: set[str],
    val_unknown_classes: set[str],
    *,
    seed: int,
    val_image_ratio: float,
    known_val_ratio: float,
    known_test_ratio: float,
) -> list[Action]:
    rng = random.Random(seed)
    actions: list[Action] = []

    for class_name in sorted(by_class):
        images = list(by_class[class_name])
        rng.shuffle(images)

        if class_name in unknown_classes:
            if class_name in val_unknown_classes and len(images) >= 2:
                val_count = max(1, split_count(len(images), val_image_ratio, leave_at_least=1))
            else:
                val_count = 0
            for index, record in enumerate(images):
                if index < val_count:
                    add_action(
                        actions,
                        record=record,
                        target_split="val",
                        open_domain_root=open_domain_root,
                        reason="unknown_class_validation_sample",
                        seed=seed,
                    )
                else:
                    add_action(
                        actions,
                        record=record,
                        target_split="test",
                        open_domain_root=open_domain_root,
                        reason="unknown_class_test_sample",
                        seed=seed,
                    )
            continue

        test_count = split_count(len(images), known_test_ratio)
        val_count = split_count(len(images) - test_count, known_val_ratio / (1 - known_test_ratio))
        for index, record in enumerate(images):
            if index < test_count:
                target_split = "test"
                reason = "known_class_test_sample"
            elif index < test_count + val_count:
                target_split = "val"
                reason = "known_class_validation_sample"
            else:
                target_split = "train"
                reason = "known_class_train_sample"
            add_action(
                actions,
                record=record,
                target_split=target_split,
                open_domain_root=open_domain_root,
                reason=reason,
                seed=seed,
            )

    return actions


def load_cross_entropy(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise RuntimeError(f"Missing cross-entropy CSV: {path}")
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "cross_entropy"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Invalid cross-entropy CSV header in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                values[row["filename"]] = float(row["cross_entropy"])
            except (TypeError, ValueError):
                continue
    return values


def load_original_split_items(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        raise RuntimeError(f"Missing original split info: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise RuntimeError(f"Invalid split info JSON: missing object field 'splits' in {path}")
    return {split: list(splits.get(split, [])) for split in ("train", "val", "test")}


def build_original_lookup(
    original_splits: dict[str, list[dict[str, object]]],
    fallback_entropy: dict[str, float],
) -> dict[str, tuple[str, float | None]]:
    lookup: dict[str, tuple[str, float | None]] = {}
    for split in ("train", "val", "test"):
        for item in original_splits[split]:
            stem = str(item.get("stem") or Path(str(item.get("path", ""))).stem)
            if not stem:
                continue
            entropy_value = item.get("entropy")
            try:
                entropy = float(entropy_value) if entropy_value is not None else fallback_entropy.get(stem)
            except (TypeError, ValueError):
                entropy = fallback_entropy.get(stem)
            lookup[stem] = (split, entropy)
    return lookup


def stable_record_key(record: ImageRecord) -> str:
    return f"{record.class_name}/{record.path.name}"


def add_strict_action(
    actions: list[Action],
    *,
    record: ImageRecord,
    target_split: str,
    open_domain_root: Path,
    reason: str,
    seed: int,
    original_lookup: dict[str, tuple[str, float | None]],
    fallback_entropy: dict[str, float],
) -> None:
    source_split, entropy = original_lookup.get(record.path.stem, ("all", fallback_entropy.get(record.path.stem)))
    add_action(
        actions,
        record=record,
        target_split=target_split,
        open_domain_root=open_domain_root,
        reason=reason,
        seed=seed,
        entropy=entropy,
    )
    actions[-1] = Action(
        class_name=actions[-1].class_name,
        source_split=source_split,
        target_split=actions[-1].target_split,
        source_path=actions[-1].source_path,
        target_path=actions[-1].target_path,
        action=actions[-1].action,
        reason=actions[-1].reason,
        seed=actions[-1].seed,
        entropy=actions[-1].entropy,
    )


def sample_entropy_matched(
    candidates: list[ImageRecord],
    target_count: int,
    *,
    entropy_by_stem: dict[str, float | None],
    bins: int,
    rng: random.Random,
) -> set[str]:
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if target_count > len(candidates):
        raise RuntimeError(f"Cannot sample {target_count} val records from only {len(candidates)} candidates")
    if target_count == 0:
        return set()

    ordered = sorted(
        candidates,
        key=lambda record: (
            entropy_by_stem.get(record.path.stem) is None,
            entropy_by_stem.get(record.path.stem, 0.0) or 0.0,
            stable_record_key(record),
        ),
    )
    bin_count = min(bins, len(ordered))
    groups: list[list[ImageRecord]] = []
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        groups.append(ordered[start:end])

    quotas: list[int] = []
    fractions: list[tuple[float, int]] = []
    assigned = 0
    for index, group in enumerate(groups):
        exact = target_count * len(group) / len(ordered)
        quota = min(len(group), math.floor(exact))
        quotas.append(quota)
        assigned += quota
        fractions.append((exact - quota, index))

    remaining = target_count - assigned
    for _, index in sorted(fractions, reverse=True):
        if remaining == 0:
            break
        if quotas[index] < len(groups[index]):
            quotas[index] += 1
            remaining -= 1

    selected: set[str] = set()
    for group, quota in zip(groups, quotas):
        shuffled = list(group)
        rng.shuffle(shuffled)
        for record in shuffled[:quota]:
            selected.add(record.path.stem)
    return selected


def plan_original_double_cross_entropy_actions(
    by_class: dict[str, list[ImageRecord]],
    open_domain_root: Path,
    unknown_classes: set[str],
    val_unknown_classes: set[str],
    *,
    seed: int,
    val_image_ratio: float,
    original_split_info: Path,
    cross_entropy_csv: Path,
    target_val_count: int,
    target_test_count: int,
    entropy_bins: int,
) -> list[Action]:
    rng = random.Random(seed)
    fallback_entropy = load_cross_entropy(cross_entropy_csv)
    original_lookup = build_original_lookup(load_original_split_items(original_split_info), fallback_entropy)
    entropy_by_stem = {stem: entropy for stem, (_, entropy) in original_lookup.items()}
    entropy_by_stem.update({stem: value for stem, value in fallback_entropy.items() if stem not in entropy_by_stem})

    actions: list[Action] = []
    assigned: dict[str, str] = {}
    known_records: list[ImageRecord] = []

    for class_name in sorted(by_class):
        images = list(by_class[class_name])
        if class_name in unknown_classes:
            rng.shuffle(images)
            if class_name in val_unknown_classes and len(images) >= 2:
                val_count = max(1, split_count(len(images), val_image_ratio, leave_at_least=1))
            else:
                val_count = 0
            for index, record in enumerate(images):
                target_split = "val" if index < val_count else "test"
                reason = "strict_unknown_class_validation_sample" if target_split == "val" else "strict_unknown_class_test_sample"
                assigned[record.path.stem] = target_split
                add_strict_action(
                    actions,
                    record=record,
                    target_split=target_split,
                    open_domain_root=open_domain_root,
                    reason=reason,
                    seed=seed,
                    original_lookup=original_lookup,
                    fallback_entropy=fallback_entropy,
                )
        else:
            known_records.extend(images)

    val_needed = target_val_count - sum(1 for split in assigned.values() if split == "val")
    test_needed = target_test_count - sum(1 for split in assigned.values() if split == "test")
    if val_needed < 0 or test_needed < 0:
        raise RuntimeError(
            "Unknown-class allocation exceeds target counts: "
            f"val_needed={val_needed} test_needed={test_needed}"
        )

    original_test = [record for record in known_records if original_lookup.get(record.path.stem, (None, None))[0] == "test"]
    if len(original_test) > test_needed:
        raise RuntimeError(f"Original known test records ({len(original_test)}) exceed strict test budget ({test_needed})")
    for record in original_test:
        assigned[record.path.stem] = "test"

    remaining_known = [record for record in known_records if record.path.stem not in assigned]
    high_entropy_needed = test_needed - len(original_test)
    high_entropy_records = sorted(
        remaining_known,
        key=lambda record: (
            entropy_by_stem.get(record.path.stem) is None,
            -(entropy_by_stem.get(record.path.stem, float("-inf")) or float("-inf")),
            stable_record_key(record),
        ),
    )[:high_entropy_needed]
    if len(high_entropy_records) < high_entropy_needed:
        raise RuntimeError(f"Cannot fill strict test budget: need {high_entropy_needed}, got {len(high_entropy_records)}")
    for record in high_entropy_records:
        assigned[record.path.stem] = "test"

    original_val = [
        record
        for record in known_records
        if record.path.stem not in assigned and original_lookup.get(record.path.stem, (None, None))[0] == "val"
    ]
    if len(original_val) > val_needed:
        raise RuntimeError(f"Original known val records ({len(original_val)}) exceed strict val budget ({val_needed})")
    for record in original_val:
        assigned[record.path.stem] = "val"

    val_candidates = [record for record in known_records if record.path.stem not in assigned]
    extra_val_needed = val_needed - len(original_val)
    selected_val_stems = sample_entropy_matched(
        val_candidates,
        extra_val_needed,
        entropy_by_stem=entropy_by_stem,
        bins=entropy_bins,
        rng=rng,
    )
    for stem in selected_val_stems:
        assigned[stem] = "val"

    for record in sorted(known_records, key=stable_record_key):
        if record.path.stem in assigned:
            continue
        assigned[record.path.stem] = "train"

    for record in sorted(known_records, key=stable_record_key):
        target_split = assigned[record.path.stem]
        if target_split == "test":
            reason = (
                "strict_original_known_test_sample"
                if original_lookup.get(record.path.stem, (None, None))[0] == "test"
                else "strict_high_entropy_known_test_sample"
            )
        elif target_split == "val":
            reason = (
                "strict_original_known_validation_sample"
                if original_lookup.get(record.path.stem, (None, None))[0] == "val"
                else "strict_entropy_matched_known_validation_sample"
            )
        else:
            reason = "strict_known_train_sample"
        add_strict_action(
            actions,
            record=record,
            target_split=target_split,
            open_domain_root=open_domain_root,
            reason=reason,
            seed=seed,
            original_lookup=original_lookup,
            fallback_entropy=fallback_entropy,
        )

    return actions


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, actions: list[Action], dataset_root: Path, copy_files: bool) -> None:
    rows = []
    for action in actions:
        rows.append(
            {
                "class_name": action.class_name,
                "source_split": action.source_split,
                "target_split": action.target_split,
                "source_path": action.source_path.relative_to(dataset_root),
                "target_path": action.target_path.relative_to(dataset_root),
                "action": "copy" if copy_files else "link",
                "reason": action.reason,
                "seed": action.seed,
                "entropy": "" if action.entropy is None else action.entropy,
            }
        )
    write_csv(
        path,
        rows,
        [
            "class_name",
            "source_split",
            "target_split",
            "source_path",
            "target_path",
            "action",
            "reason",
            "seed",
            "entropy",
        ],
    )


def write_summary(
    path: Path,
    by_class: dict[str, list[ImageRecord]],
    actions: list[Action],
    unknown_classes: set[str],
    val_unknown_classes: set[str],
) -> None:
    counts: dict[tuple[str, str], int] = {}
    for action in actions:
        key = (action.class_name, action.target_split)
        counts[key] = counts.get(key, 0) + 1

    rows = []
    for class_name in sorted(by_class, key=lambda name: (len(by_class[name]), name)):
        if class_name in val_unknown_classes:
            status = "val_unknown"
        elif class_name in unknown_classes:
            status = "test_unknown_only"
        else:
            status = "known_retained"
        rows.append(
            {
                "class_name": class_name,
                "total_images_in_all": len(by_class[class_name]),
                "open_domain_train": counts.get((class_name, "train"), 0),
                "open_domain_val": counts.get((class_name, "val"), 0),
                "open_domain_test": counts.get((class_name, "test"), 0),
                "unknown_status": status,
            }
        )

    write_csv(
        path,
        rows,
        [
            "class_name",
            "total_images_in_all",
            "open_domain_train",
            "open_domain_val",
            "open_domain_test",
            "unknown_status",
        ],
    )


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_rollback(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        handle.write("cd \"$(dirname \"$0\")/../..\"\n")
        handle.write("rm -rf -- train val test wds_split manifests logs\n")


def setup_logging(log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logging.info("dry_run=%s", dry_run)


def remove_existing_open_domain(open_domain_root: Path) -> None:
    if open_domain_root.exists():
        shutil.rmtree(open_domain_root)


def materialize_action(action: Action, copy_files: bool) -> None:
    action.target_path.parent.mkdir(parents=True, exist_ok=True)
    if action.target_path.exists() or action.target_path.is_symlink():
        raise FileExistsError(f"Target already exists: {action.target_path}")
    if copy_files:
        shutil.copy2(action.source_path, action.target_path)
    else:
        link_target = Path(os.path.relpath(action.source_path, action.target_path.parent))
        action.target_path.symlink_to(link_target)


def materialize_actions(actions: list[Action], copy_files: bool) -> None:
    for index, action in enumerate(actions, start=1):
        materialize_action(action, copy_files)
        if index % 100000 == 0:
            logging.info("materialized_actions=%d", index)


def split_totals(actions: list[Action]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for action in actions:
        totals[action.target_split] = totals.get(action.target_split, 0) + 1
    return totals


def write_split_info(
    path: Path,
    actions: list[Action],
    by_class: dict[str, list[ImageRecord]],
    unknown_classes: set[str],
    val_unknown_classes: set[str],
) -> None:
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for action in actions:
        item = {
            "path": str(action.target_path),
            "cls": action.class_name,
            "stem": action.source_path.stem,
            "source_path": str(action.source_path),
            "reason": action.reason,
        }
        if action.entropy is not None:
            item["entropy"] = action.entropy
        splits[action.target_split].append(item)

    class_map = {class_name: idx for idx, class_name in enumerate(sorted(by_class))}
    data = {
        "class_map": class_map,
        "unknown_classes": sorted(unknown_classes),
        "val_unknown_classes": sorted(val_unknown_classes),
        "test_only_unknown_classes": sorted(unknown_classes - val_unknown_classes),
        "splits": splits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def tar_add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def latest_split_info(open_domain_root: Path) -> Path:
    candidates = sorted((open_domain_root / "manifests").glob("split_info_*.json"))
    if not candidates:
        raise RuntimeError(f"No split_info_*.json found under {open_domain_root / 'manifests'}")
    return candidates[-1]


def load_split_info_items(split_info_path: Path) -> dict[str, list[dict[str, str]]]:
    with split_info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise RuntimeError(f"Invalid split info JSON: missing object field 'splits' in {split_info_path}")
    return {split: list(splits.get(split, [])) for split in ("train", "val", "test")}


def encode_image(path: Path, target_size: int) -> bytes:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        ratio = target_size / min(width, height)
        resized = (int(width * ratio), int(height * ratio))
        img = img.resize(resized, Image.BILINEAR)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def write_wds_shard(task: tuple[Path, list[dict[str, str]], int]) -> tuple[str, int, int]:
    shard_path, shard_records, target_size = task
    tmp_path = shard_path.with_suffix(".tar.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    written = 0
    skipped = 0
    with tarfile.open(tmp_path, "w") as tar:
        for item in shard_records:
            try:
                source_path = Path(item["source_path"])
                class_name = item["cls"]
                key = item["stem"]
                image_bytes = encode_image(source_path, target_size)
                tar_add_bytes(tar, f"{key}.jpg", image_bytes)
                tar_add_bytes(tar, f"{key}.cls", class_name.encode("utf-8"))
                written += 1
            except Exception as exc:
                skipped += 1
                logging.warning("skipped_wds_item path=%s error=%s", item.get("source_path"), exc)
    tmp_path.rename(shard_path)
    return str(shard_path), written, skipped


def pack_wds_split(
    open_domain_root: Path,
    split_info_path: Path,
    shard_size: int,
    workers: int,
    target_size: int,
) -> dict[str, int]:
    wds_root = open_domain_root / "wds_split"
    if wds_root.exists():
        shutil.rmtree(wds_root)
    wds_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    tasks: list[tuple[Path, list[dict[str, str]], int]] = []
    splits = load_split_info_items(split_info_path)

    for split in ("train", "val", "test"):
        records = splits[split]
        counts[split] = len(records)
        split_dir = wds_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(math.ceil(len(records) / shard_size)):
            shard_records = records[shard_idx * shard_size : (shard_idx + 1) * shard_size]
            shard_path = split_dir / f"{split}-{shard_idx:05d}.tar"
            tasks.append((shard_path, shard_records, target_size))

    logging.info("split_info=%s", split_info_path)
    logging.info("packing_wds_shards=%d workers=%d target_size=%d", len(tasks), workers, target_size)
    with multiprocessing.Pool(workers) as pool:
        for shard_path, written, skipped in pool.imap_unordered(write_wds_shard, tasks):
            logging.info("packed_shard=%s written=%d skipped=%d", shard_path, written, skipped)
    return counts


def main() -> int:
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    open_domain_root = dataset_root / "open_domain"
    manifest_path = open_domain_root / "manifests" / f"migration_plan_{timestamp}.csv"
    summary_path = open_domain_root / "manifests" / f"migration_summary_{timestamp}.csv"
    split_info_path = open_domain_root / "manifests" / f"split_info_{timestamp}.json"
    log_path = open_domain_root / "logs" / f"migration_{timestamp}.log"
    rollback_path = open_domain_root / "scripts" / f"rollback_{timestamp}.sh"

    if args.pack_only:
        setup_logging(log_path, args.dry_run)
        split_info_for_pack = args.split_info.resolve() if args.split_info else latest_split_info(open_domain_root)
        if args.dry_run:
            split_items = load_split_info_items(split_info_for_pack)
            counts = {split: len(split_items[split]) for split in ("train", "val", "test")}
            logging.info("pack_only_dry_run_counts=%s", counts)
        else:
            counts = pack_wds_split(
                open_domain_root,
                split_info_for_pack,
                args.shard_size,
                args.pack_workers,
                args.target_size,
            )
            logging.info("wds_counts=%s", counts)
        logging.info("completed")
        return 0

    by_class = load_all_records(dataset_root)
    if open_domain_root.exists() and any(open_domain_root.iterdir()) and not args.dry_run and not args.replace_open_domain:
        raise RuntimeError(
            f"{open_domain_root} already exists. Use --replace-open-domain to rebuild it, "
            "or --dry-run to only inspect a plan."
        )
    if args.replace_open_domain and not args.dry_run:
        remove_existing_open_domain(open_domain_root)

    setup_logging(log_path, args.dry_run)
    logging.info("dataset_root=%s", dataset_root)
    logging.info("source_layout=all")
    logging.info("classes=%d", len(by_class))
    logging.info("images=%d", sum(len(images) for images in by_class.values()))
    logging.info("split_policy=%s", args.split_policy)
    logging.info("rare_threshold=%d", args.rare_threshold)
    logging.info("val_unknown_ratio=%.4f", args.val_unknown_ratio)
    logging.info("val_image_ratio=%.4f", args.val_image_ratio)
    logging.info("known_val_ratio=%.4f", args.known_val_ratio)
    logging.info("known_test_ratio=%.4f", args.known_test_ratio)
    logging.info("target_val_count=%d", args.target_val_count)
    logging.info("target_test_count=%d", args.target_test_count)
    logging.info("seed=%d", args.seed)

    unknown_classes, val_unknown_classes, test_only_classes = choose_unknown_classes(
        by_class,
        args.rare_threshold,
        args.val_unknown_ratio,
    )
    if args.split_policy == "original-double-cross-entropy":
        actions = plan_original_double_cross_entropy_actions(
            by_class,
            open_domain_root,
            unknown_classes,
            val_unknown_classes,
            seed=args.seed,
            val_image_ratio=args.val_image_ratio,
            original_split_info=args.original_split_info.resolve(),
            cross_entropy_csv=args.cross_entropy_csv.resolve(),
            target_val_count=args.target_val_count,
            target_test_count=args.target_test_count,
            entropy_bins=args.entropy_bins,
        )
    else:
        actions = plan_actions(
            by_class,
            open_domain_root,
            unknown_classes,
            val_unknown_classes,
            seed=args.seed,
            val_image_ratio=args.val_image_ratio,
            known_val_ratio=args.known_val_ratio,
            known_test_ratio=args.known_test_ratio,
        )

    logging.info("unknown_classes=%d", len(unknown_classes))
    logging.info("val_unknown_classes=%d", len(val_unknown_classes))
    logging.info("test_only_unknown_classes=%d", len(test_only_classes))
    logging.info("planned_actions=%d", len(actions))
    if args.split_policy == "original-double-cross-entropy":
        totals = split_totals(actions)
        expected_total = sum(len(images) for images in by_class.values())
        if len(actions) != expected_total:
            raise RuntimeError(f"Strict split lost records: actions={len(actions)} expected={expected_total}")
        if totals.get("val", 0) != args.target_val_count or totals.get("test", 0) != args.target_test_count:
            raise RuntimeError(
                "Strict split missed target counts: "
                f"totals={totals} target_val={args.target_val_count} target_test={args.target_test_count}"
            )

    write_manifest(manifest_path, actions, dataset_root, args.copy)
    write_summary(summary_path, by_class, actions, unknown_classes, val_unknown_classes)
    write_split_info(split_info_path, actions, by_class, unknown_classes, val_unknown_classes)
    write_rollback(rollback_path)
    rollback_path.chmod(0o755)

    if not args.dry_run:
        materialize_actions(actions, args.copy)
        if args.skip_pack:
            logging.info("skip_pack=True")
        else:
            wds_counts = pack_wds_split(
                open_domain_root,
                split_info_path,
                args.shard_size,
                args.pack_workers,
                args.target_size,
            )
            logging.info("wds_counts=%s", wds_counts)

    totals = split_totals(actions)
    logging.info("target_split_totals=%s", totals)
    logging.info("manifest=%s", manifest_path)
    logging.info("summary=%s", summary_path)
    logging.info("split_info=%s", split_info_path)
    logging.info("rollback=%s", rollback_path)
    logging.info("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
