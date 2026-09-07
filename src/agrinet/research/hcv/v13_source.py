"""Prepare a fresh, image-isolated HCV v13 pilot source and bootstrap calibration.

This local-only operation is intentionally narrow: it selects exactly four
fresh images for each question-type/language/domain cell, hashes image bytes,
and writes a private truth sidecar.  It never contacts Micu or retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from agrinet.research.hcv.v13_calibration import bootstrap_calibration
from agrinet.research.hcv.v13_plan import CELLS, PRIVATE_KEYS, cell_key


REPO_ROOT = Path(__file__).resolve().parents[4]
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DEFAULT_CLASSES = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/classes.jsonl"
DEFAULT_ISOLATION_DIR = REPO_ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/isolation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hashes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values = set()
    for row in read_jsonl(path):
        value = str(row.get("image_sha256") or row.get("sha256") or "").strip()
        if value:
            values.add(value)
    return values


def load_excluded_hashes(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                excluded.update(_row_hashes(child))
        else:
            excluded.update(_row_hashes(path))
    return excluded


def _order_key(seed: str, cell: str, digest: str) -> str:
    return hashlib.sha256(f"{seed}:{cell}:{digest}".encode()).hexdigest()


def _options(classes: dict[str, dict[str, Any]], code: str, sample_id: str) -> tuple[list[dict[str, str]], str]:
    truth = classes[code]
    pool = [row for item_code, row in classes.items() if item_code != code and row["task_domain"] == truth["task_domain"]]
    if len(pool) < 3:
        raise ValueError(f"not enough {truth['task_domain']} distractors for {code}")
    selected = sorted(pool, key=lambda row: _order_key(sample_id, "option", str(row["code"])))[:3]
    values = [truth, *selected]
    values.sort(key=lambda row: _order_key(sample_id, "option-order", str(row["code"])))
    labels = [{"name": str(row["english_name"]), "name_zh": str(row["chinese_name"])} for row in values]
    correct = chr(65 + next(index for index, row in enumerate(values) if row["code"] == code))
    return labels, correct


def build_source(
    classes: list[dict[str, Any]], *, excluded_hashes: set[str], images_root: Path,
    per_cell: int = 4, seed: str = "hcv-v13-r1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build public-plus-private source rows from images not present in any ledger."""
    if per_cell != 4:
        raise ValueError("HCV v13 source preparation requires exactly four rows per cell")
    catalog = {str(row.get("code") or ""): row for row in classes}
    if not catalog or "" in catalog or len(catalog) != len(classes):
        raise ValueError("class catalog has missing or duplicate code")
    for code, row in catalog.items():
        if row.get("task_domain") not in {"disease", "pest"} or not row.get("english_name") or not row.get("chinese_name"):
            raise ValueError(f"invalid class catalog row: {code}")

    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    counts: Counter[str] = Counter()
    scanned = skipped = 0
    for cell in CELLS:
        question_type, language, domain = cell.split("/")
        codes = sorted(
            (code for code, row in catalog.items() if row["task_domain"] == domain),
            key=lambda code: _order_key(seed, cell, code),
        )
        for code in codes:
            if counts[cell] >= per_cell:
                break
            directory = images_root / code
            if not directory.is_dir():
                continue
            for image in sorted(directory.iterdir(), key=lambda value: _order_key(seed, cell, value.name)):
                if counts[cell] >= per_cell:
                    break
                if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                scanned += 1
                digest = sha256_file(image)
                if digest in excluded_hashes or digest in used:
                    skipped += 1
                    continue
                sample_id = "hcv-v13-" + hashlib.sha256(f"{seed}:{cell}:{digest}".encode()).hexdigest()[:20]
                try:
                    query_image = str(image.relative_to(REPO_ROOT))
                except ValueError:
                    query_image = str(image)
                row = {
                    "sample_id": sample_id, "query_image": query_image,
                    "image_sha256": digest, "question_type": question_type, "language": language,
                    "task_domain": domain, "source_class_code": code, "source_status": "fresh_isolated",
                    "audit_truth_code": code, "audit_truth_name": str(catalog[code]["english_name"]),
                    "audit_truth_name_zh": str(catalog[code]["chinese_name"]),
                }
                if question_type == "option":
                    labels, correct = _options(catalog, code, sample_id)
                    row["public_option_labels"] = labels
                    row["audit_correct_option"] = correct
                rows.append(row)
                used.add(digest); counts[cell] += 1
    shortages = {cell: per_cell - counts[cell] for cell in CELLS if counts[cell] < per_cell}
    report = {
        "schema_version": "agrinet.hcv-v13-source-preparation/v1",
        "rows": len(rows), "per_cell": dict(counts), "shortages": shortages,
        "scanned_images": scanned, "excluded_or_duplicate_images": skipped,
        "unique_image_hashes": len(used) == len(rows),
        "ready": len(rows) == 32 and not shortages,
        "bootstrap_calibration_mode": "bootstrap_force_rag",
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--images-root", type=Path, default=REPO_ROOT / "datasets/AgriNet-1K/all")
    parser.add_argument("--exclude", type=Path, action="append", default=[DEFAULT_ISOLATION_DIR])
    parser.add_argument("--seed", default="hcv-v13-r1")
    args = parser.parse_args()
    rows, report = build_source(read_jsonl(args.classes), excluded_hashes=load_excluded_hashes(args.exclude), images_root=args.images_root, seed=args.seed)
    if not report["ready"]:
        raise ValueError(f"v13 fresh pilot source lacks capacity: {report['shortages']}")
    write_jsonl(args.output, rows)
    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_output.write_text(json.dumps(bootstrap_calibration(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
