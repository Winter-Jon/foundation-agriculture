"""Build an image-isolated, pilot-only HCV v13 collection plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CELLS = tuple(
    f"{question_type}/{language}/{domain}"
    for question_type in ("open", "option")
    for language in ("en", "zh")
    for domain in ("disease", "pest")
)
PRIVATE_KEYS = frozenset({
    "final_label", "final_label_name", "final_label_zh", "correct_option",
    "audit_truth_code", "audit_truth_name", "audit_truth_name_zh", "audit_correct_option",
    "class_code", "canonical_class", "truth_name", "truth_name_zh",
    "source_class_code",
})


def cell_key(row: dict[str, Any]) -> str:
    return "/".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))


def image_hash(row: dict[str, Any]) -> str:
    supplied = str(row.get("image_sha256") or "").strip()
    if supplied:
        return supplied
    image = Path(str(row.get("query_image") or "").strip())
    if not image.is_file():
        raise ValueError("candidate row requires an existing query_image or image_sha256")
    digest = hashlib.sha256()
    with image.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key not in PRIVATE_KEYS}
    result["image_sha256"] = image_hash(row)
    return result


def private_row(row: dict[str, Any]) -> dict[str, Any]:
    truth = {key: row[key] for key in PRIVATE_KEYS if key in row}
    if not truth:
        raise ValueError(f"candidate row has no private truth: {row.get('sample_id')}")
    return {"sample_id": row.get("sample_id"), **truth}


def build_plan(rows: list[dict[str, Any]], *, excluded_hashes: set[str] | None = None, per_cell: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if per_cell != 4:
        raise ValueError("HCV v13 pilot requires exactly four rows per task cell")
    excluded = excluded_hashes or set()
    selected, private, seen = [], [], set(excluded)
    counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
        cell = cell_key(row)
        digest = image_hash(row)
        if cell not in CELLS or counts[cell] >= per_cell or digest in seen:
            continue
        if not str(row.get("sample_id") or "") or not str(row.get("query_image") or ""):
            continue
        selected.append(public_row(row))
        private.append(private_row(row))
        seen.add(digest)
        counts[cell] += 1
    shortages = {cell: per_cell - counts[cell] for cell in CELLS if counts[cell] < per_cell}
    report = {
        "schema_version": "agrinet.hcv-v13-pilot-plan/v1",
        "rows": len(selected), "per_cell": dict(counts), "shortages": shortages,
        "unique_image_hashes": len({row["image_sha256"] for row in selected}) == len(selected),
        "public_private_ids_aligned": [row.get("sample_id") for row in selected] == [row.get("sample_id") for row in private],
        "ready": len(selected) == 32 and not shortages,
    }
    return selected, private, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-hashes", type=Path)
    args = parser.parse_args()
    excluded = set()
    if args.exclude_hashes:
        excluded = {str(item.get("image_sha256") or "") for item in _read_jsonl(args.exclude_hashes)}
        excluded.discard("")
    public, private, report = build_plan(_read_jsonl(args.source), excluded_hashes=excluded)
    if not report["ready"]:
        raise ValueError(f"v13 pilot plan cannot be built: {report['shortages']}")
    _write_jsonl(args.output_root / "public" / "collection_plan.jsonl", public)
    _write_jsonl(args.output_root / "private" / "truth_alignment.jsonl", private)
    (args.output_root / "reports").mkdir(parents=True, exist_ok=True)
    (args.output_root / "reports" / "plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
