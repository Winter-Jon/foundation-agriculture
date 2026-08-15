#!/usr/bin/env python3
"""Build and validate the reviewed 48-RAG + 32-Direct mixed SFT freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ZH = re.compile(r"[\u3400-\u9fff]")
OPTION = re.compile(r"(?:\b[A-D][\).:]|\boptions?\b|选项|选择题)", re.I)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def language(row: dict[str, Any]) -> str:
    user = next((str(m.get("content", "")) for m in row.get("messages", []) if m.get("role") == "user"), "")
    return "zh" if ZH.search(user) else "en"

def domain(row: dict[str, Any]) -> str:
    image = str((row.get("images") or [""])[0]).replace("\\", "/")
    if "/N050" in image: return "pest"
    if "/N040" in image: return "disease"
    raise ValueError(f"cannot classify Direct domain from image: {image}")

def question_type(row: dict[str, Any]) -> str:
    text = " ".join(str(m.get("content", "")) for m in row.get("messages", []) if m.get("role") == "user")
    return "option" if OPTION.search(text) else "open"

def classify_direct(row: dict[str, Any]) -> tuple[str, str, str]:
    msgs, images = row.get("messages"), row.get("images")
    if not isinstance(msgs, list) or not isinstance(images, list) or len(images) != 1:
        raise ValueError("direct row must have messages and exactly one image")
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles or "system" not in roles:
        raise ValueError("direct row has incomplete message roles")
    lang = language(row)
    assistant = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "assistant")
    if lang == "en" and ZH.search(assistant): raise ValueError("English Direct row contains Chinese text")
    if lang == "zh" and not ZH.search(assistant): raise ValueError("Chinese Direct row has no Chinese text")
    return lang, domain(row), question_type(row)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def image_key(row: dict[str, Any]) -> str:
    return str((row.get("images") or [""])[0]).replace("\\", "/")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-file", type=Path, required=True)
    ap.add_argument("--direct-file", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, default=8)
    args = ap.parse_args()
    rag, direct = read_jsonl(args.rag_file), read_jsonl(args.direct_file)
    if len(rag) != 48: raise SystemExit(f"RAG freeze must contain exactly 48 rows, got {len(rag)}")
    rag_images = [image_key(r) for r in rag]
    if len(set(rag_images)) != len(rag_images): raise SystemExit("RAG freeze contains duplicate images")
    rag_image_set = set(rag_images)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for idx, row in enumerate(direct):
        try:
            cell = classify_direct(row)
            if image_key(row) in rag_image_set: raise ValueError("image overlaps RAG freeze")
            buckets[cell].append(row)
        except ValueError as exc: rejected.append({"index": idx, "reason": str(exc)})
    # The retained Direct artifact is an open-question corpus.  Do not infer or
    # synthesize Direct Option rows from answer text; Option supervision belongs
    # to the RAG side and its four-choice contract is audited there.
    required = [(lang, dom, "open") for lang in ("en", "zh") for dom in ("disease", "pest")]
    selected: list[dict[str, Any]] = []
    cell_report: dict[str, Any] = {}
    globally_selected_images: set[str] = set()
    for cell in required:
        rows = sorted(buckets[cell], key=image_key)
        unique_rows = []
        seen_cell_images = set()
        for row in rows:
            if image_key(row) in seen_cell_images:
                continue
            seen_cell_images.add(image_key(row))
            unique_rows.append(row)
        rows = unique_rows
        available_unique = [r for r in rows if image_key(r) not in globally_selected_images]
        if len(available_unique) < args.per_cell: raise SystemExit(f"Direct cell {cell} has only {len(available_unique)} globally unique rows; need {args.per_cell}")
        chosen = available_unique[:args.per_cell]
        for rank, row in enumerate(chosen):
            if image_key(row) in globally_selected_images:
                raise SystemExit(f"Direct image selected in multiple cells: {image_key(row)}")
            globally_selected_images.add(image_key(row))
            item = dict(row); meta = dict(item.get("metadata") or {})
            meta.update({"source_route":"direct", "language":cell[0], "task_domain":cell[1], "question_type":cell[2], "selection_policy":"deterministic_image_path_sort", "selection_rank":rank})
            item["metadata"] = meta; selected.append(item)
        cell_report["/".join(cell)] = {"available":len(rows), "selected":len(chosen), "selected_images":[image_key(r) for r in chosen]}
    direct_images = [image_key(r) for r in selected]
    if len(set(direct_images)) != len(direct_images) or set(direct_images) & rag_image_set: raise SystemExit("mixed freeze image uniqueness failed")
    mixed = rag + selected; args.destination.mkdir(parents=True, exist_ok=True); data = args.destination / "data.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for row in mixed: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    direct_cov = {"/".join(c): sum(1 for r in selected if (r["metadata"]["language"], r["metadata"]["task_domain"], r["metadata"]["question_type"]) == c) for c in required}
    rag_cov = Counter("/".join((str((r.get("metadata") or {}).get("language")), str((r.get("metadata") or {}).get("task_domain")), str((r.get("metadata") or {}).get("question_type")))) for r in rag)
    stats = {"rows":len(mixed), "rag_rows":len(rag), "direct_rows":len(selected), "unique_images":len(set(rag_images + direct_images)), "rag_coverage":dict(rag_cov), "direct_coverage":direct_cov, "cell_report":cell_report, "rejected_direct_rows":len(rejected), "valid":len(mixed)==80 and len(set(rag_images + direct_images))==80}
    (args.destination / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"artifact_id":args.destination.name, "data_sha256":sha256(data), "source_rag":str(args.rag_file), "source_direct":str(args.direct_file), "selection_policy":"8 Direct rows per language/domain cell (open-only), deterministic image-path order", "direct_option_rows":0, "training_authorized":False}
    (args.destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.destination / "direct_selection_rejections.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact":str(args.destination), "rows":len(mixed), "sha256":manifest["data_sha256"], "valid":stats["valid"], "coverage":direct_cov}, ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
