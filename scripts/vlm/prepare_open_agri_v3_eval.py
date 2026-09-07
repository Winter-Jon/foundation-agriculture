#!/usr/bin/env python3
"""Create public-only bilingual OpenAgri v3 evaluation manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/test_public.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/artifacts/datasets/open-agri-v3-evaluation-v1"
USERS = {
    "en": "<image> Task: identify the image with one canonical disease or pest name. Answer with only the canonical name.",
    "zh": "<image> 任务：根据图像给出一个规范的病害或虫害名称。请只输出规范名称。",
}
SYSTEM = "You identify agricultural diseases and pests from images. Follow the user task constraints. Return the final result as <think>...</think><answer>...</answer>."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable evaluation artifact: {output}")
    source_rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(source_rows) != 1019 or len({row["id"] for row in source_rows}) != len(source_rows):
        raise SystemExit("v3 public test must contain exactly 1,019 unique rows")
    output.mkdir(parents=True)
    manifests = {}
    for language, prompt in USERS.items():
        path = output / f"test_{language}.jsonl"
        rows = []
        for row in source_rows:
            images, metadata = row.get("images") or [], row.get("metadata") or {}
            if len(images) != 1 or not isinstance(images[0], str):
                raise ValueError(f"{row.get('id')}: invalid public image")
            rows.append({"id": row["id"], "image_path": images[0], "image_sha256": row["image_sha256"],
                         "language": language, "task_domain": metadata.get("domain"), "question_type": "open",
                         "question": prompt, "class_role": row["class_role"], "evaluation_bucket": row["evaluation_bucket"]})
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        manifests[language] = {"path": display_path(path), "rows": len(rows), "sha256": sha(path),
                               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    (output / "direct_system.xml.txt").write_text(SYSTEM + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"schema_version": "agrinet.open-agri-v3-evaluation/v1", "public_only": True,
        "source": display_path(SOURCE), "source_sha256": sha(SOURCE), "rows": len(source_rows),
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "manifests": manifests}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
