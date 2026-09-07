#!/usr/bin/env python3
"""Build public-only bilingual OpenAgri v2 evaluation manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V2 = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2/vlm_data/accepted"
OUT = REPO_ROOT / "outputs/artifacts/open-agri-v2-sft-ablation-v1/evaluation"
USERS = {
    "en": "<image> Task: identify the image with one canonical disease or pest name. Answer with only the canonical name.",
    "zh": "<image> 任务：根据图像给出一个规范的病害或虫害名称。请只输出规范名称。",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if root.exists():
        if not args.replace:
            raise SystemExit(f"output exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    all_meta = {}
    for split, source_name in (("dev", "dev.jsonl"), ("test", "test_public.jsonl")):
        source = V2 / source_name
        rows = load(source)
        for language, prompt in USERS.items():
            out = root / f"{split}_{language}.jsonl"
            public_rows = []
            for row in rows:
                images = row.get("images") or []
                if len(images) != 1 or not isinstance(images[0], str):
                    raise ValueError(f"{source}: invalid public image row")
                public_rows.append({
                    "id": str(row["id"]), "image_path": images[0],
                    "image_sha256": row.get("image_sha256"), "language": language,
                    "task_domain": "agriculture", "question_type": "open", "question": prompt,
                })
            out.write_text(
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in public_rows),
                encoding="utf-8",
            )
            all_meta[f"{split}_{language}"] = {
                "path": str(out.relative_to(REPO_ROOT)), "rows": len(public_rows),
                "sha256": sha(out), "source_sha256": sha(source),
                "user_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": "agrinet.open-agri-v2-public-eval/v1", "public_only": True, "manifests": all_meta}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
