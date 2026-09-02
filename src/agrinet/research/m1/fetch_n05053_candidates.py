#!/usr/bin/env python3
"""Fetch licensed Saccharicoccus sacchari candidates with provenance."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path("/tmp/inat_obs_n05053.json")
OUTPUT = ROOT / "outputs/artifacts/datasets/m1-direct-current-hcv-v1/n05053_candidates"


def original_url(url: str) -> str:
    return url.replace("/square.", "/original.")


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for observation in payload.get("results") or []:
        for index, photo in enumerate(observation.get("photos") or [], 1):
            if not str(photo.get("license_code") or "").startswith("cc-"):
                continue
            url = original_url(str(photo["url"]))
            suffix = Path(url).suffix.lower() or ".jpg"
            destination = OUTPUT / f"inat-{observation['id']}-{index}{suffix}"
            urllib.request.urlretrieve(url, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            rows.append({
                "class_code": "N05053", "scientific_name": "Saccharicoccus sacchari",
                "occurrence_id": str(observation["id"]), "quality_grade": observation.get("quality_grade"),
                "observed_on": observation.get("observed_on"), "source_url": observation.get("uri"),
                "media_url": url, "license": photo.get("license_code"),
                "attribution": photo.get("attribution"),
                "candidate_path": str(destination.relative_to(ROOT)), "image_sha256": digest,
                "preflight_status": "pending_visual_and_label_review",
            })
    manifest = OUTPUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    print(json.dumps({"candidates": len(rows), "unique_occurrences": len({row['occurrence_id'] for row in rows}), "manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
