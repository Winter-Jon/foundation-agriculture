"""Create an immutable 1k image-view derivative for unified auto-route SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from agrinet.data.io import DataError, write_jsonl_atomic

PARENT_ID = "agrinet-e343-three-route-sft-v5-balanced"
ARTIFACT_ID = "agrinet-e343-three-route-sft-v6-balanced-image1k"
IMAGE_MAX_SIDE = 1024


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_image(source: Path, destination: Path) -> tuple[str, list[int]]:
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
            if max(image.size) > IMAGE_MAX_SIDE:
                raise DataError(f"resize limit not met: {source}")
            image.save(destination, format="JPEG", quality=88, optimize=True)
            size = list(image.size)
    except Exception as exc:
        raise DataError(f"cannot create 1k view for {source}: {exc}") from exc
    return digest(destination), size


def build(*, parent: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise DataError(f"refuse existing immutable destination: {output}")
    manifest = yaml.safe_load((parent / "artifact.yaml").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != PARENT_ID or manifest.get("data_sha256") != digest(parent / "data.jsonl"):
        raise DataError("parent artifact identity or data hash mismatch")
    rows = read_jsonl(parent / "data.jsonl")
    lineage = read_jsonl(parent / "lineage.jsonl")
    if len(rows) != 495 or len(lineage) != len(rows):
        raise DataError("parent row coverage mismatch")
    if {str(item.get("sample_id")) for item in rows} != {str(item.get("sample_id")) for item in lineage}:
        raise DataError("parent row/lineage IDs mismatch")
    output.mkdir(parents=True); image_root = output / "images"; image_root.mkdir()
    views: dict[str, dict[str, Any]] = {}
    derived_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        images = row.get("images")
        if not sample_id or not isinstance(images, list) or len(images) != 1:
            raise DataError(f"row needs exactly one image: {sample_id}")
        source = Path(str(images[0])).resolve()
        if not source.is_file():
            raise DataError(f"missing source image: {source}")
        destination = image_root / f"{sample_id}.jpg"
        derived_sha, size = render_image(source, destination)
        views[sample_id] = {"source_path": str(source), "source_sha256": digest(source),
                            "derived_path": str(destination.resolve()), "derived_sha256": derived_sha,
                            "derived_size": size, "max_side": IMAGE_MAX_SIDE,
                            "transport": "rgb_jpeg_lanczos_q88"}
        derived_rows.append({**row, "images": [str(destination.resolve())]})
    derived_lineage = [{**item, "image_view": views[str(item["sample_id"])]} for item in lineage]
    write_jsonl_atomic(output / "data.jsonl", derived_rows)
    write_jsonl_atomic(output / "lineage.jsonl", derived_lineage)
    for filename in ("exclusions.jsonl", "evaluation-isolation.json"):
        (output / filename).write_bytes((parent / filename).read_bytes())
    stats = {"rows": len(rows), "routes": manifest["statistics"]["routes"],
             "image_max_side": IMAGE_MAX_SIDE, "image_transport": "rgb_jpeg_lanczos_q88",
             "training_eligible": True, "training_authorized": True, "sft_may_start": True}
    artifact = {"schema_version": "agrinet.sft.frozen/v1k-image-view", "artifact_id": ARTIFACT_ID,
                "artifact_type": "datasets", "immutable": True, "parent_artifact_id": PARENT_ID,
                "parent_manifest_sha256": digest(parent / "artifact.yaml"), "statistics": stats,
                "data_sha256": digest(output / "data.jsonl"), "lineage_sha256": digest(output / "lineage.jsonl"),
                "exclusions_sha256": digest(output / "exclusions.jsonl"),
                "image_max_side": IMAGE_MAX_SIDE, "image_transport": "rgb_jpeg_lanczos_q88",
                **{key: manifest[key] for key in ("protocol_version", "system_prompt_sha256", "tool_schema_sha256")},
                "training_eligible": True, "training_authorized": True, "sft_may_start": True}
    (output / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "artifact.yaml").write_text(yaml.safe_dump(artifact, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "README.md").write_text(f"# {ARTIFACT_ID}\n\nImmutable 495-row v5 derivative with one 1024px-max image view per row.\n", encoding="utf-8")
    return {"artifact_dir": str(output), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(parent=args.parent, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
