from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agrinet.data.io import DataError, write_jsonl_atomic


def load_wiki_base(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read AgriNet wiki base {path}: {exc}") from exc
    descriptions = payload.get("description") if isinstance(payload, dict) else None
    if not isinstance(descriptions, dict):
        raise DataError(f"AgriNet wiki base has no description mapping: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for value in descriptions.values():
        if isinstance(value, dict) and value.get("code"):
            rows[str(value["code"])] = value
    return rows


def _evidence(row: dict[str, Any]) -> list[str]:
    description = row.get("description") or {}
    if not isinstance(description, dict):
        return []
    return [str(value).strip() for key, value in sorted(description.items()) if key.startswith("content_") and value]


def _images(class_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def prepare_bounded_contrast(
    data_root: Path,
    wiki_path: Path,
    output_dir: Path,
    class_count: int = 8,
    samples_per_class: int = 1,
    candidate_count: int = 4,
    seed: int = 42,
) -> dict[str, int]:
    if class_count < candidate_count * 2:
        raise DataError("class_count must provide at least candidate_count classes per disease/pest domain")
    if samples_per_class < 1 or candidate_count < 2:
        raise DataError("samples_per_class must be positive and candidate_count at least 2")
    wiki = load_wiki_base(wiki_path)
    eligible: list[dict[str, Any]] = []
    for code, row in sorted(wiki.items()):
        if not (code.startswith("N04") or code.startswith("N05")):
            continue
        images = _images(data_root / "all" / code) if (data_root / "all" / code).is_dir() else []
        evidence = _evidence(row)
        if len(images) >= samples_per_class + 1 and evidence:
            eligible.append({"code": code, "row": row, "images": images, "evidence": evidence})
    if len(eligible) < class_count:
        raise DataError(f"requested {class_count} classes but only {len(eligible)} are eligible")
    rng = random.Random(seed)
    disease = [item for item in eligible if item["code"].startswith("N04")]
    pest = [item for item in eligible if item["code"].startswith("N05")]
    disease_count = class_count // 2 + class_count % 2
    pest_count = class_count // 2
    if disease_count < candidate_count or pest_count < candidate_count:
        raise DataError("bounded selection would not provide enough same-domain candidates")
    if len(disease) < disease_count or len(pest) < pest_count:
        raise DataError("not enough eligible disease or pest classes for stratified selection")
    selected = sorted(
        [*rng.sample(disease, disease_count), *rng.sample(pest, pest_count)], key=lambda item: item["code"]
    )
    classes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        code = item["code"]
        domain = "disease" if code.startswith("N04") else "pest"
        row = item["row"]
        classes.append(
            {
                "code": code,
                "task_domain": domain,
                "english_name": row.get("english_name", code),
                "chinese_name": row.get("chinese_name", ""),
                "wiki_available": True,
                "wiki_evidence": item["evidence"],
            }
        )
        same_domain = [candidate for candidate in selected if candidate["code"] != code and candidate["code"].startswith(code[:3])]
        negatives = same_domain[: candidate_count - 1]
        pairs.append({"anchor": code, "task_domain": domain, "similar_classes": [n["code"] for n in negatives]})
        chosen_images = rng.sample(item["images"], samples_per_class + 1)
        for sample_index in range(samples_per_class):
            candidates = [item, *negatives]
            samples.append(
                {
                    "sample_id": f"bounded-{code}-{sample_index + 1:02d}",
                    "query_image": str(chosen_images[sample_index]),
                    "task_domain": domain,
                    "final_label": code,
                    "final_label_zh": row.get("chinese_name", ""),
                    "candidate_labels": [
                        {"code": c["code"], "english_name": c["row"].get("english_name", c["code"]), "chinese_name": c["row"].get("chinese_name", "")}
                        for c in candidates
                    ],
                    "positive_reference_images": [str(chosen_images[-1])],
                    "negative_reference_images": [
                        {"code": n["code"], "image_path": str(n["images"][0])} for n in negatives
                    ],
                    "wiki_available": True,
                    "wiki_evidence": item["evidence"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "classes.jsonl", classes)
    write_jsonl_atomic(output_dir / "pairs.jsonl", pairs)
    write_jsonl_atomic(output_dir / "samples.jsonl", samples)
    return {"classes": len(classes), "pairs": len(pairs), "samples": len(samples)}


def validate_bounded_contrast(directory: Path) -> dict[str, int]:
    def rows(name: str) -> list[dict[str, Any]]:
        path = directory / name
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot validate {path}: {exc}") from exc

    classes, pairs, samples = rows("classes.jsonl"), rows("pairs.jsonl"), rows("samples.jsonl")
    codes = {str(row.get("code")) for row in classes}
    if len(codes) != len(classes) or not codes:
        raise DataError("bounded classes must contain unique non-empty codes")
    for row in pairs:
        anchor = str(row.get("anchor", ""))
        candidates = [str(value) for value in row.get("similar_classes", [])]
        if anchor not in codes or any(value not in codes for value in candidates):
            raise DataError(f"pair references an unknown class: {anchor}")
        if any(value[:3] != anchor[:3] for value in candidates):
            raise DataError(f"pair contains a cross-domain candidate: {anchor}")
    for row in samples:
        label = str(row.get("final_label", ""))
        candidate_codes = [str(value.get("code", "")) for value in row.get("candidate_labels", [])]
        if label not in candidate_codes or any(value[:3] != label[:3] for value in candidate_codes):
            raise DataError(f"invalid candidates for sample {row.get('sample_id')}")
        image_paths = [
            row.get("query_image"),
            *row.get("positive_reference_images", []),
            *[value.get("image_path") for value in row.get("negative_reference_images", [])],
        ]
        if any(not value or not Path(value).is_file() for value in image_paths):
            raise DataError(f"missing image for sample {row.get('sample_id')}")
    return {"classes": len(classes), "pairs": len(pairs), "samples": len(samples)}
