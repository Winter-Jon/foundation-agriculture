import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CODE_RE = re.compile(r"^N(\d{2})\d{3}$")
DOMAIN_BY_MAJOR = {"04": "disease", "05": "pest"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_wiki(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_json(path)
    return {str(row["code"]): row for row in rows if row.get("code")}


def ok_wiki_contents(row: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    for key, value in sorted(row.items()):
        if not key.startswith("status_") or value != "ok":
            continue
        suffix = key.split("_", 1)[1]
        content = row.get(f"content_{suffix}")
        if content:
            evidence.append(str(content))
    return evidence


def scan_domain_classes(all_root: Path, wiki_by_code: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_domain: Dict[str, List[Dict[str, Any]]] = {"disease": [], "pest": []}
    for class_dir in sorted(all_root.iterdir()):
        if not class_dir.is_dir():
            continue
        match = CODE_RE.match(class_dir.name)
        if not match:
            continue
        domain = DOMAIN_BY_MAJOR.get(match.group(1))
        if not domain:
            continue
        wiki = wiki_by_code.get(class_dir.name)
        if wiki:
            evidence = ok_wiki_contents(wiki)
            wiki_available = bool(evidence)
            name = wiki.get("name") or wiki.get("english_name") or class_dir.name
            chinese_name = wiki.get("chinese_name") or ""
        else:
            evidence = []
            wiki_available = False
            name = class_dir.name
            chinese_name = ""

        rows_by_domain[domain].append(
            {
                "code": class_dir.name,
                "label": class_dir.name,
                "name": name,
                "english_name": wiki.get("english_name", name) if wiki else name,
                "chinese_name": chinese_name,
                "task_domain": domain,
                "wiki_available": wiki_available,
                "wiki_evidence": evidence if evidence else ["insufficient wiki knowledge"],
                "wiki_sources": [
                    {
                        "source": wiki.get(f"source_{i}"),
                        "title": wiki.get(f"title_{i}"),
                        "url": wiki.get(f"url_{i}"),
                    }
                    for i in range(1, 6)
                    if wiki and wiki.get(f"status_{i}") == "ok"
                ],
            }
        )
    return rows_by_domain


def write_coverage_report(path: Path, rows_by_domain: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = ["# AgriNet Disease/Pest Wiki Coverage", ""]
    for domain in ["disease", "pest"]:
        rows = rows_by_domain[domain]
        missing = [row["code"] for row in rows if not row["wiki_available"]]
        lines.extend(
            [
                f"## {domain}",
                "",
                f"- classes: {len(rows)}",
                f"- wiki_available: {len(rows) - len(missing)}",
                f"- wiki_missing_or_no_ok_content: {len(missing)}",
                f"- missing_codes: {', '.join(missing) if missing else '(none)'}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_representatives(path: Path) -> Dict[str, List[str]]:
    data = read_json(path)
    out: Dict[str, List[str]] = {}
    for code, payload in data.items():
        reps = []
        for rep in payload.get("representatives", []):
            image_path = rep.get("image_path")
            if image_path:
                reps.append(str(image_path))
        out[code] = reps
    return out


def iter_images(class_dir: Path) -> List[str]:
    return [
        str(path)
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def compute_pairs(feature_paths: Sequence[Path], class_codes: Sequence[str], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required to compute ViT hard-negative pairs.") from exc

    class_codes = list(class_codes)
    sums = None
    counts = None
    code_to_pos = {code: i for i, code in enumerate(class_codes)}

    for feature_path in feature_paths:
        payload = torch.load(feature_path, map_location="cpu")
        class_ids = list(payload["class_ids"])
        compact = torch.full((len(class_ids),), -1, dtype=torch.long)
        for class_idx, code in enumerate(class_ids):
            pos = code_to_pos.get(code)
            if pos is not None:
                compact[class_idx] = pos

        class_indices = payload["class_indices"].long()
        compact_indices = compact[class_indices]
        mask = compact_indices >= 0
        if not bool(mask.any()):
            continue

        features = payload["features"][mask].float()
        compact_indices = compact_indices[mask]
        if sums is None:
            sums = torch.zeros((len(class_codes), features.shape[1]), dtype=torch.float32)
            counts = torch.zeros((len(class_codes),), dtype=torch.long)
        sums.index_add_(0, compact_indices, features)
        counts.index_add_(0, compact_indices, torch.ones_like(compact_indices, dtype=torch.long))

    if sums is None or counts is None:
        return {code: [] for code in class_codes}

    valid = counts > 0
    prototypes = sums / counts.clamp_min(1).unsqueeze(1)
    prototypes = F.normalize(prototypes, dim=1)
    sim = prototypes @ prototypes.T
    sim.fill_diagonal_(-2.0)

    pairs: Dict[str, List[Dict[str, Any]]] = {}
    for i, code in enumerate(class_codes):
        if not bool(valid[i]):
            pairs[code] = []
            continue
        order = torch.argsort(sim[i], descending=True)
        negatives: List[Dict[str, Any]] = []
        for j in order.tolist():
            if not bool(valid[j]):
                continue
            negatives.append({"code": class_codes[j], "similarity": float(sim[i, j])})
            if len(negatives) >= top_k:
                break
        pairs[code] = negatives
    return pairs


def build_samples(
    rows_by_domain: Dict[str, List[Dict[str, Any]]],
    pairs_by_code: Dict[str, List[Dict[str, Any]]],
    representatives: Dict[str, List[str]],
    all_root: Path,
    samples_per_class: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows_by_code = {row["code"]: row for rows in rows_by_domain.values() for row in rows}
    warnings: List[str] = []
    samples: List[Dict[str, Any]] = []
    for domain in ["disease", "pest"]:
        for row in rows_by_domain[domain]:
            code = row["code"]
            reps = representatives.get(code, [])
            rep_set = set(reps)
            images = [path for path in iter_images(all_root / code) if path not in rep_set]
            if not images:
                images = iter_images(all_root / code)
                warnings.append(f"{code}: query fallback includes representative candidates")
            if not images:
                warnings.append(f"{code}: no query images available")
                continue
            negatives = [neg for neg in pairs_by_code.get(code, []) if representatives.get(neg["code"])]
            if len(negatives) < 3:
                warnings.append(f"{code}: only {len(negatives)} representative-backed negatives")
            chosen_negatives = negatives[:3]
            candidate_codes = [code] + [neg["code"] for neg in chosen_negatives]
            for image_path in images[:samples_per_class]:
                positive_refs = [rep for rep in reps if rep != image_path][:2]
                if len(positive_refs) < 2:
                    warnings.append(f"{code}: only {len(positive_refs)} non-query positive representatives for {Path(image_path).name}")
                samples.append(
                    {
                        "sample_id": f"{domain}_{code}_{Path(image_path).stem}",
                        "task_domain": domain,
                        "query_image": image_path,
                        "positive_reference_images": positive_refs,
                        "negative_reference_images": [
                            {
                                "code": neg["code"],
                                "image_path": representatives[neg["code"]][0],
                                "similarity": neg["similarity"],
                            }
                            for neg in chosen_negatives
                        ],
                        "candidate_labels": [
                            {
                                "code": candidate_code,
                                "name": rows_by_code[candidate_code]["name"],
                                "chinese_name": rows_by_code[candidate_code]["chinese_name"],
                                "task_domain": rows_by_code[candidate_code]["task_domain"],
                            }
                            for candidate_code in candidate_codes
                        ],
                        "final_label": code,
                        "final_label_zh": row["chinese_name"],
                        "wiki_available": row["wiki_available"],
                        "wiki_evidence": row["wiki_evidence"],
                    }
                )
    return samples, warnings


def _load_pairs_file(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    pairs_by_code: Dict[str, List[Dict[str, Any]]] = {}
    all_pairs: List[Dict[str, Any]] = []
    if not path.exists():
        return pairs_by_code, all_pairs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            code = row.get("anchor") or row.get("label")
            if not code:
                continue
            pairs_by_code[code] = row.get("hard_negatives", [])
            all_pairs.append(row)
    return pairs_by_code, all_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AgriNet N04 disease and N05 pest data for VLooM contrast-CoT.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/AgriNet-1K"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/vit-base-cluster/vit-base-1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vlm_data/disease_pest"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--reuse-pairs", type=Path, default=None, help="Reuse existing finegrained pairs JSONL (skip PyTorch computation)")
    parser.add_argument("--random-sample-classes", type=int, default=None, help="Randomly sample N classes from all disease+pest classes (for test batches)")
    args = parser.parse_args()

    wiki_by_code = load_wiki(args.data_root / "wiki.json")
    rows_by_domain = scan_domain_classes(args.data_root / "all", wiki_by_code)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.random_sample_classes:
        all_rows = rows_by_domain["disease"] + rows_by_domain["pest"]
        if len(all_rows) > args.random_sample_classes:
            random.seed(42)
            sampled = random.sample(all_rows, args.random_sample_classes)
            rows_by_domain = {"disease": [r for r in sampled if r["task_domain"] == "disease"], "pest": [r for r in sampled if r["task_domain"] == "pest"]}
            print(f"Randomly sampled {args.random_sample_classes} classes: {len(rows_by_domain['disease'])} disease, {len(rows_by_domain['pest'])} pest")

    write_jsonl(args.output_dir / "classes_all_disease.jsonl", rows_by_domain["disease"])
    write_jsonl(args.output_dir / "classes_all_pest.jsonl", rows_by_domain["pest"])
    write_jsonl(
        args.output_dir / "classes_all_disease_pest.jsonl",
        rows_by_domain["disease"] + rows_by_domain["pest"],
    )
    write_coverage_report(args.output_dir / "wiki_coverage_report.md", rows_by_domain)

    all_pairs: List[Dict[str, Any]] = []
    pairs_by_code: Dict[str, List[Dict[str, Any]]] = {}
    if args.reuse_pairs and args.reuse_pairs.exists():
        pairs_by_code, all_pairs = _load_pairs_file(args.reuse_pairs)
        print(f"Reused pairs from {args.reuse_pairs}: {len(all_pairs)} entries")
    else:
        feature_paths = sorted(args.feature_root.glob("features_rank*.pt"))
        for domain in ["disease", "pest"]:
            codes = [row["code"] for row in rows_by_domain[domain]]
            domain_pairs = compute_pairs(feature_paths, codes, args.top_k)
            pairs_by_code.update(domain_pairs)
            for code, negatives in domain_pairs.items():
                all_pairs.append(
                    {
                        "anchor": code,
                        "label": code,
                        "task_domain": domain,
                        "similar_classes": [neg["code"] for neg in negatives],
                        "hard_negatives": negatives,
                        "degraded": len(negatives) < 3,
                        "degrade_reason": None if len(negatives) >= 3 else f"only {len(negatives)} usable negatives",
                    }
                )
    # If random sampling was applied, filter pairs to sampled classes only
    if args.random_sample_classes:
        sampled_codes = {row["code"] for rows in rows_by_domain.values() for row in rows}
        pairs_by_code = {
            code: [neg for neg in negs if neg["code"] in sampled_codes]
            for code, negs in pairs_by_code.items()
            if code in sampled_codes
        }
        all_pairs = [
            {
                **row,
                "similar_classes": [c for c in row.get("similar_classes", []) if c in sampled_codes],
                "hard_negatives": [neg for neg in row.get("hard_negatives", []) if neg.get("code") in sampled_codes],
            }
            for row in all_pairs
            if row.get("anchor") in sampled_codes
        ]
        print(f"Filtered pairs to {len(all_pairs)} sampled classes")

    write_jsonl(args.output_dir / "finegrained_pairs_vit_base.jsonl", all_pairs)

    representatives = load_representatives(args.feature_root / "representatives.json")
    samples, warnings = build_samples(
        rows_by_domain=rows_by_domain,
        pairs_by_code=pairs_by_code,
        representatives=representatives,
        all_root=args.data_root / "all",
        samples_per_class=args.samples_per_class,
    )
    write_jsonl(args.output_dir / "contrast_samples_vit_base.jsonl", samples)
    (args.output_dir / "sample_warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")

    print(f"disease_classes={len(rows_by_domain['disease'])}")
    print(f"pest_classes={len(rows_by_domain['pest'])}")
    print(f"samples={len(samples)}")
    print(f"warnings={len(warnings)}")


if __name__ == "__main__":
    main()
