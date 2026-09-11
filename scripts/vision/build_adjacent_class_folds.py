#!/usr/bin/env python3
"""Build classifier-only class folds with frozen RAG-neighbour bridges.

The builder first attempts three folds and falls back to five only when a
balanced three-fold assignment cannot satisfy the bridge contract.  Retrieval
coverage is never filtered by these classifier folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v3"
DEFAULT_RAG_RANKING = REPO_ROOT / "outputs/milvus/wiki_similar_classes_siglip2_report.top12.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/artifacts/vision/openagri-v3-known-vitl-e3-adjacent-classfold-v1"
SEED = "micu-classifier-hcv-e3-adjacent-classfold-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256((seed + ":" + ":".join(values)).encode()).hexdigest()


def load_known_classes(dataset_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset_root / "manifests/class_split.jsonl")
    known = {str(row["canonical_class_code"]): row for row in rows if row.get("class_role") == "known"}
    if len(known) != 107:
        raise ValueError(f"expected 107 Known classes, found {len(known)}")
    if Counter(str(row.get("domain")) for row in known.values()) != {"disease": 72, "pest": 35}:
        raise ValueError("expected 72 disease and 35 pest Known classes")
    return known


def build_known_rag_graph(
    known: dict[str, dict[str, Any]], ranking_path: Path, *, neighbours_per_class: int = 3,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Project the frozen RAG ranking onto canonical Known classes.

    This follows the existing Known/Unknown bridge semantics: directed,
    same-domain ranked neighbours.  We scan the frozen Top-12 list and retain
    the first distinct canonical Known neighbours, without symmetrising edges.
    """
    payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking = {str(row.get("code") or ""): row for row in payload.get("classes") or []}
    source_to_canonical: dict[str, str] = {}
    for code, row in known.items():
        for source in row.get("source_codes") or [code]:
            source = str(source)
            if source in source_to_canonical and source_to_canonical[source] != code:
                raise ValueError(f"source code maps to multiple canonical classes: {source}")
            source_to_canonical[source] = code

    graph: dict[str, list[str]] = {}
    for code, class_row in sorted(known.items()):
        candidates: list[str] = []
        for source in class_row.get("source_codes") or [code]:
            for item in (ranking.get(str(source)) or {}).get("similar") or []:
                neighbour = source_to_canonical.get(str(item.get("code") or ""))
                if (neighbour in known and neighbour != code
                        and known[neighbour].get("domain") == class_row.get("domain")
                        and neighbour not in candidates):
                    candidates.append(neighbour)
                if len(candidates) >= neighbours_per_class:
                    break
            if len(candidates) >= neighbours_per_class:
                break
        graph[code] = candidates

    empty = sorted(code for code, neighbours in graph.items() if not neighbours)
    if empty:
        raise ValueError(f"Known classes without a frozen RAG neighbour: {empty}")
    metadata = {
        "source_path": str(ranking_path),
        "source_sha256": sha256(ranking_path),
        "source_model_path": payload.get("model_path"),
        "source_top_k": payload.get("top_k"),
        "source_grouping_policy": payload.get("grouping_policy"),
        "projection": "first three directed same-domain canonical Known neighbours in frozen RAG order",
        "neighbours_per_class": neighbours_per_class,
        "symmetrized": False,
    }
    return graph, metadata


def balanced_capacities(total: int, folds: int) -> list[int]:
    base, remainder = divmod(total, folds)
    return [base + (index < remainder) for index in range(folds)]


def assignment_violations(assignment: dict[str, int], graph: dict[str, list[str]]) -> list[str]:
    return sorted(
        code for code, neighbours in graph.items()
        if not any(assignment[neighbour] != assignment[code] for neighbour in neighbours)
    )


def assignment_objective(
    assignment: dict[str, int], known: dict[str, dict[str, Any]], graph: dict[str, list[str]], folds: int,
) -> tuple[int, int, tuple[str, ...]]:
    violations = assignment_violations(assignment, graph)
    image_totals = [sum(int(known[code].get("train_candidate_images") or 0)
                        for code, fold in assignment.items() if fold == index) for index in range(folds)]
    return (len(violations), max(image_totals) - min(image_totals), tuple(violations))


def candidate_assignment(
    known: dict[str, dict[str, Any]], *, folds: int, seed: str, attempt: int,
) -> dict[str, int]:
    """Create an exactly domain-balanced deterministic candidate."""
    assignment: dict[str, int] = {}
    for domain in ("disease", "pest"):
        codes = [code for code, row in known.items() if row.get("domain") == domain]
        rng = random.Random(int(stable_key(seed, str(folds), str(attempt), domain)[:16], 16))
        rng.shuffle(codes)
        capacities = balanced_capacities(len(codes), folds)
        offset = 0
        for fold, capacity in enumerate(capacities):
            for code in codes[offset:offset + capacity]:
                assignment[code] = fold
            offset += capacity
    return assignment


def solve_assignment(
    known: dict[str, dict[str, Any]], graph: dict[str, list[str]], *, folds: int, seed: str, attempts: int = 20_000,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    best: dict[str, int] | None = None
    best_objective: tuple[int, int, tuple[str, ...]] | None = None
    valid = 0
    for attempt in range(attempts):
        assignment = candidate_assignment(known, folds=folds, seed=seed, attempt=attempt)
        objective = assignment_objective(assignment, known, graph, folds)
        if best_objective is None or objective < best_objective:
            best, best_objective = assignment, objective
        if objective[0] == 0:
            valid += 1
            # Keep searching a bounded deterministic set for image-count balance.
    report = {
        "folds": folds, "attempts": attempts, "valid_candidates": valid,
        "best_bridge_violations": best_objective[0] if best_objective else None,
        "best_train_image_spread": best_objective[1] if best_objective else None,
    }
    return (best if best_objective and best_objective[0] == 0 else None), report


def choose_fold_count(
    known: dict[str, dict[str, Any]], graph: dict[str, list[str]], *, seed: str, attempts: int = 20_000,
) -> tuple[int, dict[str, int], list[dict[str, Any]]]:
    reports = []
    for folds in (3, 5):
        assignment, report = solve_assignment(known, graph, folds=folds, seed=seed, attempts=attempts)
        reports.append(report)
        if assignment is not None:
            return folds, assignment, reports
    raise ValueError("neither three nor five balanced folds satisfy every frozen RAG bridge")


def write_fold_artifacts(
    dataset_root: Path, output_root: Path, known: dict[str, dict[str, Any]], graph: dict[str, list[str]],
    assignment: dict[str, int], folds: int, graph_metadata: dict[str, Any], solver_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    # Import lazily so graph-only unit tests do not initialize torch/timm.
    from agrinet.vision.workflow import build_manifests

    output_root.mkdir(parents=True, exist_ok=True)
    images = read_jsonl(dataset_root / "manifests/images.jsonl")
    class_rows = []
    for code in sorted(known):
        held_out = assignment[code]
        neighbours = graph[code]
        witnesses = [item for item in neighbours if assignment[item] != held_out]
        class_rows.append({
            "canonical_class_code": code,
            "domain": known[code]["domain"],
            "held_out_fold": held_out,
            "held_out_train_candidate_images": int(known[code].get("train_candidate_images") or 0),
            "frozen_rag_neighbours": neighbours,
            "training_neighbour_witnesses": witnesses,
            "reciprocal_witnesses": [item for item in witnesses if code in graph.get(item, [])],
        })
        if not witnesses:
            raise ValueError(f"class has no training-side RAG bridge: {code}")

    for fold in range(folds):
        held_out = {code for code, value in assignment.items() if value == fold}
        fold_root = output_root / f"fold-{fold}"
        build_manifests(dataset_root, fold_root, held_out, include_mae=False)
        label_map = json.loads((fold_root / "label_map.json").read_text(encoding="utf-8"))
        labels = {row["canonical_class_code"] for row in label_map}
        if labels & held_out or len(labels) != len(known) - len(held_out):
            raise ValueError(f"classifier label leakage or count error in fold {fold}")
        holdout_rows = [{
            "image_id": row["image_name"],
            "image_path": str(Path(row["image_path"]).resolve()),
            "image_sha256": row["image_sha256"],
            "canonical_class_code": row["canonical_class_code"],
            "domain": row["domain"],
            "source_group_id": row.get("source_group_id"),
            "near_duplicate_group_id": row.get("near_duplicate_group_id"),
        } for row in images if row.get("image_split") == "train_candidate"
            and row.get("canonical_class_code") in held_out]
        holdout_rows.sort(key=lambda row: (row["canonical_class_code"], row["image_sha256"]))
        (fold_root / "manifests/holdout_train_candidate.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in holdout_rows),
            encoding="utf-8",
        )
        if {row["canonical_class_code"] for row in holdout_rows} != held_out:
            raise ValueError(f"holdout scoring manifest lacks classes in fold {fold}")
        (fold_root / "classifier_contract.json").write_text(json.dumps({
            "training_classes": sorted(labels),
            "held_out_classes": sorted(held_out),
            "holdout_train_candidate_images": len(holdout_rows),
            "holdout_scoring_manifest": str(fold_root / "manifests/holdout_train_candidate.jsonl"),
            "rag_registry_visibility": "complete",
            "rag_does_not_exclude_held_out_classes": True,
            "training_eligible": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "agrinet.micu-classifier-hcv-e3-adjacent-classfold/v1",
        "dataset_version": "open_agri_v3",
        "known_classes": 107,
        "selected_folds": folds,
        "fold_sizes": [sum(value == fold for value in assignment.values()) for fold in range(folds)],
        "domain_counts": [{domain: sum(value == fold and known[code]["domain"] == domain for code, value in assignment.items()) for domain in ("disease", "pest")} for fold in range(folds)],
        "assignment_seed": SEED,
        "graph": graph_metadata,
        "solver_reports": solver_reports,
        "bridge_contract": {"directed": True, "same_domain": True, "minimum_training_neighbours": 1, "all_held_out_classes_have_training_neighbour": True},
        "classifier": {"held_out_classes_excluded_from_supervision": True, "full_rag_registry_retained": True},
        "training_eligible": False,
    }
    (output_root / "class_assignments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in class_rows), encoding="utf-8"
    )
    (output_root / "folds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rag-ranking", type=Path, default=DEFAULT_RAG_RANKING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=int, default=20_000)
    args = parser.parse_args()
    known = load_known_classes(args.dataset_root)
    graph, graph_metadata = build_known_rag_graph(known, args.rag_ranking)
    folds, assignment, reports = choose_fold_count(known, graph, seed=SEED, attempts=args.attempts)
    summary = write_fold_artifacts(args.dataset_root, args.output_root, known, graph, assignment, folds, graph_metadata, reports)
    print(json.dumps({key: summary[key] for key in ("selected_folds", "fold_sizes", "domain_counts", "training_eligible")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
