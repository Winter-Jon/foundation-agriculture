"""Full-tool deterministic selection primitives and honest input readiness audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PRIORITY = (("simulated_unknown", "open", 6), ("known", "open", 6),
            ("simulated_unknown", "option", 3), ("known", "option", 3))
IDENTITIES = ("image_sha256", "source_group_id", "near_duplicate_group_id")


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cluster(features, independent_groups):
    import numpy as np
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(independent_groups):
        raise ValueError("feature_group_shape_mismatch")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if x.ndim != 2 or not np.isfinite(x).all() or (norms == 0).any():
        raise ValueError("invalid_features")
    x = x / norms
    count = min(18, len(set(independent_groups)), len(np.unique(x, axis=0)))
    if count < 1:
        raise ValueError("empty_cluster_inventory")
    rng = np.random.default_rng(42)
    centers = [x[int(rng.integers(len(x)))]]
    for _ in range(1, count):
        distances = np.min(((x[:, None] - np.asarray(centers)) ** 2).sum(axis=2), axis=1)
        centers.append(x[int(rng.choice(len(x), p=distances / distances.sum()))])
    centers = np.asarray(centers)
    for _ in range(100):
        labels = ((x[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
        updated = np.asarray([x[labels == i].mean(axis=0) if (labels == i).any() else centers[i] for i in range(count)])
        if np.allclose(centers, updated, atol=1e-6):
            centers = updated
            break
        centers = updated
    labels = ((x[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
    distances = np.linalg.norm(x - centers[labels], axis=1)
    return labels.tolist(), distances.tolist()


def select(pool, used=None, preferred_cluster=None, quota_multiplier=1):
    """Allocate unique images/groups, cycling least covered clusters per class."""
    if not isinstance(quota_multiplier, int) or quota_multiplier < 1:
        raise ValueError("invalid_quota_multiplier")
    used = {k: set((used or {}).get(k, [])) for k in IDENTITIES}
    selected, deficits = [], []
    by_class = defaultdict(list)
    for row in pool:
        if any(not row.get(k) for k in IDENTITIES):
            raise ValueError("missing_identity")
        by_class[row["canonical_class_code"]].append(row)
    for code, candidates in sorted(by_class.items()):
        coverage = Counter()
        for arm, question, quota in PRIORITY:
            quota *= quota_multiplier
            count = 0
            for _ in range(quota):
                available = [r for r in candidates if arm in r["bindings"] and
                             all(r[k] not in used[k] for k in IDENTITIES)]
                if not available:
                    break
                row = min(available, key=lambda r: (
                    0 if preferred_cluster is not None and r["cluster"] == preferred_cluster else 1,
                    coverage[r["cluster"]], r["center_distance"], r["image_sha256"]))
                selected.append({**row, "arm": arm, "question_type": question,
                                 "selection_reason": "least_covered_cluster_then_center"})
                for key in IDENTITIES:
                    used[key].add(row[key])
                coverage[row["cluster"]] += 1
                count += 1
            deficits.append({"canonical_class_code": code, "arm": arm,
                             "question_type": question, "target": quota, "selected": count,
                             "qualified": 0, "deficit": quota})
    return selected, deficits


def pilot(selected):
    result, classes = [], Counter()
    for arm in ("known", "simulated_unknown"):
        for question in ("open", "option"):
            for domain in ("disease", "pest"):
                candidates = [r for r in selected if (r["arm"], r["question_type"], r["domain"]) == (arm, question, domain)]
                for _ in range(4):
                    if not candidates:
                        raise ValueError(f"pilot_cell_shortage:{arm}:{question}:{domain}")
                    row = min(candidates, key=lambda r: (classes[r["canonical_class_code"]], r["center_distance"], r["image_sha256"]))
                    candidates.remove(row)
                    result.append(row)
                    classes[row["canonical_class_code"]] += 1
    return result


def audit(config):
    """Read all metadata without contacting teachers or claiming missing isolation."""
    pool = list(rows(config["candidate_pool"]))
    candidates = [r for r in pool if r.get("sft_eligible") is True and r.get("image_split") == "train_candidate" and r.get("class_role") == "known"]
    by_sha = {r["image_sha256"]: r for r in pool}
    heldouts = [r for path in config["evaluation_manifests"] for r in rows(path)]
    overlap = {r["image_sha256"] for r in candidates} & {r["image_sha256"] for r in heldouts}
    missing = Counter()
    for row in heldouts:
        metadata = {**by_sha.get(row.get("image_sha256"), {}), **row}
        for key in ("source_group_id", "near_duplicate_group_id"):
            if not metadata.get(key):
                missing[key] += 1
    report = {"schema_version": "agrinet.e344-readiness/v1", "candidate_rows": len(candidates),
              "classes": len({r["canonical_class_code"] for r in candidates}),
              "evaluation_rows": len(heldouts), "exact_sha_conflicts": len(overlap),
              "unresolved_evaluation_isolation": dict(missing),
              "checkpoint_exists": Path(config["feature_checkpoint"]).is_file(),
              "ready_for_collection": False, "provider_requests": 0,
              "remaining": ["resolve group isolation", "verify six classifier bindings", "extract frozen features",
                            "cluster and prepare options", "collect and audit 32 pilot images", "validate actual SFT template"]}
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["parameters"]
    print(json.dumps(audit(config)))


if __name__ == "__main__":
    main()


def replacement(pool, failed, attempted, accepted):
    """One deterministic replacement; no quota is satisfied by a failed attempt."""
    used = {key: {row[key] for row in attempted + accepted} for key in IDENTITIES}
    coverage = Counter(row['cluster'] for row in accepted if row['canonical_class_code'] == failed['canonical_class_code'])
    available = [row for row in pool if row['canonical_class_code'] == failed['canonical_class_code']
                 and failed['arm'] in row['bindings'] and all(row[key] not in used[key] for key in IDENTITIES)]
    if not available:
        return None
    row = min(available, key=lambda row: (row['cluster'] != failed['cluster'], coverage[row['cluster']],
                                          row['center_distance'], row['image_sha256']))
    return {**row, 'arm': failed['arm'], 'question_type': failed['question_type'],
            'selection_reason': 'replace_same_cluster' if row['cluster'] == failed['cluster'] else 'replace_undercovered_cluster'}
