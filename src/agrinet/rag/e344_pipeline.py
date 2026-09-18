"""Registered fresh full-tool preparation; sampling never reuses old trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from agrinet.rag.e344_prepare import rows, digest, audit, cluster, select, pilot
from agrinet.rag.e344_assets import verify_binding, eligible_binding, extract_features


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_rows(path, values):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(config):
    output = Path(config["output_root"])
    readiness = audit(config)
    if readiness["classes"] != 107 or readiness["exact_sha_conflicts"] or not readiness["checkpoint_exists"]:
        raise ValueError("candidate_readiness_failed")
    source_hash = digest(config["candidate_pool"])
    binding_path = output / "preparation-inputs.json"
    input_hashes = {"config": config, "candidate_sha256": source_hash,
                    "evaluation_sha256": {p: digest(p) for p in config["evaluation_manifests"]}}
    if binding_path.exists() and json.loads(binding_path.read_text()) != input_hashes:
        raise ValueError("preparation_inputs_changed")
    write(binding_path, input_hashes)
    bindings, heldout_metadata = {}, {}
    for arm in ("known", "simulated_unknown"):
        for fold in range(3):
            root = Path(config["known_root"] if arm == "known" else config["unknown_root"])
            root = root / (f"fold-artifacts/fold-{fold}" if arm == "known" else f"fold-{fold}")
            heldout = Path(config["known_heldout_root"]) / f"fold-{fold}/heldout.jsonl" if arm == "known" else None
            if heldout:
                heldout_metadata.update({r["image_sha256"]: r for r in rows(heldout)})
            key = f"{arm}:{fold}"
            bindings[key] = verify_binding(root / "classifier/model_best.pth.tar", root / "classifier/label_map.json",
                                            root / "manifests/train.jsonl", heldout)
    persisted_bindings = {key: {k: v for k, v in value.items() if not isinstance(v, set)} for key, value in bindings.items()}
    existing = output / "classifier-bindings.json"
    if existing.exists() and json.loads(existing.read_text()) != persisted_bindings:
        raise ValueError("classifier_bindings_changed")
    write(existing, persisted_bindings)
    excluded_shas = {r["training_sha256"] for r in rows(config["near_duplicate_audit"])}
    evaluations = [r for path in config["evaluation_manifests"] for r in rows(path)]
    metadata = {r["image_sha256"]: r for r in rows(config["candidate_pool"])}
    eval_sources = {metadata.get(r["image_sha256"], {}).get("source_path") for r in evaluations} - {None}
    eval_phashes = {metadata.get(r["image_sha256"], {}).get("phash") for r in evaluations} - {None}
    candidates, exclusions = [], []
    for row in metadata.values():
        if not (row.get("sft_eligible") and row.get("class_role") == "known" and row.get("image_split") == "train_candidate"):
            continue
        sha = row["image_sha256"]
        reason = None
        if sha in excluded_shas or row.get("source_path") in eval_sources or row.get("phash") in eval_phashes:
            reason = "evaluation_identity_or_recorded_near_duplicate"
        if not row.get("phash") or not row.get("source_path"):
            reason = "unresolved_candidate_identity"
        eligible = {}
        for arm in ("known", "simulated_unknown"):
            choices = [key for key, binding in bindings.items() if key.startswith(arm + ":") and eligible_binding(row, binding, arm)]
            if choices:
                eligible[arm] = choices[0]
        if not eligible:
            reason = "no_verified_classifier_binding"
        if reason:
            exclusions.append({"image_sha256": sha, "reason": reason})
            continue
        group = heldout_metadata.get(sha, {})
        candidates.append({**row, "bindings": eligible,
                           "source_group_id": group.get("source_group_id") or "source:" + hashlib.sha256(row["source_path"].encode()).hexdigest(),
                           "near_duplicate_group_id": group.get("near_duplicate_group_id") or "phash:" + hashlib.sha256(row["phash"].encode()).hexdigest()})
    candidates.sort(key=lambda r: (r["canonical_class_code"], r["image_sha256"]))
    write_rows(output / "exclusions.jsonl", exclusions)
    write(output / "preparation-status.json", {"stage": "features", "candidates": len(candidates),
                                               "isolation_limitations": readiness["unresolved_evaluation_isolation"],
                                               "near_duplicate_method": "existing exact-phash groups plus recorded cross-split near-duplicate exclusions"})
    features = extract_features(candidates, config["feature_checkpoint"], output / "features",
                                device=config.get("device", "cuda:0"), batch_size=config.get("batch_size", 64))
    indices = defaultdict(list)
    for i, row in enumerate(candidates):
        indices[row["canonical_class_code"]].append(i)
    for code, positions in indices.items():
        labels, distances = cluster(features[positions], [candidates[i]["near_duplicate_group_id"] for i in positions])
        for i, label, distance in zip(positions, labels, distances):
            candidates[i].update(cluster=label, center_distance=distance)
    write_rows(output / "cluster-inventory.jsonl", candidates)
    selected, deficits = select(candidates)
    write_rows(output / "selected.jsonl", selected)
    write_rows(output / "pilot-selection.jsonl", pilot(selected))
    write(output / "deficits.json", deficits)
    write(output / "preparation-status.json", {"stage": "selected_pending_options_and_live_binding_validation", "selected": len(selected),
                                               "pilot": 32, "qualified": 0, "ready_for_collection": False})


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text())["parameters"])


if __name__ == "__main__":
    main()
