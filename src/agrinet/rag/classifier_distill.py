"""Offline readiness and public input boundary for classifier-assisted HCV.

This module never calls a teacher and never grants training eligibility.
Source rows and exclusion ledgers are private local artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from agrinet.research.hcv.v13_plan import CELLS, cell_key
from agrinet.research.open_agri_v2_canonical.registry import load_registry

REGISTRY_SHA = "4fa6426203c64631315a2d754f3d53d6c3a7639e26c1c8617b2cbf92caf00cb8"
GROUP_KEYS = ("image_group_id", "source_group_id", "near_duplicate_group_id")
EXCLUDED_STATES = {"rejected", "retired", "contacted", "unknown_delivery"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{number}: expected object")
                yield row


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(contract: dict) -> None:
    if not isinstance(contract, dict):
        raise ValueError("contract must be a mapping")
    required = {
        "schema_version": "agrinet.hcv-classifier-distill/v1",
        "teacher.service": "micu_slb", "teacher.model": "gpt-5.6-terra",
        "dataset.version": "open_agri_v3",
        "dataset.known_classes": 107, "dataset.unknown_classes": 104,
        "dataset.registry_sha256": REGISTRY_SHA,
        "dataset.prediction_kinds": ["fresh", "out_of_fold"],
        "dataset.oof_folds": 3, "dataset.grouping": list(GROUP_KEYS),
        "dataset.formal_unknown_sft": False, "dataset.final_test_selection": False,
        "candidates.initial_top_k": 3, "candidates.stored_top_k": 5,
        "candidates.expansion_requires_event": True,
        "candidates.retrieval_outside_top5": True,
        "candidates.scores_are_known_probability": False,
        "candidates.add_retrieval_and_classifier_scores": False,
        "views": ["without_candidates", "with_candidates"],
        "stages.smoke.independent_images": 32, "stages.smoke.per_cell": 4,
        "stages.exploration.independent_images": 160,
        "stages.exploration.target_per_pattern": 16,
        "gates.pilot_training_eligible": False,
        "gates.stable_rounds_required": 2, "gates.human_confirmation_required": True,
        "gates.independent_public_private_requests": True,
        "gates.private_truth_teacher_repair": False,
        "gates.private_outcomes": ["accept", "reject", "human_review"],
        "gates.ambiguous_delivery_auto_replay": False,
        "gates.actual_token_mask_check_required": True,
        "sampling.duplicate_padding": False, "sampling.target_pattern_is_private": True,
        "sampling.interventions_separate": True,
        "sampling.ordinary_to_hard": [1, 1],
        "sampling.compare_equal_size_random": True,
        "evaluation.confidence_interval_unit": "image_group_id",
    }
    for dotted, expected in required.items():
        actual: Any = contract
        for key in dotted.split("."):
            actual = actual.get(key) if isinstance(actual, dict) else None
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"contract mismatch: {dotted}")
    for key in ("service_version", "model_version", "prompt_version", "parameters_version"):
        if not contract["teacher"].get(key):
            raise ValueError(f"missing teacher version: {key}")
    if set(contract.get("patterns", {})) != {f"P{i}" for i in range(1, 11)}:
        raise ValueError("all ten patterns must remain represented")
    if set(contract.get("routes", {})) != set("ABCD"):
        raise ValueError("all four routes must remain represented")
    if set(contract["gates"].get("exclude", [])) != EXCLUDED_STATES:
        raise ValueError("exclusion policy mismatch")
    for key, maximum in (("rag_calls_per_trajectory", 5),
                         ("micu_requests_per_trajectory", 7),
                         ("private_audits_per_trajectory", 1)):
        value = contract.get("budgets", {}).get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"invalid budget: {key}")


def budget_summary(contract: dict, independent_images: int) -> dict:
    trajectories = independent_images * len(contract["views"])
    limits = contract["budgets"]
    generation = trajectories * limits["micu_requests_per_trajectory"]
    audits = trajectories * limits["private_audits_per_trajectory"]
    return {
        "independent_images": independent_images, "trajectories": trajectories,
        "rag_calls_max": trajectories * limits["rag_calls_per_trajectory"],
        "generation_requests_max": generation, "private_audits_max": audits,
        "total_micu_requests_max": generation + audits,
    }


def candidate_card(row: dict, registry, *, expanded: bool = False,
                   expansion_event: dict | None = None, scores: bool = True) -> list[dict]:
    """Render canonical names only; classifier lineage stays outside the prompt."""
    if expanded and (not expansion_event or
                     expansion_event.get("type") != "expand_candidates" or
                     not expansion_event.get("reason") or
                     expansion_event.get("view") != "with_candidates"):
        raise ValueError("Top-5 visibility requires a candidate-view expansion event")
    return [
        {"rank": rank, "name": registry.display_name(item["code"], row["language"]),
         **({"score": item["score"]} if scores else {})}
        for rank, item in enumerate(row["prediction"]["top5"][:5 if expanded else 3], 1)
    ]


def teacher_views(row: dict, registry, *, scores: bool = True) -> dict:
    """Build independent request payloads by allowlist, never by deleting truth.

    The transport must encode image bytes from image_path; the path itself is
    not teacher text (source paths can contain labels). No generation history
    or precomputed retrieval is copied between these requests.
    """
    base = {
        "image_path": row["image_path"],
        "messages": [{"role": "user", "content": row["question"]}],
    }
    blind, assisted = deepcopy(base), deepcopy(base)
    assisted["candidate_card"] = candidate_card(row, registry, scores=scores)
    return {"without_candidates": blind, "with_candidates": assisted}


def validate_source_row(row: dict, registry, roles: dict, contract: dict) -> None:
    if not isinstance(row, dict):
        raise ValueError("source row must be an object")
    for key in (*GROUP_KEYS, "image_sha256", "image_path", "question", "sample_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing source field: {key}")
    if cell_key(row) not in CELLS:
        raise ValueError("invalid task/language/domain cell")
    if row.get("dataset_version") != contract["dataset"]["version"]:
        raise ValueError("source dataset version mismatch")
    if row.get("split") not in {"train", "train_candidate"}:
        raise ValueError("pilot source must exclude dev and final test")
    if file_sha(Path(row["image_path"])) != row["image_sha256"]:
        raise ValueError("image SHA mismatch")
    private = row.get("private", {})
    if not isinstance(private, dict):
        raise ValueError("private sidecar must be an object")
    truth = private.get("truth_code")
    if roles.get(truth) != "known" or private.get("class_role") != "known":
        raise ValueError("formal Unknown is excluded from collection for SFT")
    if registry.rows_by_code[truth]["domain"] != row["task_domain"]:
        raise ValueError("truth domain mismatch")
    pattern = private.get("target_pattern")
    if pattern not in contract["patterns"]:
        raise ValueError("missing target pattern")
    if pattern == "P10" and row["question_type"] != "open":
        raise ValueError("P10 requires Open")
    if private.get("sampling_bucket") not in {"ordinary", "hard"}:
        raise ValueError("invalid sampling bucket")
    if private.get("status") != "fresh" or private.get("intervention") is not False:
        raise ValueError("contacted or intervention source is not a natural fresh sample")
    prediction = row.get("prediction", {})
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be an object")
    if prediction.get("kind") not in contract["dataset"]["prediction_kinds"]:
        raise ValueError("in-sample predictions are rehearsal-only")
    for key in ("classifier_version", "checkpoint_sha256", "training_manifest",
                "training_manifest_sha256"):
        if not prediction.get(key):
            raise ValueError(f"missing prediction provenance: {key}")
    if prediction.get("registry_sha256") != registry.digest:
        raise ValueError("prediction registry mismatch")
    if prediction.get("image_sha256") != row["image_sha256"]:
        raise ValueError("prediction image mismatch")
    if prediction["kind"] == "out_of_fold":
        if prediction.get("folds") != 3 or type(prediction.get("held_out_fold")) is not int or prediction["held_out_fold"] not in range(3):
            raise ValueError("OOF requires one of three held-out folds")
    if private.get("simulated_unknown") is not False:
        if private.get("simulated_unknown") is not True:
            raise ValueError("simulated_unknown must be explicit")
        if truth not in prediction.get("excluded_supervised_codes", []):
            raise ValueError("simulated Unknown class was not held out")
        if private.get("mae_saw_related_unlabeled") not in (True, False, "unknown"):
            raise ValueError("missing MAE exposure record")
    if pattern == "P6" and private.get("simulated_unknown") is not True:
        raise ValueError("P6 requires simulated Unknown")
    top5 = prediction.get("top5", [])
    if not isinstance(top5, list) or any(not isinstance(item, dict) for item in top5):
        raise ValueError("Top-5 must be a list of candidate objects")
    if len(top5) != 5 or len({item["code"] for item in top5}) != 5:
        raise ValueError("five distinct classifier candidates required")
    previous = 1.0
    for item in top5:
        score = item.get("score")
        if roles.get(item["code"]) != "known" or item["code"] in prediction.get("excluded_supervised_codes", []):
            raise ValueError("candidate is outside classifier supervised classes")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= previous:
            raise ValueError("candidate scores must be finite probabilities in rank order")
        previous = score
    if sum(item["score"] for item in top5) > 1.000001:
        raise ValueError("Top-5 softmax mass exceeds one")


def identity_sets(rows) -> dict[str, set[str]]:
    result = {key: set() for key in (*GROUP_KEYS, "image_sha256")}
    for row in rows:
        for key in result:
            value = row.get(key)
            # OOF classifier manifests predate the public-source schema and
            # carry only the byte identity.  Its image group is intentionally
            # the one-image SHA-256 group, never a weaker synthetic grouping.
            if key == "image_group_id" and not value:
                digest = row.get("image_sha256")
                value = f"image:{digest}" if isinstance(digest, str) and digest else None
            if not isinstance(value, str) or not value:
                raise ValueError(f"isolation manifest missing {key}")
            result[key].add(value)
    return result


def check_training_isolation(row: dict, cache: dict) -> None:
    prediction = row["prediction"]
    path = Path(prediction["training_manifest"])
    expected = prediction["training_manifest_sha256"]
    key = (str(path.resolve()), expected)
    if key not in cache:
        if file_sha(path) != expected:
            raise ValueError("classifier training manifest SHA mismatch")
        training = list(read_jsonl(path))
        if not training:
            raise ValueError("empty classifier training manifest")
        identities = identity_sets(training)
        codes = {item["canonical_class_code"] for item in training}
        cache[key] = identities, codes
    identities, codes = cache[key]
    if any(row[field] in values for field, values in identities.items()):
        raise ValueError("classifier training overlap by image/source/near-duplicate group")
    if codes & set(prediction.get("excluded_supervised_codes", [])):
        raise ValueError("held-out class appears in classifier training manifest")


def inspect_sources(rows: list[dict], *, registry, roles: dict, contract: dict,
                    exclusions: dict, evaluation_hashes: set[str], stage: str) -> dict:
    if stage not in {"smoke", "exploration"}:
        raise ValueError("unsupported stage")
    if not isinstance(exclusions, dict):
        raise ValueError("exclusions must be an object")
    if exclusions.get("schema_version") != "agrinet.hcv-classifier-exclusions/v1":
        raise ValueError("exclusion ledger schema mismatch")
    if exclusions.get("complete") is not True or not exclusions.get("provenance"):
        raise ValueError("exclusion ledger requires complete historical provenance")
    records = exclusions.get("records")
    if not isinstance(records, list):
        raise ValueError("exclusion records must be a list")
    if any(not isinstance(item, dict) or item.get("status") not in EXCLUDED_STATES for item in records):
        raise ValueError("unrecognized exclusion status")
    excluded = identity_sets(records)
    seen = {key: set() for key in (*GROUP_KEYS, "image_sha256", "sample_id")}
    counts: Counter = Counter()
    patterns: Counter = Counter()
    buckets: Counter = Counter()
    failures, selected = [], []
    cache: dict = {}
    for index, row in enumerate(rows):
        try:
            validate_source_row(row, registry, roles, contract)
            if row["image_sha256"] in evaluation_hashes:
                raise ValueError("image overlaps frozen dev/test")
            if any(row[key] in values for key, values in excluded.items()):
                raise ValueError("image/source group overlaps historical exclusion")
            check_training_isolation(row, cache)
            if any(row[key] in values for key, values in seen.items()):
                raise ValueError("duplicate image or source group in pilot")
            for key in seen:
                seen[key].add(row[key])
            cell = cell_key(row)
            if stage == "smoke" and counts[cell] >= 4:
                continue
            pattern = row["private"]["target_pattern"]
            if stage == "exploration" and patterns[pattern] >= 16:
                continue
            selected.append(row["sample_id"])
            counts[cell] += 1
            patterns[pattern] += 1
            buckets[row["private"]["sampling_bucket"]] += 1
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append({"row_index": index, "reason": str(exc)[:240]})
    shortages = ({cell: 4 - counts[cell] for cell in CELLS if counts[cell] < 4}
                 if stage == "smoke" else
                 {p: 16 - patterns[p] for p in contract["patterns"] if patterns[p] < 16})
    return {"selected_sample_ids": selected, "valid_independent_images": len(selected),
            "per_cell": dict(counts), "target_patterns": dict(patterns),
            "sampling_buckets": dict(buckets), "shortages": shortages,
            "failures": failures, "source_ready": not shortages and not failures,
            "selected_budget": budget_summary(contract, len(selected))}


def preflight(*, contract_path: Path, dataset_root: Path, source: Path,
              exclusions: Path, stage: str) -> dict:
    report: dict = {
        "schema_version": "agrinet.hcv-classifier-preflight/v1",
        "stage": stage, "ready": False, "training_eligible": False,
        "live_collection_implemented": False, "blockers": [], "inputs": {},
    }
    blockers = report["blockers"]
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        validate_contract(contract)
        if stage not in {"smoke", "exploration"}:
            raise ValueError("unsupported stage")
        report["inputs"]["contract_sha256"] = file_sha(contract_path)
        report["planned_budget"] = budget_summary(contract, contract["stages"][stage]["independent_images"])
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        blockers.append(f"contract: {exc}")
        return report
    registry = None
    roles: dict = {}
    evaluation_hashes: set[str] = set()
    try:
        registry = load_registry(
            dataset_root / "taxonomy/canonical_label_registry.jsonl",
            dataset_root / "taxonomy/approval.json", require_approval=True)
        if registry.digest != REGISTRY_SHA:
            raise ValueError("frozen registry digest mismatch")
        classes = list(read_jsonl(dataset_root / "manifests/class_split.jsonl"))
        roles = {row["canonical_class_code"]: row["class_role"] for row in classes}
        if len(classes) != len(roles) or Counter(roles.values()) != {"known": 107, "unknown": 104}:
            raise ValueError("expected 107 Known and 104 Unknown classes")
        if set(roles) != set(registry.rows_by_code):
            raise ValueError("class split and registry codes differ")
        # Only identity/split metadata is used; no final-test labels or scores.
        evaluation_hashes = {row["image_sha256"] for row in read_jsonl(
            dataset_root / "manifests/images.jsonl") if row["image_split"] in {"dev", "test"}}
        if not evaluation_hashes:
            raise ValueError("missing frozen dev/test identities")
        report["dataset"] = {"registry_sha256": registry.digest,
                             "class_counts": dict(Counter(roles.values())),
                             "evaluation_hashes_excluded": len(evaluation_hashes)}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        blockers.append(f"dataset: {exc}")
    for name, path in (("source", source), ("exclusions", exclusions)):
        if not path.is_file():
            blockers.append(f"missing {name}: {path}")
        else:
            report["inputs"][name] = {"path": str(path), "sha256": file_sha(path)}
    if not blockers:
        try:
            report["source"] = inspect_sources(
                list(read_jsonl(source)), registry=registry, roles=roles, contract=contract,
                exclusions=json.loads(exclusions.read_text(encoding="utf-8")),
                evaluation_hashes=evaluation_hashes, stage=stage)
            if not report["source"]["source_ready"]:
                blockers.append("source validation failures or independent-image shortages")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(f"source: {exc}")
    report["offline_ready"] = not blockers
    # Readiness describes offline input checks, never permission to call a service.
    report["ready"] = report["offline_ready"]
    return report


def main(argv: list[str] | None = None) -> int:
    from datetime import datetime, timezone
    import uuid

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("contract", "dataset-root", "source", "exclusions", "output-root"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--stage", choices=("smoke", "exploration"), default="smoke")
    args = parser.parse_args(argv)
    report = preflight(contract_path=args.contract, dataset_root=args.dataset_root,
                       source=args.source, exclusions=args.exclusions, stage=args.stage)
    # Each invocation retains its own report, including failures. No replay or overwrite.
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = args.output_root / attempt
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "report.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(destination), "ready": report["ready"],
                      "blockers": report["blockers"]}, ensure_ascii=False))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
