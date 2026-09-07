"""Plan and audit a human-gated OpenAgri v2 supervision supplement.

All outputs remain outside the formal dataset root until an explicit human
approval is supplied to the merge command.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from agrinet.data.io import DataError, write_jsonl_atomic

LANGUAGES = ("en", "zh")
QUESTION_TYPES = ("open", "option")
ROUTES = ("direct", "rag")
ORACLE_RECOVERY_CODES = {"N04094", "N04113"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_order(rows: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row.get('class_code')}:{row.get('image_sha256')}".encode()
    ).hexdigest())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "datasets/AgriNet-1K/vlm_data/disease_pest/classes_all_disease_pest.jsonl"
    catalog = {str(row["code"]): row for row in read_jsonl(path)}
    if not catalog:
        raise DataError(f"missing class catalog: {path}")
    return catalog


def existing_images(dataset_root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for path in sorted((dataset_root / "vlm_data/accepted/train").glob("*.jsonl")):
        for row in read_jsonl(path):
            metadata = row.get("metadata") or {}
            code = str(metadata.get("class_code") or "")
            digest = str(metadata.get("v2_image_sha256") or metadata.get("image_sha256") or "")
            if code and digest:
                result[code].add(digest)
    return result


def public_class(catalog: dict[str, dict[str, Any]], code: str) -> dict[str, str]:
    row = catalog[code]
    evidence = row.get("wiki_evidence") or []
    knowledge = " ".join(str(item) for item in evidence[:2]).strip()[:400]
    return {
        "name": str(row["english_name"]),
        "name_zh": str(row["chinese_name"]),
        "chinese_name": str(row["chinese_name"]),
        "public_knowledge": knowledge,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(path, rows)


def build_initial_plan(
    repo_root: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    seed: str,
    oversample: float = 1.5,
    round_index: int = 0,
    retired_images: set[str] | None = None,
    credited_images: dict[str, set[str]] | None = None,
    replay_authorized_images: dict[str, set[str]] | None = None,
    report_name: str = "initial_plan.json",
    allow_capacity_limited: bool = False,
) -> dict[str, Any]:
    """Write a staging-only, 1.5x first-round plan for Known class shortfalls."""
    if oversample < 1:
        raise DataError("oversample must be at least 1")
    catalog = load_catalog(repo_root)
    gate_path = repo_root / "configs/sampling/m1-direct-acceptance-gates-v2.yaml"
    if not gate_path.is_file():
        raise DataError(f"missing Direct acceptance gates: {gate_path}")
    gate_config_sha256 = file_sha256(gate_path)
    roles = {
        str(row["class_code"]): row
        for row in read_jsonl(dataset_root / "manifests/class_split.jsonl")
    }
    known = {
        code for code, row in roles.items()
        if row.get("class_role") == "known" and row.get("sft_eligible") is True
    }
    if len(known) != 109:
        raise DataError(f"expected 109 Known classes, found {len(known)}")
    missing_catalog = known - set(catalog)
    if missing_catalog:
        raise DataError(f"catalog lacks Known classes: {sorted(missing_catalog)}")

    existing = existing_images(dataset_root)
    for code, images in (credited_images or {}).items():
        existing[code].update(images)
    retired_images = retired_images or set()
    replay_authorized_images = replay_authorized_images or {}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(dataset_root / "vlm_data/candidates/image_pool.jsonl"):
        code = str(row.get("class_code") or "")
        digest = str(row.get("image_sha256") or "")
        if (
            code in known
            and row.get("sft_eligible") is True
            and row.get("image_split") == "train_candidate"
            and digest
            and digest not in existing[code]
            and (digest not in retired_images or digest in replay_authorized_images.get(code, set()))
        ):
            candidates[code].append({
                **row,
                "authorized_replay": digest in replay_authorized_images.get(code, set()),
            })

    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    capacity_limited: dict[str, dict[str, int]] = {}
    for code in sorted(known):
        shortfall = max(0, 5 - len(existing[code]))
        if not shortfall:
            continue
        initial_count = math.ceil(shortfall * oversample)
        fresh = [row for row in candidates[code] if not row.get("authorized_replay")]
        replay = [row for row in candidates[code] if row.get("authorized_replay")]
        available = len(fresh)
        choices = [
            *stable_order(fresh, f"{seed}:round-{round_index}:fresh"),
            *stable_order(replay, f"{seed}:round-{round_index}:authorized-replay"),
        ][:initial_count]
        if len(choices) != initial_count and not allow_capacity_limited:
            raise DataError(
                f"insufficient fresh v2 candidates for {code}: {len(choices)} < {initial_count}"
            )
        if len(choices) != initial_count:
            capacity_limited[code] = {
                "image_shortfall": shortfall,
                "planned_oversample_images": initial_count,
                "fresh_available_images": available,
                "authorized_replay_available_images": len(replay),
                "selected_images": len(choices),
                "unavoidable_floor_shortfall": max(0, shortfall - len(choices)),
            }
        shortfalls[code] = shortfall
        selected.extend({**row, "round": round_index, "required_images": shortfall} for row in choices)

    direct, rag, private = [], [], []
    for image in selected:
        code = str(image["class_code"])
        digest = str(image["image_sha256"])
        negatives = [
            str(item) for item in roles[code].get("milvus_top3_similar_classes") or []
            if str(item) in catalog and str(item) != code
        ]
        if len(negatives) != 3 or len(set(negatives)) != 3:
            raise DataError(f"{code} lacks three frozen catalog hard negatives")
        option_classes = [public_class(catalog, item) for item in [code, *negatives]]
        image_ref = hashlib.sha256(f"{seed}:round-{round_index}:{digest}".encode()).hexdigest()[:20]
        for route in ROUTES:
            for language in LANGUAGES:
                for question_type in QUESTION_TYPES:
                    sample_id = f"oav2s-{image_ref}-{route}-{question_type}-{language}"
                    ordered = stable_order(option_classes, f"{seed}:{sample_id}:options")
                    option_index = next(
                        index for index, item in enumerate(ordered)
                        if item["name"] == str(catalog[code]["english_name"])
                    )
                    common = {
                        "sample_id": sample_id,
                        "image_ref": image_ref,
                        "image_sha256": digest,
                        "query_image": str(image["image_path"]),
                        "task_domain": str(image["domain"]),
                        "language": language,
                        "question_type": question_type,
                        "candidate_classes": ordered,
                        "round": round_index,
                        "collection_route": route,
                        "authorized_replay": bool(image.get("authorized_replay")),
                        "special_replay": image.get("special_replay"),
                    }
                    private.append({
                        "sample_id": sample_id,
                        "class_code": code,
                        "query_image": str(image["image_path"]),
                        "image_sha256": digest,
                        "truth_name": str(catalog[code]["english_name"]),
                        "truth_name_zh": str(catalog[code]["chinese_name"]),
                        "correct_option": "ABCD"[option_index] if question_type == "option" else None,
                        "authorized_replay": bool(image.get("authorized_replay")),
                        "special_replay": image.get("special_replay"),
                    })
                    if route == "direct":
                        direct.append({
                            **common,
                            "hard_negative_lineage": {"source": "formal_v2_frozen_top3", "count": 3},
                            "teacher_prompt_version": "open-agri-v2-supplement-direct/v1",
                            "auditor_prompt_version": "open-agri-v2-supplement-auditor/v1",
                            "gate_config_sha256": gate_config_sha256,
                        })
                    else:
                        rag.append({
                            **common,
                            "source_sample_id": sample_id,
                            "target_id": sample_id,
                            "final_label": code,
                            "canonical_class": code,
                            "final_label_name": str(catalog[code]["english_name"]),
                            "final_label_zh": str(catalog[code]["chinese_name"]),
                            "candidate_labels": [
                                {"code": item, **public_class(catalog, item)}
                                for item in [code, *negatives]
                            ],
                            "public_option_labels": ordered,
                            "generation_route": "blind_evidence",
                            "label_visible_to_teacher": False,
                            "trajectory_mode": "standard",
                            "strategy_id": "hcv_visual_expand",
                            "approval_only": False,
                            "reserve": False,
                        })

    write_rows(output_root / "public/direct_plan.jsonl", direct)
    # The blind collector needs provenance to validate its final answer, but
    # these fields are not included in the teacher-facing prompt. Keep this
    # executable plan private and expose only a redacted request manifest.
    write_rows(output_root / "private/rag_plan.jsonl", rag)
    write_rows(output_root / "private/alignment.jsonl", private)
    write_rows(
        output_root / "public/rag_request_manifest.jsonl",
        [{key: row[key] for key in (
            "sample_id", "image_ref", "image_sha256", "query_image", "task_domain",
            "language", "question_type", "round", "collection_route",
        )} for row in rag],
    )
    report = {
        "schema_version": "agrinet.open-agri-v2.supplement-plan/v1",
        "dataset_root": str(dataset_root),
        "seed": seed,
        "oversample": oversample,
        "known_classes": len(known),
        "classes_short": len(shortfalls),
        "accepted_image_shortfall": sum(shortfalls.values()),
        "planned_images": len(selected),
        "planned_views": {"direct": len(direct), "rag": len(rag)},
        "direct_gate_config_sha256": gate_config_sha256,
        "shortfalls": shortfalls,
        "capacity_limited": capacity_limited,
        "authorized_replay_selected_images": {
            code: sum(
                1 for row in selected
                if str(row["class_code"]) == code and row.get("authorized_replay")
            )
            for code in sorted(replay_authorized_images)
        },
        "formal_dataset_modified": False,
    }
    report_path = output_root / "reports" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def rag_service_preflight(
    plan_path: Path,
    report_path: Path,
    *,
    rag_api: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Verify one real visual-retrieval request before any teacher request."""
    samples = read_jsonl(plan_path)
    if not samples:
        raise DataError(f"RAG preflight needs a non-empty plan: {plan_path}")
    sample = samples[0]
    payload = json.dumps({
        "image_path": sample["query_image"],
        "text": "visible agricultural symptoms",
        "top_k": 3,
        "preset": "visual",
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{rag_api.rstrip('/')}/search/visual", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    report: dict[str, Any] = {
        "schema_version": "agrinet.open-agri-v2.supplement-rag-preflight/v1",
        "rag_api": rag_api,
        "sample_id": sample.get("sample_id"),
        "image_sha256": sample.get("image_sha256"),
        "status": "failed",
        "teacher_contacted": False,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = json.loads(response.read().decode("utf-8"))
        hits = raw.get("hybrid") or raw.get("image_vector") or raw.get("text_vector")
        if status != 200 or not isinstance(hits, list) or not hits:
            raise DataError("visual retrieval returned no usable result list")
        report.update({"status": "passed", "http_status": status, "result_count": len(hits)})
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, DataError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise DataError(f"RAG service preflight failed; no Micu request was issued: {report['error']}")
    return report


def build_replenishment_plan(
    repo_root: Path,
    dataset_root: Path,
    staging_root: Path,
    *,
    round_index: int,
    seed: str,
    oversample: float = 1.5,
    authorized_replay_codes: set[str] | None = None,
) -> dict[str, Any]:
    """Plan a fresh retry round with an explicit, narrow replay exception."""
    if round_index not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}:
        raise DataError("authorized replenishment is limited to rounds 1 through 11")
    authorized_replay_codes = authorized_replay_codes or set()
    if authorized_replay_codes - {"N04094", "N04117", "N04113"}:
        raise DataError("only the user-authorized N04094/N04117/N04113 replay scope is supported")
    roles = {
        str(row["class_code"]): row
        for row in read_jsonl(dataset_root / "manifests/class_split.jsonl")
    }
    known = {
        code for code, row in roles.items()
        if row.get("class_role") == "known" and row.get("sft_eligible") is True
    }
    if len(known) != 109:
        raise DataError(f"expected 109 Known classes, found {len(known)}")
    contacted = {
        str(row.get("image_sha256") or "")
        for row in read_jsonl(staging_root / "collection/contacted_images.jsonl")
    } - {""}
    decisions = read_jsonl(staging_root / "review/image_decisions.jsonl")
    credited: dict[str, set[str]] = defaultdict(set)
    replay_authorized_images: dict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        if decision.get("accepted") is True:
            credited[str(decision["class_code"])].add(str(decision["image_sha256"]))
        elif str(decision.get("class_code") or "") in authorized_replay_codes:
            replay_authorized_images[str(decision["class_code"])].add(str(decision["image_sha256"]))
    round_root = staging_root / "rounds" / f"v{round_index}"
    report = build_initial_plan(
        repo_root, dataset_root, round_root, seed=seed, oversample=oversample,
        round_index=round_index, retired_images=contacted, credited_images=credited,
        replay_authorized_images=replay_authorized_images, report_name="plan.json",
        allow_capacity_limited=round_index in {3, 4, 5, 6, 7, 8, 9, 10, 11},
    )
    replayed = {
        str(row["image_sha256"])
        for row in read_jsonl(round_root / "private/alignment.jsonl")
        if row.get("authorized_replay") is True
    }
    if replayed:
        for path in (round_root / "public/direct_plan.jsonl", round_root / "private/rag_plan.jsonl", round_root / "private/alignment.jsonl"):
            rows = read_jsonl(path)
            write_rows(path, [
                {
                    **row,
                    "special_replay": {
                        "authorized_by": "user", "reason": "round6_threefold_selected_class_replay",
                        "scope": "N04094_N04117_N04113_only", "replay_count": 1,
                    },
                } if str(row.get("image_sha256") or "") in replayed else row
                for row in rows
            ])
    report.update({
        "schema_version": "agrinet.open-agri-v2.supplement-replenishment-plan/v1",
        "round": round_index,
        "retired_images_excluded": len(contacted),
        "oversample_strategy": f"ceil({oversample} x image_shortfall)",
        "capacity_limited_strategy": "fresh_first_then_user_authorized_replay_only" if authorized_replay_codes else (
            "use_all_remaining_fresh_images_only" if round_index in {3, 4, 5, 6, 7, 8, 9, 10, 11} else None
        ),
        "authorized_replay_codes": sorted(authorized_replay_codes),
        "authorized_replay_images": sorted(replayed),
        "formal_dataset_modified": False,
    })
    report_path = round_root / "reports/plan.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _round_roots(staging_root: Path) -> list[Path]:
    return [staging_root, *sorted(path for path in (staging_root / "rounds").glob("v*") if path.is_dir())]


def _direct_clean_replays(staging_root: Path, code: str) -> set[str]:
    """Find images with a complete Direct pass across a prior round."""
    result: set[str] = set()
    for root in _round_roots(staging_root):
        ledger = {str(row.get("sample_id") or ""): row for row in read_jsonl(root / "collection/direct/ledger.jsonl")}
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in read_jsonl(root / "private/alignment.jsonl"):
            if row.get("class_code") == code and "-direct-" in str(row.get("sample_id") or ""):
                grouped[str(row["image_sha256"])].append(str(row["sample_id"]))
        for digest, sample_ids in grouped.items():
            if len(sample_ids) == 4 and all(ledger.get(sample_id, {}).get("accepted") is True for sample_id in sample_ids):
                result.add(digest)
    return result


def _oracle_recovery_root(staging_root: Path, recovery_id: str) -> Path:
    if not re.fullmatch(r"oracle-[a-z0-9-]+-v\d+", recovery_id):
        raise DataError(f"invalid Oracle recovery id: {recovery_id!r}")
    return staging_root / "recoveries" / recovery_id


def _oracle_contacted_images(staging_root: Path, codes: set[str]) -> set[str]:
    """Return images already assigned to any Oracle batch for these classes."""
    used: set[str] = set()
    for plan_path in (staging_root / "recoveries").glob("oracle-*/private/rag_plan.jsonl"):
        for row in read_jsonl(plan_path):
            if str(row.get("final_label") or "") in codes:
                digest = str(row.get("image_sha256") or "")
                if digest:
                    used.add(digest)
    return used


def _oracle_complete_images(staging_root: Path, codes: set[str]) -> set[str]:
    """Return only images that already passed all four Oracle-RAG views."""
    complete: set[str] = set()
    for recovery_root in (staging_root / "recoveries").glob("oracle-*"):
        plan = {
            str(row.get("sample_id") or ""): row
            for row in read_jsonl(recovery_root / "private/rag_plan.jsonl")
            if str(row.get("final_label") or "") in codes
        }
        accepted = {
            str(row.get("sample_id") or "")
            for row in read_jsonl(recovery_root / "collection/rag/accepted.jsonl")
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        for sample_id, row in plan.items():
            grouped[str(row.get("image_sha256") or "")].append(sample_id)
        for digest, sample_ids in grouped.items():
            if digest and len(sample_ids) == 4 and all(sample_id in accepted for sample_id in sample_ids):
                complete.add(digest)
    return complete


def build_oracle_recovery_plan(
    repo_root: Path, dataset_root: Path, staging_root: Path, *, seed: str, oversample: float = 3.0,
    recovery_id: str = "oracle-n04094-n04113-v1", recovery_codes: set[str] | None = None,
    allow_oracle_replay: bool = False,
) -> dict[str, Any]:
    """Plan the user-authorized Oracle-RAG evidence route for two classes.

    The plan is private and RAG-only. It deliberately never replaces normal
    blind-review decisions: successful trajectories remain separately marked
    `oracle_grounded` until a human reviews that evidence route.
    """
    if oversample < 1:
        raise DataError("oversample must be at least 1")
    codes = recovery_codes or set(ORACLE_RECOVERY_CODES)
    if not codes or codes - ORACLE_RECOVERY_CODES:
        raise DataError("Oracle recovery supports only N04094 and/or N04113")
    recovery_root = _oracle_recovery_root(staging_root, recovery_id)
    if (recovery_root / "private/rag_plan.jsonl").exists():
        raise DataError(f"Oracle recovery plan already exists: {recovery_root}")
    catalog = load_catalog(repo_root)
    roles = {str(row["class_code"]): row for row in read_jsonl(dataset_root / "manifests/class_split.jsonl")}
    if any(roles.get(code, {}).get("sft_eligible") is not True for code in codes):
        raise DataError("oracle recovery requires SFT-eligible Known classes")

    existing = existing_images(dataset_root)
    decisions = read_jsonl(staging_root / "review/image_decisions.jsonl")
    rejected: dict[str, set[str]] = defaultdict(set)
    selected: dict[str, set[str]] = defaultdict(set)
    for row in decisions:
        code = str(row.get("class_code") or "")
        digest = str(row.get("image_sha256") or "")
        if code not in codes or not digest:
            continue
        if row.get("selected_for_merge") is True:
            selected[code].add(digest)
        if row.get("accepted") is False and not any(
            "unknown_delivery" in str(reason) for reason in row.get("rejection_reasons") or []
        ):
            rejected[code].add(digest)
    for code, digests in selected.items():
        existing[code].update(digests)
    contacted = {
        str(row.get("image_sha256") or "")
        for row in read_jsonl(staging_root / "collection/contacted_images.jsonl")
    } - {""}
    oracle_contacted = _oracle_contacted_images(staging_root, codes)
    oracle_complete = _oracle_complete_images(staging_root, codes)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(dataset_root / "vlm_data/candidates/image_pool.jsonl"):
        code = str(row.get("class_code") or "")
        digest = str(row.get("image_sha256") or "")
        if code not in codes or not digest or digest in existing[code] or digest in oracle_complete:
            continue
        if digest in oracle_contacted and not allow_oracle_replay:
            continue
        if row.get("sft_eligible") is not True or row.get("image_split") != "train_candidate":
            continue
        if digest not in contacted or digest in rejected[code]:
            candidates[code].append({**row, "authorized_oracle_replay": digest in rejected[code]})

    chosen: list[dict[str, Any]] = []
    capacity_limited: dict[str, dict[str, int]] = {}
    shortfalls: dict[str, int] = {}
    for code in sorted(codes):
        shortfall = max(0, 5 - len(existing[code]))
        requested = math.ceil(shortfall * oversample)
        shortfalls[code] = shortfall
        fresh = [row for row in candidates[code] if not row["authorized_oracle_replay"]]
        replay = [row for row in candidates[code] if row["authorized_oracle_replay"]]
        direct_clean = _direct_clean_replays(staging_root, code)
        clean_replay = [row for row in replay if str(row["image_sha256"]) in direct_clean]
        other_replay = [row for row in replay if str(row["image_sha256"]) not in direct_clean]
        choices = [
            *stable_order(fresh, f"{seed}:oracle:{code}:fresh"),
            *stable_order(clean_replay, f"{seed}:oracle:{code}:direct-clean"),
            *stable_order(other_replay, f"{seed}:oracle:{code}:replay"),
        ][:requested]
        if len(choices) < requested:
            capacity_limited[code] = {
                "shortfall": shortfall, "requested_images": requested,
                "fresh_available_images": len(fresh), "replay_available_images": len(replay),
                "selected_images": len(choices),
            }
        chosen.extend({**row, "required_images": shortfall} for row in choices)

    plan, alignment = [], []
    for image in chosen:
        code = str(image["class_code"]); digest = str(image["image_sha256"])
        negatives = [str(item) for item in roles[code].get("milvus_top3_similar_classes") or [] if str(item) in catalog and str(item) != code]
        if len(negatives) != 3 or len(set(negatives)) != 3:
            raise DataError(f"{code} lacks three frozen catalog hard negatives")
        classes = [public_class(catalog, item) for item in [code, *negatives]]
        image_ref = hashlib.sha256(f"{seed}:{recovery_id}:oracle:{digest}".encode()).hexdigest()[:20]
        for language in LANGUAGES:
            for question_type in QUESTION_TYPES:
                sample_id = f"oav2o-{image_ref}-rag-{question_type}-{language}"
                ordered = stable_order(classes, f"{seed}:{sample_id}:options")
                option_index = next(index for index, item in enumerate(ordered) if item["name"] == str(catalog[code]["english_name"]))
                lineage = {
                    "authorized_by": "user", "scope": "N04094_N04113_oracle_rag_only",
                    "standard_answer_private_to_micu": True,
                    "visible_trajectory_must_be_evidence_anchored_and_leak_free": True,
                    "authorized_oracle_replay": bool(image["authorized_oracle_replay"]),
                    "oracle_replay_allowed": allow_oracle_replay,
                    "prior_oracle_attempted": digest in oracle_contacted,
                }
                common = {
                    "sample_id": sample_id, "source_sample_id": sample_id, "target_id": sample_id,
                    "image_ref": image_ref, "image_sha256": digest, "query_image": str(image["image_path"]),
                    "task_domain": str(image["domain"]), "language": language, "question_type": question_type,
                    "final_label": code, "canonical_class": code,
                    "final_label_name": str(catalog[code]["english_name"]),
                    "final_label_zh": str(catalog[code]["chinese_name"]),
                    "correct_option": "ABCD"[option_index] if question_type == "option" else None,
                    "candidate_labels": [{"code": item, **public_class(catalog, item)} for item in [code, *negatives]],
                    "public_option_labels": ordered, "generation_route": "oracle_grounded",
                    "label_visible_to_teacher": True, "trajectory_mode": "standard",
                    "strategy_id": "hcv_visual_expand", "approval_only": False, "reserve": False,
                    "special_oracle": lineage,
                }
                plan.append(common)
                alignment.append({
                    "sample_id": sample_id, "class_code": code, "query_image": str(image["image_path"]),
                    "image_sha256": digest, "truth_name": str(catalog[code]["english_name"]),
                    "truth_name_zh": str(catalog[code]["chinese_name"]),
                    "correct_option": common["correct_option"], "special_oracle": lineage,
                })
    write_rows(recovery_root / "private/rag_plan.jsonl", plan)
    write_rows(recovery_root / "private/alignment.jsonl", alignment)
    report = {
        "schema_version": "agrinet.open-agri-v2.oracle-rag-recovery-plan/v1",
        "recovery_id": recovery_id, "recovery_root": str(recovery_root), "oversample": oversample, "shortfalls": shortfalls,
        "recovery_codes": sorted(codes), "allow_oracle_replay": allow_oracle_replay,
        "oracle_contacted_images_excluded": 0 if allow_oracle_replay else len(oracle_contacted),
        "oracle_complete_images_excluded": len(oracle_complete),
        "planned_images": len(chosen), "planned_views": len(plan),
        "planned_images_by_class": dict(sorted(Counter(str(row["class_code"]) for row in chosen).items())),
        "oracle_replay_images_by_class": dict(sorted(Counter(str(row["class_code"]) for row in chosen if row["authorized_oracle_replay"]).items())),
        "capacity_limited": capacity_limited, "formal_dataset_modified": False,
        "ordinary_blind_review_superseded": False, "human_review_required": True,
    }
    report_path = recovery_root / "reports/plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_oracle_recovery_review(
    staging_root: Path, *, recovery_id: str = "oracle-n04094-n04113-v1",
) -> dict[str, Any]:
    """Audit the user-authorized Oracle trajectories without promoting them.

    The collector already rejects premature target queries and internal-knowledge
    language. This second, compact package makes that contract human-auditable
    and keeps the Oracle route separate from the normal blind eight-view ledger.
    """
    root = _oracle_recovery_root(staging_root, recovery_id)
    accepted = read_jsonl(root / "collection/rag/accepted.jsonl")
    rejected = read_jsonl(root / "collection/rag/rejected.jsonl")
    forbidden = (
        "oracle label", "teacher forcing", "teacher-forced", "ground truth",
        "hidden label", "private label", "answer key", "标准答案", "真实标签",
    )
    rows, failures = [], []
    for row in accepted:
        metadata = row.get("metadata") or {}
        sample_id = str(row.get("sample_id") or "")
        visible = "\n".join(
            str(message.get("content") or "")
            for message in row.get("messages") or []
        )
        influence = metadata.get("label_influence_audit") or {}
        errors = []
        if metadata.get("generation_route") != "oracle_grounded":
            errors.append("wrong_generation_route")
        if metadata.get("label_visible_to_teacher") is not True:
            errors.append("oracle_teacher_visibility_not_recorded")
        if influence.get("premature_target_query") is True:
            errors.append("premature_target_query")
        if influence.get("final_evidence_supported") is not True:
            errors.append("target_not_in_public_retrieval_evidence")
        lowered = visible.lower()
        if any(token in lowered for token in forbidden):
            errors.append("oracle_or_truth_language_in_visible_trajectory")
        if re.search(r"\bN0[45]\d{3}\b", visible, flags=re.IGNORECASE):
            errors.append("class_code_in_visible_trajectory")
        if re.search(r"(?:^|[\s\n])[A-D]\s*(?:=|:|对应|是)\s*", visible):
            errors.append("option_mapping_in_visible_trajectory")
        if re.search(r"(?:file://|/data/)", visible, flags=re.IGNORECASE):
            errors.append("local_path_in_visible_trajectory")
        item = {
            "sample_id": sample_id, "image_sha256": metadata.get("image_sha256"),
            "class_code": metadata.get("canonical_class"), "accepted": not errors,
            "errors": errors, "label_influence_audit": influence,
        }
        rows.append(item)
        if errors:
            failures.append(item)
    write_rows(root / "review/trajectory_decisions.jsonl", rows)
    report = {
        "schema_version": "agrinet.open-agri-v2.oracle-rag-recovery-review/v1", "recovery_id": recovery_id,
        "accepted_trajectories": len(accepted), "rejected_trajectories": len(rejected),
        "leak_free_accepted_trajectories": sum(row["accepted"] for row in rows),
        "leak_or_evidence_failures": len(failures),
        "failures_by_reason": dict(sorted(Counter(reason for row in failures for reason in row["errors"]).items())),
        "ordinary_blind_review_superseded": False, "eligible_for_formal_merge": False,
        "human_review_required": True, "formal_dataset_modified": False,
    }
    destination = root / "review/summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_oracle_provisional_review(staging_root: Path) -> dict[str, Any]:
    """Reaudit and package Oracle-RAG evidence without promoting it.

    This is intentionally a separate human-review layer.  It accepts an image
    into the provisional package only when one Oracle recovery supplies all
    four planned RAG views, each collector-accepted and each passing the
    no-leakage/public-evidence review.  Normal/blind review ledgers and the
    formal dataset are never read as writable targets here.
    """
    recovery_roots = [
        path for path in sorted((staging_root / "recoveries").glob("oracle-*"))
        if (path / "private/rag_plan.jsonl").is_file()
    ]
    if not recovery_roots:
        raise DataError("Oracle provisional audit requires at least one recovery plan")

    provisional = staging_root / "oracle_provisional"
    trajectories: list[dict[str, Any]] = []
    candidates_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    audit_reports: dict[str, dict[str, Any]] = {}
    integrity_failures: list[str] = []

    for root in recovery_roots:
        recovery_id = root.name
        # Re-run the content audit from raw accepted traces before aggregating.
        audit_reports[recovery_id] = write_oracle_recovery_review(
            staging_root, recovery_id=recovery_id,
        )
        plan = read_jsonl(root / "private/rag_plan.jsonl")
        accepted = read_jsonl(root / "collection/rag/accepted.jsonl")
        rejected = read_jsonl(root / "collection/rag/rejected.jsonl")
        review = {str(row.get("sample_id") or ""): row for row in read_jsonl(root / "review/trajectory_decisions.jsonl")}
        plan_by_id = {str(row.get("sample_id") or ""): row for row in plan}
        accepted_by_id = {str(row.get("sample_id") or ""): row for row in accepted}
        rejected_by_id = {str(row.get("sample_id") or ""): row for row in rejected}
        if len(plan_by_id) != len(plan):
            integrity_failures.append(f"{recovery_id}:duplicate_planned_sample_id")
        if set(accepted_by_id) & set(rejected_by_id):
            integrity_failures.append(f"{recovery_id}:terminal_sample_in_both_accepted_and_rejected")
        if set(plan_by_id) != set(accepted_by_id) | set(rejected_by_id):
            integrity_failures.append(f"{recovery_id}:planned_and_terminal_sample_sets_differ")

        grouped: dict[str, list[str]] = defaultdict(list)
        for sample_id, planned in plan_by_id.items():
            digest = str(planned.get("image_sha256") or "")
            code = str(planned.get("final_label") or "")
            if not digest or code not in ORACLE_RECOVERY_CODES:
                integrity_failures.append(f"{recovery_id}:invalid_plan_identity:{sample_id}")
                continue
            grouped[digest].append(sample_id)
            collected = accepted_by_id.get(sample_id)
            collector_accepted = collected is not None
            decision = review.get(sample_id) if collector_accepted else None
            audit_accepted = bool(decision and decision.get("accepted") is True)
            if collector_accepted:
                metadata = collected.get("metadata") or {}
                if str(metadata.get("image_sha256") or "") != digest:
                    integrity_failures.append(f"{recovery_id}:accepted_image_mismatch:{sample_id}")
                if str(metadata.get("canonical_class") or "") != code:
                    integrity_failures.append(f"{recovery_id}:accepted_class_mismatch:{sample_id}")
                if decision is None:
                    integrity_failures.append(f"{recovery_id}:missing_audit_decision:{sample_id}")
            reason = []
            if not collector_accepted:
                reason.extend(str(item) for item in (rejected_by_id.get(sample_id, {}).get("reasons") or ["collector_rejected"]))
            if collector_accepted and not audit_accepted:
                reason.extend(str(item) for item in (decision.get("errors") or ["oracle_audit_rejected"]))
            trajectories.append({
                "sample_id": sample_id, "recovery_id": recovery_id,
                "source": "oracle_rag", "class_code": code, "image_sha256": digest,
                "collector_accepted": collector_accepted, "oracle_audit_accepted": audit_accepted,
                "provisional_status": "pending_image_group" if collector_accepted and audit_accepted else "rejected",
                "rejection_reasons": sorted(set(reason)),
                "source_record": str((root / ("collection/rag/accepted.jsonl" if collector_accepted else "collection/rag/rejected.jsonl")).relative_to(staging_root)),
            })
        for digest, sample_ids in grouped.items():
            first = plan_by_id[sample_ids[0]]
            complete = len(sample_ids) == 4 and all(
                accepted_by_id.get(sample_id) is not None
                and review.get(sample_id, {}).get("accepted") is True
                for sample_id in sample_ids
            )
            if len(sample_ids) != 4:
                integrity_failures.append(f"{recovery_id}:image_without_four_planned_views:{digest}")
            candidates_by_image[digest].append({
                "recovery_id": recovery_id, "class_code": str(first["final_label"]),
                "sample_ids": sorted(sample_ids), "complete": complete,
                "prior_oracle_attempted": bool((first.get("special_oracle") or {}).get("prior_oracle_attempted")),
            })
            if complete:
                accepted_rows[(recovery_id, digest)] = [accepted_by_id[sample_id] for sample_id in sorted(sample_ids)]

    if integrity_failures:
        raise DataError("Oracle provisional audit integrity failure: " + "; ".join(sorted(integrity_failures)))

    selected: list[dict[str, Any]] = []
    selected_sample_ids: set[str] = set()
    image_decisions: list[dict[str, Any]] = []
    for digest, candidates in sorted(candidates_by_image.items()):
        codes = {str(item["class_code"]) for item in candidates}
        if len(codes) != 1:
            raise DataError(f"Oracle provisional audit found conflicting classes for image {digest}")
        passing = [item for item in candidates if item["complete"]]
        # Prefer an original Oracle attempt over an explicit replay, then use a
        # deterministic recovery id.  Alternates remain in the decision ledger.
        chosen = sorted(passing, key=lambda item: (item["prior_oracle_attempted"], item["recovery_id"]))[0] if passing else None
        if chosen:
            selected_sample_ids.update(str(sample_id) for sample_id in chosen["sample_ids"])
            for row in accepted_rows[(str(chosen["recovery_id"]), digest)]:
                item = dict(row)
                metadata = dict(item.get("metadata") or {})
                metadata.update({
                    "source": "oracle_rag", "supplement_staging": True,
                    "oracle_provisional": True, "oracle_recovery_id": chosen["recovery_id"],
                    "ordinary_blind_review_superseded": False,
                    "eligible_for_formal_merge": False, "human_review_required": True,
                    "training_eligible": False,
                })
                item["metadata"] = metadata
                selected.append(item)
        image_decisions.append({
            "image_sha256": digest, "class_code": next(iter(codes)), "source": "oracle_rag",
            "provisional_accepted": chosen is not None,
            "selected_recovery_id": chosen["recovery_id"] if chosen else None,
            "eligible_for_formal_merge": False, "human_review_required": True,
            "ordinary_blind_review_superseded": False,
            "attempts": candidates,
        })
    for trajectory in trajectories:
        if not (trajectory["collector_accepted"] and trajectory["oracle_audit_accepted"]):
            trajectory["provisional_selected"] = False
        elif trajectory["sample_id"] in selected_sample_ids:
            trajectory["provisional_status"] = "selected"
            trajectory["provisional_selected"] = True
        else:
            trajectory["provisional_status"] = "excluded_incomplete_image"
            trajectory["provisional_selected"] = False

    write_rows(provisional / "accepted/agent_sft.accepted.jsonl", selected)
    write_rows(provisional / "review/trajectory_decisions.jsonl", trajectories)
    write_rows(provisional / "review/image_decisions.jsonl", image_decisions)
    report = {
        "schema_version": "agrinet.open-agri-v2.oracle-provisional-review/v1",
        "recovery_ids": [root.name for root in recovery_roots],
        "source": "oracle_rag", "reviewed_trajectories": len(trajectories),
        "trajectory_audit_accepted": sum(
            row["collector_accepted"] and row["oracle_audit_accepted"]
            for row in trajectories
        ),
        "rejected_trajectories": sum(row["provisional_status"] == "rejected" for row in trajectories),
        "provisional_selected_trajectories": len(selected),
        "trajectory_passes_excluded_for_incomplete_image": sum(
            row["collector_accepted"] and row["oracle_audit_accepted"]
            for row in trajectories
        ) - len(selected),
        "accepted_images_by_class": dict(sorted(Counter(str(row["class_code"]) for row in image_decisions if row["provisional_accepted"]).items())),
        "rejected_images_by_class": dict(sorted(Counter(str(row["class_code"]) for row in image_decisions if not row["provisional_accepted"]).items())),
        "audit_reports": audit_reports,
        "ordinary_blind_review_superseded": False, "eligible_for_formal_merge": False,
        "human_review_required": True, "training_eligible": False,
        "formal_dataset_modified": False,
    }
    destination = provisional / "reports/audit_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_oracle_final_export(staging_root: Path) -> dict[str, Any]:
    """Freeze complete Oracle outcomes as a human-review-only evidence export."""
    review = write_oracle_provisional_review(staging_root)
    provisional = staging_root / "oracle_provisional"
    export_root = staging_root / "oracle_final_export"
    decisions = read_jsonl(provisional / "review/trajectory_decisions.jsonl")
    images = read_jsonl(provisional / "review/image_decisions.jsonl")
    accepted = read_jsonl(provisional / "accepted/agent_sft.accepted.jsonl")
    decision_by_id = {str(row.get("sample_id") or ""): row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise DataError("Oracle final export requires unique trajectory decisions")
    selected_ids = {sample_id for sample_id, row in decision_by_id.items() if row.get("provisional_status") == "selected"}
    accepted_ids = {str(row.get("sample_id") or "") for row in accepted}
    if selected_ids != accepted_ids:
        raise DataError("Oracle final export selected decisions do not match accepted records")

    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    for recovery_id in review["recovery_ids"]:
        root = _oracle_recovery_root(staging_root, str(recovery_id))
        for outcome, relative in (("collector_accepted", "collection/rag/accepted.jsonl"), ("collector_rejected", "collection/rag/rejected.jsonl")):
            path = root / relative
            rows = read_jsonl(path)
            source_files.append({"recovery_id": recovery_id, "outcome": outcome, "path": str(path.relative_to(staging_root)), "rows": len(rows), "sha256": file_sha256(path)})
            for row in rows:
                sample_id = str(row.get("sample_id") or "")
                key = (str(recovery_id), sample_id)
                if not sample_id or key in source_rows:
                    raise DataError(f"Oracle final export has invalid or duplicate source record: {key}")
                source_rows[key] = row

    terminal, rejected, incomplete = [], [], []
    for decision in decisions:
        recovery_id = str(decision["recovery_id"])
        sample_id = str(decision["sample_id"])
        source = source_rows.get((recovery_id, sample_id))
        if source is None:
            raise DataError(f"Oracle final export is missing source record: {recovery_id}:{sample_id}")
        status = str(decision["provisional_status"])
        item = {"schema_version": "agrinet.open-agri-v2.oracle-final-export-record/v1", "outcome": status, "terminal_decision": decision, "source_record": source}
        terminal.append(item)
        if status == "rejected":
            rejected.append(item)
        elif status == "excluded_incomplete_image":
            incomplete.append(item)
        elif status != "selected":
            raise DataError(f"Oracle final export found unfinished decision status: {status}")
    if len(terminal) != len(source_rows) or len(terminal) != review["reviewed_trajectories"]:
        raise DataError("Oracle final export terminal coverage is incomplete")

    write_rows(export_root / "accepted/agent_sft.complete_four_view.jsonl", accepted)
    write_rows(export_root / "rejected/collector_or_audit_rejected.jsonl", rejected)
    write_rows(export_root / "rejected/audit_pass_but_incomplete_image.jsonl", incomplete)
    write_rows(export_root / "review/terminal_records.jsonl", terminal)
    write_rows(export_root / "review/trajectory_decisions.jsonl", decisions)
    write_rows(export_root / "review/image_decisions.jsonl", images)
    write_rows(export_root / "provenance/source_files.jsonl", source_files)
    report = {"schema_version": "agrinet.open-agri-v2.oracle-final-export/v1", "status": "complete", "source": "oracle_rag", "recovery_ids": review["recovery_ids"], "terminal_trajectories": len(terminal), "accepted_complete_four_view_trajectories": len(accepted), "rejected_collector_or_audit_trajectories": len(rejected), "rejected_incomplete_image_trajectories": len(incomplete), "accepted_images_by_class": review["accepted_images_by_class"], "rejected_images_by_class": review["rejected_images_by_class"], "source_files": source_files, "ordinary_blind_review_superseded": False, "eligible_for_formal_merge": False, "human_review_required": True, "training_eligible": False, "formal_dataset_modified": False}
    destination = export_root / "reports/final_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_n04094_rag_502_recovery(staging_root: Path) -> dict[str, Any]:
    """Authorize only the user-approved N04094 RAG retry after the 8077 outage.

    The four images must have passed all Direct views, while all four original
    RAG views failed because the local retrieval service returned no result.
    This writes a new RAG-only plan using the original sample IDs so review can
    replace only the outage-derived RAG decisions; it never deletes history.
    """
    source_plan = read_jsonl(staging_root / "private/rag_plan.jsonl")
    alignment = {str(row["sample_id"]): row for row in read_jsonl(staging_root / "private/alignment.jsonl")}
    direct = {str(row.get("sample_id") or ""): row for row in read_jsonl(staging_root / "collection/direct/ledger.jsonl")}
    prior_rejected = {str(row.get("sample_id") or ""): row for row in read_jsonl(staging_root / "collection/rag-v2/rejected.jsonl")}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_plan:
        truth = alignment.get(str(row.get("sample_id") or ""))
        if truth is not None and truth.get("class_code") == "N04094":
            by_image[str(row["image_sha256"])].append(row)
    selected: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for digest, rows in sorted(by_image.items()):
        direct_ids = [str(row["sample_id"]).replace("-rag-", "-direct-") for row in rows]
        direct_pass = len(direct_ids) == 4 and all(direct.get(sample_id, {}).get("accepted") is True for sample_id in direct_ids)
        rag_reasons = [
            set(prior_rejected.get(str(row["sample_id"]), {}).get("reasons") or [])
            for row in rows
        ]
        outage_only = len(rows) == 4 and all("no_successful_retrieval" in reasons for reasons in rag_reasons)
        if direct_pass and outage_only:
            selected.extend(rows)
            evidence.append({"image_sha256": digest, "original_rag_sample_ids": [str(row["sample_id"]) for row in rows]})
    if len(selected) != 16 or len(evidence) != 4:
        raise DataError(f"N04094 recovery scope changed: expected 4 images / 16 views, got {len(evidence)} / {len(selected)}")
    recovery_root = staging_root / "recoveries/n04094-rag-8077-502-v1"
    recovered_plan = [
        {**row, "special_recovery": {
            "authorized_by": "user", "reason": "local_rag_8077_http_502",
            "replay_scope": "N04094_direct_four_pass_only", "replay_count": 1,
        }}
        for row in selected
    ]
    write_rows(recovery_root / "private/rag_plan.jsonl", recovered_plan)
    write_rows(recovery_root / "private/alignment.jsonl", [alignment[str(row["sample_id"])] for row in selected])
    report = {
        "schema_version": "agrinet.open-agri-v2.n04094-rag-502-recovery-plan/v1",
        "class_code": "N04094", "recovery_root": str(recovery_root),
        "recovered_images": len(evidence), "recovered_views": len(recovered_plan),
        "authorized_replay_reason": "user-approved local RAG 8077 HTTP 502 recovery",
        "original_rejections_preserved": True, "formal_dataset_modified": False,
        "evidence": evidence,
    }
    path = recovery_root / "reports/recovery_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def direct_private_alignment(staging_root: Path) -> list[dict[str, Any]]:
    return [
        row for row in read_jsonl(staging_root / "private/alignment.jsonl")
        if "-direct-" in str(row.get("sample_id") or "")
    ]


def _formal_metadata(row: dict[str, Any], truth: dict[str, Any], route: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update({
        "class_code": truth["class_code"],
        "class_role": "known",
        "task_domain": metadata.get("task_domain") or row.get("task_domain"),
        "language": metadata.get("language") or row.get("language"),
        "question_type": metadata.get("question_type") or row.get("question_type"),
        "route": route,
        "v2_image_sha256": truth["image_sha256"],
        "v2_image_split": "train_candidate",
        "supplement_staging": True,
    })
    return metadata


def stage_direct_acceptance(
    staging_root: Path,
    *,
    collection_root: Path | None = None,
    plan_root: Path | None = None,
) -> dict[str, Any]:
    """Audit Direct results and stage records; this never imports them."""
    from agrinet.research.m1.collection import audit_and_convert

    plan_root = plan_root or staging_root
    public = read_jsonl(plan_root / "public/direct_plan.jsonl")
    private = direct_private_alignment(plan_root)
    collection = collection_root or staging_root / "collection/direct"
    rows, audit = audit_and_convert(
        public,
        private,
        read_jsonl(collection / "teacher_results.jsonl"),
        read_jsonl(collection / "auditor_results.jsonl"),
        reasoning_render="m1_comparison_user_guidance_v3",
    )
    truth = {str(row["sample_id"]): row for row in private}
    staged = []
    for row in rows:
        item = dict(row)
        item["tools"] = "[]"
        item["schema_version"] = "agrinet.open-agri-v2.ms-swift-agent/v1"
        item["metadata"] = _formal_metadata(item, truth[str(item["sample_id"])], "direct")
        staged.append(item)
    write_rows(collection / "accepted.jsonl", staged)
    write_rows(collection / "rejected.jsonl", audit["rejections"])
    summary = {**audit, "formal_dataset_modified": False, "staged": len(staged)}
    (collection / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def stage_rag_acceptance(
    staging_root: Path,
    *,
    collection_root: Path | None = None,
    plan_root: Path | None = None,
) -> dict[str, Any]:
    """Normalize HCV collector output into one-image formal-v2 staging rows."""
    collection = collection_root or staging_root / "collection/rag-v2"
    plan_root = plan_root or staging_root
    private = {
        str(row["sample_id"]): row
        for row in read_jsonl(plan_root / "private/alignment.jsonl")
        if "-rag-" in str(row.get("sample_id") or "")
    }
    accepted, rejected = [], read_jsonl(collection / "traces/rejected_trajectories.jsonl")
    for row in read_jsonl(collection / "train/agent_sft.accepted.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        truth = private.get(sample_id)
        if truth is None:
            rejected.append({"sample_id": sample_id or "<missing>", "reasons": ["missing_private_alignment"]})
            continue
        item = dict(row)
        item["schema_version"] = "agrinet.open-agri-v2.ms-swift-agent/v1"
        item["images"] = [truth["query_image"]]
        item["metadata"] = _formal_metadata(item, truth, "rag")
        accepted.append(item)
    write_rows(collection / "accepted.jsonl", accepted)
    write_rows(collection / "rejected.jsonl", rejected)
    summary = {
        "schema_version": "agrinet.open-agri-v2.supplement-rag-stage/v1",
        "accepted": len(accepted), "rejected": len(rejected),
        "formal_dataset_modified": False,
    }
    (collection / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def image_group_decisions(
    direct_ledger: list[dict[str, Any]],
    rag_accepted: list[dict[str, Any]],
    rag_rejected: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Require every Direct and blind-RAG view of an image to pass together."""
    truth = {str(row["sample_id"]): row for row in alignment}
    direct_by_id = {str(row.get("sample_id") or ""): row for row in direct_ledger}
    rag_by_id = {
        str(row.get("sample_id") or ""): (True, row) for row in rag_accepted
    }
    rag_by_id.update({str(row.get("sample_id") or ""): (False, row) for row in rag_rejected})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, item in truth.items():
        grouped[str(item["image_sha256"])].append({"sample_id": sample_id, "truth": item})

    accepted_images: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for digest, views in sorted(grouped.items()):
        errors: list[str] = []
        code = str(views[0]["truth"]["class_code"])
        direct_ids = [view["sample_id"] for view in views if "-direct-" in view["sample_id"]]
        rag_ids = [view["sample_id"] for view in views if "-rag-" in view["sample_id"]]
        if len(direct_ids) != 4 or len(rag_ids) != 4:
            errors.append("invalid_eight_view_plan")
        for sample_id in direct_ids:
            row = direct_by_id.get(sample_id)
            if row is None:
                errors.append(f"direct_missing:{sample_id}")
            elif row.get("accepted") is not True:
                errors.extend(f"direct:{reason}" for reason in row.get("errors") or ["rejected"])
        for sample_id in rag_ids:
            result = rag_by_id.get(sample_id)
            if result is None:
                errors.append(f"rag_missing:{sample_id}")
            elif result[0] is not True:
                errors.extend(f"rag:{reason}" for reason in result[1].get("reasons") or ["rejected"])
        accepted = not errors
        if accepted:
            accepted_images.add(digest)
        decisions.append({
            "image_sha256": digest,
            "class_code": code,
            "accepted": accepted,
            "rejection_reasons": sorted(set(errors)),
            "direct_views": sorted(direct_ids),
            "rag_views": sorted(rag_ids),
        })
    return decisions, accepted_images


def write_review_package(
    staging_root: Path,
    *,
    direct_ledger_path: Path,
    rag_accepted_path: Path,
    rag_rejected_path: Path,
) -> dict[str, Any]:
    """Produce a complete review ledger without changing the formal dataset."""
    alignment = read_jsonl(staging_root / "private/alignment.jsonl")
    decisions, accepted_images = image_group_decisions(
        read_jsonl(direct_ledger_path),
        read_jsonl(rag_accepted_path),
        read_jsonl(rag_rejected_path),
        alignment,
    )
    initial_plan = json.loads((staging_root / "reports/initial_plan.json").read_text(encoding="utf-8"))
    required = {str(code): int(count) for code, count in (initial_plan.get("shortfalls") or {}).items()}
    accepted_by_code = Counter(row["class_code"] for row in decisions if row["accepted"])
    planned_by_code = Counter(row["class_code"] for row in decisions)
    accepted_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row["accepted"]:
            accepted_rows[row["class_code"]].append(row)
        else:
            row["selected_for_merge"] = False
            row["disposition"] = "quality_or_delivery_rejected"
    for code, rows in accepted_rows.items():
        for index, row in enumerate(stable_order(rows, f"{initial_plan['seed']}:quota:{code}")):
            row["selected_for_merge"] = index < required[code]
            if row["selected_for_merge"]:
                row["disposition"] = "accepted_for_human_review"
            else:
                row["disposition"] = "quota_surplus"
                row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["quota_surplus"]))
    write_rows(staging_root / "review/image_decisions.jsonl", decisions)
    report = {
        "schema_version": "agrinet.open-agri-v2.supplement-review/v1",
        "accepted_images": len(accepted_images),
        "rejected_images": len(decisions) - len(accepted_images),
        "accepted_by_class": dict(sorted(accepted_by_code.items())),
        "selected_for_merge_by_class": dict(sorted(Counter(
            row["class_code"] for row in decisions if row.get("selected_for_merge")
        ).items())),
        "planned_by_class": dict(sorted(planned_by_code.items())),
        "formal_dataset_modified": False,
    }
    path = staging_root / "review/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_aggregate_review(staging_root: Path) -> dict[str, Any]:
    """Combine completed rounds into one quota-aware, human-review ledger."""
    roots = [staging_root]
    roots.extend(sorted(path for path in (staging_root / "rounds").glob("v*") if path.is_dir()))
    alignment, direct_ledger, rag_accepted, rag_rejected = [], [], [], []
    for root in roots:
        alignment.extend(read_jsonl(root / "private/alignment.jsonl"))
        direct_ledger.extend(read_jsonl(root / "collection/direct/ledger.jsonl"))
        rag_root = root / "collection/rag-v2" if root == staging_root else root / "collection/rag"
        rag_accepted.extend(read_jsonl(rag_root / "accepted.jsonl"))
        rag_rejected.extend(read_jsonl(rag_root / "rejected.jsonl"))
    replay_sample_ids = {
        str(row.get("sample_id") or "")
        for row in alignment
        if row.get("special_replay")
    } - {""}
    replay_digests = {
        str(row.get("image_sha256") or "")
        for row in alignment
        if str(row.get("sample_id") or "") in replay_sample_ids
    } - {""}
    if replay_digests:
        prior_ids = {
            str(row.get("sample_id") or "")
            for row in alignment
            if str(row.get("image_sha256") or "") in replay_digests
            and str(row.get("sample_id") or "") not in replay_sample_ids
        }
        # Old evidence stays on disk; the explicit replay is the only active
        # eight-view decision for the selected image groups.
        alignment = [row for row in alignment if str(row.get("sample_id") or "") not in prior_ids]
        direct_ledger = [row for row in direct_ledger if str(row.get("sample_id") or "") not in prior_ids]
        rag_accepted = [row for row in rag_accepted if str(row.get("sample_id") or "") not in prior_ids]
        rag_rejected = [row for row in rag_rejected if str(row.get("sample_id") or "") not in prior_ids]
    recovery_root = staging_root / "recoveries/n04094-rag-8077-502-v1"
    recovery_accepted = read_jsonl(recovery_root / "collection/rag/accepted.jsonl")
    recovery_rejected = read_jsonl(recovery_root / "collection/rag/rejected.jsonl")
    recovery_ids = {
        str(row.get("sample_id") or "") for row in [*recovery_accepted, *recovery_rejected]
    } - {""}
    if recovery_ids:
        # The original 502 records remain on disk, but their terminal decision
        # is superseded only for this explicit user-authorized retry scope.
        rag_rejected = [row for row in rag_rejected if str(row.get("sample_id") or "") not in recovery_ids]
        rag_accepted = [row for row in rag_accepted if str(row.get("sample_id") or "") not in recovery_ids]
        rag_accepted.extend(recovery_accepted)
        rag_rejected.extend(recovery_rejected)
    inputs = staging_root / "review/aggregate_inputs"
    write_rows(inputs / "alignment.jsonl", alignment)
    write_rows(inputs / "direct_ledger.jsonl", direct_ledger)
    write_rows(inputs / "rag_accepted.jsonl", rag_accepted)
    write_rows(inputs / "rag_rejected.jsonl", rag_rejected)
    decisions, accepted_images = image_group_decisions(direct_ledger, rag_accepted, rag_rejected, alignment)
    initial_plan = json.loads((staging_root / "reports/initial_plan.json").read_text(encoding="utf-8"))
    required = {str(code): int(count) for code, count in (initial_plan.get("shortfalls") or {}).items()}
    accepted_by_code = Counter(row["class_code"] for row in decisions if row["accepted"])
    planned_by_code = Counter(row["class_code"] for row in decisions)
    accepted_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row["accepted"]:
            accepted_rows[row["class_code"]].append(row)
        else:
            row["selected_for_merge"] = False
            row["disposition"] = "quality_or_delivery_rejected"
    for code, rows in accepted_rows.items():
        for index, row in enumerate(stable_order(rows, f"{initial_plan['seed']}:quota:{code}")):
            row["selected_for_merge"] = index < required[code]
            row["disposition"] = "accepted_for_human_review" if row["selected_for_merge"] else "quota_surplus"
            if not row["selected_for_merge"]:
                row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["quota_surplus"]))
    write_rows(staging_root / "review/image_decisions.jsonl", decisions)
    report = {
        "schema_version": "agrinet.open-agri-v2.supplement-review/v1",
        "round_roots": [str(root) for root in roots],
        "accepted_images": len(accepted_images), "rejected_images": len(decisions) - len(accepted_images),
        "accepted_by_class": dict(sorted(accepted_by_code.items())),
        "selected_for_merge_by_class": dict(sorted(Counter(row["class_code"] for row in decisions if row.get("selected_for_merge")).items())),
        "planned_by_class": dict(sorted(planned_by_code.items())),
        "recovery_overrides": {
            "n04094_rag_8077_502": len(recovery_ids),
            "round6_user_authorized_replay_images": len(replay_digests),
        },
        "formal_dataset_modified": False,
    }
    (staging_root / "review/summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_final_shortfall_report(
    repo_root: Path,
    dataset_root: Path,
    staging_root: Path,
    *,
    completed_rounds: int,
) -> dict[str, Any]:
    """Record final capacity after all currently authorized fresh retry rounds."""
    decisions = read_jsonl(staging_root / "review/image_decisions.jsonl")
    roles = {
        str(row["class_code"]): row
        for row in read_jsonl(dataset_root / "manifests/class_split.jsonl")
    }
    known = {
        code for code, row in roles.items()
        if row.get("class_role") == "known" and row.get("sft_eligible") is True
    }
    if len(known) != 109:
        raise DataError(f"expected 109 Known classes, found {len(known)}")
    existing = existing_images(dataset_root)
    selected: dict[str, set[str]] = defaultdict(set)
    for row in decisions:
        if row.get("selected_for_merge") is True:
            selected[str(row["class_code"])].add(str(row["image_sha256"]))
    accepted_by_class = {
        code: len(existing[code] | selected[code]) for code in sorted(known)
    }
    shortfalls = {
        code: 5 - count for code, count in accepted_by_class.items() if count < 5
    }
    report = {
        "schema_version": "agrinet.open-agri-v2.supplement-final-shortfall/v1",
        "known_classes": len(known),
        "target_images_per_class": 5,
        "selected_staged_images": sum(len(images) for images in selected.values()),
        "coverage_after_selected_staging": accepted_by_class,
        "classes_below_floor": len(shortfalls),
        "accepted_image_shortfall": sum(shortfalls.values()),
        "shortfalls": shortfalls,
        "automatic_replenishment_rounds_completed": completed_rounds,
        "automatic_replenishment_stopped": True,
        "stop_reason": "maximum_permitted_replenishment_rounds_exhausted",
        "formal_dataset_modified": False,
    }
    path = staging_root / f"reports/final_shortfall_after_round{completed_rounds}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def require_human_approval(staging_root: Path) -> dict[str, Any]:
    path = staging_root / "review/approval.json"
    if not path.is_file():
        raise DataError(f"human approval file is required: {path}")
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True:
        raise DataError("human approval does not set approved=true")
    if not str(approval.get("reviewer") or "").strip():
        raise DataError("human approval lacks reviewer")
    if not str(approval.get("reviewed_at") or "").strip():
        raise DataError("human approval lacks reviewed_at")
    return approval


def merge_sources(staging_root: Path, accepted_images: set[str]) -> tuple[Path, Path]:
    """Select only complete reviewed image groups into staging merge sources."""
    roots = [staging_root]
    roots.extend(sorted(path for path in (staging_root / "rounds").glob("v*") if path.is_dir()))
    sources: dict[str, list[dict[str, Any]]] = {"direct": [], "rag": []}
    for root in roots:
        sources["direct"].extend(read_jsonl(root / "collection/direct/accepted.jsonl"))
        rag_root = root / ("collection/rag-v2" if root == staging_root else "collection/rag")
        sources["rag"].extend(read_jsonl(rag_root / "accepted.jsonl"))
    outputs: dict[str, Path] = {}
    for route, rows in sources.items():
        selected = [
            row for row in rows
            if str((row.get("metadata") or {}).get("v2_image_sha256") or (row.get("metadata") or {}).get("image_sha256") or "") in accepted_images
        ]
        path = staging_root / "merge" / f"reviewed_{route}.jsonl"
        write_rows(path, selected)
        outputs[route] = path
    return outputs["direct"], outputs["rag"]
