"""Offline contract and state-machine checks for the E3.5 dual-arm RAG distillation line.

This deliberately contains no provider client.  A separate collector may only
consume manifests accepted here; truth, folds, and private-audit text stay in
private sidecars and are never projected to teacher payloads.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml

from agrinet.rag.hermes_protocol import normalize_training_messages

ARMS = ("known", "simulated_unknown")
ROUTES = ("direct", "classifier", "rag")
IDENTITIES = ("image_sha256", "source_group_id", "near_duplicate_group_id")
DELIVERY_FAILURES = {"unknown_delivery", "truncated", "invalid_response", "ledger_incomplete"}
BUDGET_SHORTFALL = "budget_shortfall"
TOOL_SHORTFALL = "tool_shortfall"
PRIVATE_TOKENS = {"canonical_class_code", "truth_code", "held_out_fold", "label_map",
                  "checkpoint_sha256", "training_manifest", "private_audit", "simulated_unknown",
                  "audit_protocol", "rag_witness"}
SYSTEM_PROMPT = ("You are an agricultural visual diagnostician. Use only the image, the user question, "
                 "and actual public tool results. Do not infer private labels or classifier coverage. "
                 "Answer with a supported public class name, or INSUFFICIENT_EVIDENCE only after RAG evidence is insufficient.")
DEFAULT_TEACHER_SYSTEM_PROMPTS = {
    "direct": SYSTEM_PROMPT + " This is the Direct route: no tools are available. Return <think>brief visible evidence</think><answer>canonical public class name</answer>.",
    "classifier": SYSTEM_PROMPT + " This is the Classifier route. The API exposes only agrinet_classifier_predict and agrinet_classifier_expand with their full JSON Schemas. Use native tool calls only when needed; after actual public results, return <think>evidence</think><answer>canonical public class name</answer>.",
    "rag": SYSTEM_PROMPT + " This is the RAG route. The API exposes agrinet_classifier_predict, agrinet_classifier_expand, and agrinet_rag_search with their full JSON Schemas. Your first action must be the native agrinet_rag_search tool call. Do not finalize before its actual response. Then return <think>public evidence</think><answer>canonical public class name or INSUFFICIENT_EVIDENCE</answer>.",
}


def teacher_system_prompt_for_route(row: dict[str, Any], route: str) -> str:
    """Teacher-only collection prompt; never the Hermes export prompt."""
    prompts = row.get("teacher_system_prompts")
    if isinstance(prompts, dict) and isinstance(prompts.get(route), str) and prompts[route].strip():
        return prompts[route]
    return str(row.get("system_prompt") or SYSTEM_PROMPT)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_version": "agrinet.e35-dual-arm-rag-distill/v1",
        "dataset.known_classes": 107, "sampling.per_class": 5,
        "sampling.open_per_class": 3, "sampling.option_per_class": 2,
        "audit.images": 32, "audit.per_arm": 16,
        "recovery.rounds": ["R0", "R1", "R2"],
        "recovery.max_new_attempts": 2, "budgets.micu_intents": 8000,
        "gates.training_eligible": False, "gates.training_authorized": False,
        "gates.sft_may_start": False, "rag.full_registry_visible": True,
        "rag.combine_classifier_and_retrieval_scores": False,
    }
    for dotted, expected in required.items():
        value: Any = contract
        for key in dotted.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"contract mismatch: {dotted}")
    if set(contract.get("arms", {})) != set(ARMS):
        raise ValueError("contract must bind both arms")
    if contract["arms"]["known"].get("classifier_count") != 3:
        raise ValueError("Known arm requires three OOF classifiers")
    if contract["arms"]["simulated_unknown"].get("classifier_count") != 3:
        raise ValueError("simulated Unknown arm requires three class-fold classifiers")


def public_teacher_input(row: dict[str, Any], route: str) -> dict[str, Any]:
    """Strict projection: source paths and all private provenance are excluded."""
    if route not in ROUTES:
        raise ValueError("unknown route")
    result = {"system": teacher_system_prompt_for_route(row, route), "question": row["question"],
              "image": {"sha256": row["image_sha256"]}}
    if row.get("question_type") == "option":
        options = row.get("public_options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("Option teacher input requires four public options")
        result["options"] = [{"label": item.get("label"), "name": item.get("name")}
                             for item in options if isinstance(item, dict)]
        if len(result["options"]) != 4 or any(not isinstance(item["label"], str) or not isinstance(item["name"], str)
                                           for item in result["options"]):
            raise ValueError("Option teacher input contains malformed public options")
    if route in {"classifier", "rag"}:
        result["tools"] = ["agrinet_classifier_predict", "agrinet_classifier_expand"]
    if route == "rag":
        result["tools"].append("agrinet_rag_search")
    return result


def public_classifier_card(row: dict[str, Any], *, expanded: bool = False) -> dict[str, Any]:
    """Expose only ranked public names and scores, never classifier lineage."""
    prediction = row.get("classifier", {}).get("top5")
    if not isinstance(prediction, list) or len(prediction) != 5:
        raise ValueError("frozen classifier card requires exactly five predictions")
    limit = 5 if expanded else 3
    candidates, previous = [], 1.0
    for rank, item in enumerate(prediction[:limit], 1):
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                not isinstance(item.get("score"), (float, int)) or not 0 <= item["score"] <= previous):
            raise ValueError("classifier card candidate lacks public name")
        previous = item["score"]
        score = float(Decimal(str(item["score"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        candidates.append({"rank": rank, "name": item["name"], "score": score})
    return {"tool": "agrinet_classifier_expand" if expanded else "agrinet_classifier_predict",
            "candidates": candidates}


def cascade_outcome(*, work_item: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the recorded per-image cascade before it can be a winner."""
    resume_route = str(work_item.get("resume_route") or "direct")
    if resume_route not in ROUTES:
        raise ValueError("cascade continuation route is invalid")
    expected = list(ROUTES[ROUTES.index(resume_route):])
    final_route = None
    for index, stage in enumerate(stages):
        route = stage.get("route")
        if route != expected[index]:
            raise ValueError("cascade route order is invalid")
        delivery = stage.get("delivery_status")
        if delivery in DELIVERY_FAILURES:
            return {"work_id": work_item["work_id"], "delivery_status": delivery,
                    "request_id": stage.get("request_id"), "winner": False, "final_route": route,
                    "quality_status": None, "parent_path": stage.get("parent_path"),
                    "contract_error": stage.get("contract_error")}
        # A tool failure happens only after the teacher response carrying the
        # native call has been durably recorded.  It is consequently neither a
        # provider-delivery ambiguity nor eligible for automatic R1/R2 replay.
        # A later repair must use an explicit, newly-bound continuation scope.
        if delivery == TOOL_SHORTFALL:
            return {"work_id": work_item["work_id"], "delivery_status": "delivered",
                    "request_id": stage.get("request_id"), "winner": False, "final_route": route,
                    "quality_status": str(stage.get("quality_status") or "local_tool_failure"),
                    "parent_path": stage.get("parent_path"),
                    "contract_error": stage.get("contract_error")}
        if delivery != "delivered":
            raise ValueError("cascade stage lacks a valid delivery result")
        audit = stage.get("private_audit")
        if audit == "accept":
            final_route = route
            return {"work_id": work_item["work_id"], "delivery_status": "delivered",
                    "request_id": stage.get("request_id"), "winner": True, "final_route": final_route,
                    "quality_status": "accepted", "parent_path": stage.get("parent_path")}
        if audit != "reject":
            return {"work_id": work_item["work_id"], "delivery_status": "delivered",
                    "request_id": stage.get("request_id"), "winner": False, "final_route": route,
                    "quality_status": str(audit or "missing_private_audit"), "parent_path": stage.get("parent_path"),
                    "contract_error": stage.get("contract_error")}
        if stage.get("terminal_reject") is True:
            return {"work_id": work_item["work_id"], "delivery_status": "delivered",
                    "request_id": stage.get("request_id"), "winner": False, "final_route": route,
                    "quality_status": "quality_rejected", "parent_path": stage.get("parent_path"),
                    "contract_error": stage.get("contract_error")}
        if route == "rag":
            return {"work_id": work_item["work_id"], "delivery_status": "delivered",
                    "request_id": stage.get("request_id"), "winner": False, "final_route": route,
                    "quality_status": "quality_rejected", "parent_path": stage.get("parent_path")}
    raise ValueError("rejected stage must progress to its next route")


def build_candidate_source(pools: dict[str, list[dict[str, Any]]], *, registry: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Freeze five deterministic, three-key-isolated rows per class and arm.

    Pool rows must already carry their frozen classifier card.  This function
    only assigns Open/Open/Open/Option/Option once; it never clones an image to
    manufacture a second question form.
    """
    chosen: list[dict[str, Any]] = []
    identities = {key: set() for key in IDENTITIES}
    for arm in ARMS:
        by_class: dict[str, list[dict[str, Any]]] = {}
        for row in pools.get(arm, []):
            by_class.setdefault(str(row.get("canonical_class_code") or ""), []).append(row)
        if len(by_class) != 107:
            raise ValueError(f"{arm} pool must cover exactly 107 canonical classes")
        for code in sorted(by_class):
            ranked = sorted(by_class[code], key=lambda row: str(row.get("sample_id") or ""))
            available = []
            for row in ranked:
                if all(isinstance(row.get(key), str) and row[key] not in identities[key] for key in IDENTITIES):
                    available.append(row)
                if len(available) == 5: break
            if len(available) != 5:
                raise ValueError(f"{arm}/{code} lacks five identity-isolated rows")
            for number, row in enumerate(available):
                frozen = dict(row)
                frozen["arm"] = arm
                frozen["question_type"] = "open" if number < 3 else "option"
                frozen["sample_id"] = f"e35-{arm}-{code}-{frozen['image_sha256'][:20]}"
                frozen["system_prompt"] = SYSTEM_PROMPT
                frozen["task_domain"] = str(frozen.get("domain") or (registry or {}).get(code, {}).get("domain") or "")
                frozen["question"] = ("Identify the agricultural condition or organism shown in this image."
                                      if frozen["question_type"] == "open" else "Select the best identification for this agricultural image.")
                frozen["private"] = {"truth_code": code, "arm": arm,
                                     "classifier_kind": frozen["classifier"]["kind"]}
                if frozen["question_type"] == "option":
                    if registry is None or code not in registry: raise ValueError("Option source requires frozen registry names")
                    domain = frozen["task_domain"]
                    distractors = [item_code for item_code, item in sorted(registry.items())
                                   if item_code != code and item.get("domain") == domain]
                    if len(distractors) < 3: raise ValueError(f"insufficient same-domain distractors for {code}")
                    choices = [code, *distractors[:3]]
                    choices.sort(key=lambda value: hashlib.sha256(f"{frozen['sample_id']}:{value}".encode()).hexdigest())
                    frozen["public_options"] = [{"label": "ABCD"[index], "name": registry[item_code]["name"]}
                                                for index, item_code in enumerate(choices)]
                    frozen["private"]["correct_option"] = "ABCD"[choices.index(code)]
                for key in IDENTITIES: identities[key].add(frozen[key])
                chosen.append(frozen)
    report = validate_source_rows(chosen)
    if not report["ready"]: raise ValueError("candidate source failed its own contract")
    return chosen


def _identity_value(row: dict[str, Any], key: str) -> str:
    """Resolve legacy E3 identities without mutating its historical manifests."""
    value = row.get(key)
    if isinstance(value, str) and value:
        return value
    if key == "source_group_id" and isinstance(row.get("image_path"), str):
        return "source:" + hashlib.sha256(row["image_path"].encode()).hexdigest()
    if key == "near_duplicate_group_id" and isinstance(row.get("phash"), str) and row["phash"]:
        return "phash:" + hashlib.sha256(row["phash"].encode()).hexdigest()
    raise ValueError(f"cannot derive E3.5 {key}")


def prepare_scoring_scopes(*, oof_rows: list[dict[str, Any]], e3_holdouts: dict[int, list[dict[str, Any]]],
                           image_metadata: dict[str, dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Freeze two disjoint 535-image scopes before class-fold inference.

    The Known side already has OOF cards. The simulated-Unknown side intentionally
    contains only selected local scoring inputs; it cannot become a teacher source
    until a class-fold Top-5 card is joined later.
    """
    selected: dict[str, list[dict[str, Any]]] = {"known": [], "simulated_unknown": []}
    used = {key: set() for key in IDENTITIES}
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in oof_rows:
        code = row.get("canonical_class_code")
        if isinstance(code, str): by_class.setdefault(code, []).append(row)
    if len(by_class) != 107: raise ValueError("OOF pool does not cover 107 Known classes")
    for code in sorted(by_class):
        for row in sorted(by_class[code], key=lambda item: str(item.get("image_sha256"))):
            identities = {key: _identity_value(row, key) for key in IDENTITIES}
            if all(value not in used[key] for key, value in identities.items()):
                selected["known"].append({**row, **identities})
                for key, value in identities.items(): used[key].add(value)
            if sum(item.get("canonical_class_code") == code for item in selected["known"]) == 5: break
        if sum(item.get("canonical_class_code") == code for item in selected["known"]) != 5:
            raise ValueError(f"OOF pool lacks five isolated rows for {code}")
    by_unknown: dict[str, list[dict[str, Any]]] = {}
    for fold, rows in e3_holdouts.items():
        if fold not in {0, 1, 2}: raise ValueError("E3 fold must be 0, 1 or 2")
        for row in rows:
            code = row.get("canonical_class_code")
            metadata = (image_metadata or {}).get(str(row.get("image_sha256")))
            if metadata is None:
                raise ValueError("E3 holdout row lacks frozen image identity metadata")
            enriched = {**metadata, **row}
            for key in ("source_group_id", "near_duplicate_group_id", "phash"):
                if metadata.get(key) is not None: enriched[key] = metadata[key]
            if isinstance(code, str): by_unknown.setdefault(code, []).append({**enriched, "e3_fold": fold})
    if len(by_unknown) != 107: raise ValueError("E3 holdouts do not cover 107 Known classes")
    for code in sorted(by_unknown):
        for row in sorted(by_unknown[code], key=lambda item: str(item.get("image_sha256"))):
            identities = {key: _identity_value(row, key) for key in IDENTITIES}
            if all(value not in used[key] for key, value in identities.items()):
                selected["simulated_unknown"].append({**row, **identities})
                for key, value in identities.items(): used[key].add(value)
            if sum(item.get("canonical_class_code") == code for item in selected["simulated_unknown"]) == 5: break
        if sum(item.get("canonical_class_code") == code for item in selected["simulated_unknown"]) != 5:
            raise ValueError(f"E3 holdout pool lacks five cross-arm-isolated rows for {code}")
    return selected


def build_card_pools(*, known_scope: list[dict[str, Any]], unknown_scope: list[dict[str, Any]],
                     e3_predictions: dict[int, list[dict[str, Any]]], label_maps: dict[int, list[dict[str, Any]]],
                     checkpoint_paths: dict[int, Path], training_manifests: dict[int, Path], registry_sha256: str) -> dict[str, list[dict[str, Any]]]:
    """Join frozen local predictions to scopes with classifier provenance."""
    known = []
    for row in known_scope:
        p = row.get("prediction")
        if not isinstance(p, dict): raise ValueError("Known scope lacks frozen OOF prediction")
        known.append({**row, "classifier": {"kind": "oof", "held_out_fold": p.get("held_out_fold"),
            "checkpoint_sha256": p.get("checkpoint_sha256"), "training_manifest_sha256": p.get("training_manifest_sha256"),
            "label_map_sha256": p.get("label_map_sha256"), "registry_sha256": p.get("registry_sha256"), "top5": p.get("top5")}})
    predicted = {fold: {str(item.get("image_sha256")): item for item in rows} for fold, rows in e3_predictions.items()}
    provenance = {}
    for fold in range(3):
        labels, checkpoint, manifest = label_maps.get(fold), checkpoint_paths.get(fold), training_manifests.get(fold)
        if not isinstance(labels, list) or checkpoint is None or manifest is None:
            raise ValueError("E3 classifier provenance is incomplete")
        provenance[fold] = {
            "label_codes": [str(item.get("canonical_class_code")) for item in labels],
            "names": {str(item.get("canonical_class_code")): str(item.get("canonical_english_name") or item.get("english_name") or "") for item in labels},
            "checkpoint_sha256": sha256(checkpoint), "training_manifest_sha256": sha256(manifest),
            "label_map_sha256": sha256(checkpoint.parent / "label_map.json"),
        }
    unknown = []
    for row in unknown_scope:
        fold = row.get("e3_fold")
        p = predicted.get(fold, {}).get(str(row.get("image_sha256")))
        card = provenance.get(fold)
        if p is None or card is None: raise ValueError("E3 classifier provenance is incomplete")
        label_codes = card["label_codes"]
        if row["canonical_class_code"] in label_codes: raise ValueError("simulated Unknown truth appears in E3 classifier label map")
        top5 = [{"code": item.get("canonical_class_code"), "name": card["names"].get(str(item.get("canonical_class_code")), ""), "score": item.get("confidence")} for item in p.get("topk", [])]
        if len(top5) != 5 or any(not item["name"] for item in top5): raise ValueError("E3 score Top-5 cannot map to public registry names")
        unknown.append({**row, "classifier": {"kind": "classfold", "held_out_fold": fold, "label_map_codes": label_codes,
            "checkpoint_sha256": card["checkpoint_sha256"], "training_manifest_sha256": card["training_manifest_sha256"],
            "label_map_sha256": card["label_map_sha256"], "registry_sha256": registry_sha256, "top5": top5}})
    return {"known": known, "simulated_unknown": unknown}


def initial_manifest(*, campaign_id: str, source_path: Path, rows: list[dict[str, Any]],
                     output: Path, audit_only: bool) -> dict[str, Any]:
    """Write the immutable R0 plan; collection is deliberately separate."""
    if output.exists():
        raise ValueError("immutable E3.5 manifest destination already exists")
    report = validate_source_rows(rows, require_full_coverage=not audit_only)
    if not report["ready"]:
        raise ValueError("E3.5 source failed contract")
    expected = 32 if audit_only else 1070
    if len(rows) != expected:
        raise ValueError(f"E3.5 {'audit' if audit_only else 'campaign'} must contain {expected} rows")
    work = [{"work_id": f"R0:{row['sample_id']}:cascade", "round": "R0",
             "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
             "attempt_ordinal": 0, "predecessor_request_id": None,
             "route_progression": list(ROUTES)} for row in rows]
    payload = {
        "schema_version": "agrinet.e35-cascade-manifest/v1", "campaign_id": campaign_id,
        "round": "R0", "source": str(source_path), "source_sha256": sha256(source_path),
        "source_rows_expected": expected, "audit_only": audit_only, "work_items": work,
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload


def freeze_round_summary(*, manifest: dict[str, Any], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Close one round without converting a provider failure into quality data."""
    work = {item["work_id"]: item for item in manifest.get("work_items", []) if isinstance(item, dict)}
    indexed = {item.get("work_id"): item for item in outcomes if isinstance(item, dict)}
    if set(work) != set(indexed) or len(indexed) != len(outcomes):
        raise ValueError("outcomes must contain exactly one record for every immutable work item")
    rows = []
    for work_id, item in work.items():
        outcome = indexed[work_id]
        status = outcome.get("delivery_status")
        if status not in {"delivered", BUDGET_SHORTFALL, *DELIVERY_FAILURES}:
            raise ValueError("unrecognized E3.5 delivery status")
        if status in DELIVERY_FAILURES and not outcome.get("request_id"):
            raise ValueError("unresolved delivery must retain its request ID")
        rows.append({**item, "delivery_status": status, "request_id": outcome.get("request_id"),
                     "winner": bool(outcome.get("winner")) if status == "delivered" else False,
                     "final_route": outcome.get("final_route"),
                     "quality_status": outcome.get("quality_status"),
                     "parent_path": outcome.get("parent_path"),
                     "contract_error": outcome.get("contract_error")})
    return {"schema_version": "agrinet.e35-cascade-summary/v1", "campaign_id": manifest.get("campaign_id"),
            "round": manifest.get("round"), "rows": rows, "automatic_replay_allowed": False,
            **({"collection_controls": manifest["collection_controls"]} if manifest.get("collection_controls") else {}),
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}


def replenishment_manifest(*, summary: dict[str, Any], next_round: str) -> dict[str, Any]:
    if next_round not in {"R1", "R2"} or summary.get("round") != {"R1": "R0", "R2": "R1"}[next_round]:
        raise ValueError("E3.5 recovery must proceed R0 -> R1 -> R2")
    work = []
    for row in summary.get("rows", []):
        # Quality-terminal and accepted items are closed permanently. Only a
        # provider delivery ambiguity can create a new R1/R2 attempt.
        if row.get("delivery_status") not in DELIVERY_FAILURES:
            continue
        recovery = recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal", -1)),
                                    status=str(row.get("delivery_status")),
                                    predecessor_request_id=str(row.get("request_id") or ""))
        if recovery["new_attempt"]:
            work.append({**{key: row[key] for key in ("sample_id", "image_group_id", "route_progression")},
                         **({"resume_route": row["resume_route"]} if row.get("resume_route") in ROUTES else {}),
                         "work_id": f"{next_round}:{row['sample_id']}:cascade", "round": next_round, **recovery})
    controls = summary.get("collection_controls")
    return {"schema_version": "agrinet.e35-cascade-manifest/v2" if controls else "agrinet.e35-cascade-manifest/v1", "campaign_id": summary.get("campaign_id"),
            "round": next_round, "work_items": work, "workers": 4, "micu_intent_limit": 8000,
            "automatic_replay_allowed": False, **({"collection_controls": controls} if controls else {}), "training_eligible": False,
            "training_authorized": False, "sft_may_start": False}


def delivery_reauthorization_manifest(*, source_rows: list[dict[str, Any]], prior_summary: dict[str, Any],
                                     campaign_id: str, source_output: Path,
                                     manifest_output: Path) -> dict[str, Any]:
    """Freeze an explicitly authorized fresh attempt for exhausted delivery gaps.

    It never edits or resumes an R2 ledger.  Every selected item receives a new
    R0 request identity under a new campaign root, while preserving the final
    exhausted request ID as non-operative predecessor provenance.
    """
    if prior_summary.get("round") != "R2":
        raise ValueError("delivery reauthorization requires an exhausted R2 summary")
    source_by_id = {row.get("sample_id"): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("delivery reauthorization source has duplicate IDs")
    terminal = [row for row in prior_summary.get("rows", []) if row.get("delivery_status") in DELIVERY_FAILURES]
    if not terminal or len(terminal) > 32:
        raise ValueError("delivery reauthorization requires 1..32 exhausted delivery gaps")
    if any(row.get("sample_id") not in source_by_id or not row.get("request_id") for row in terminal):
        raise ValueError("delivery reauthorization lacks prior source/request binding")
    chosen = [source_by_id[row["sample_id"]] for row in terminal]
    validation = validate_source_rows(chosen, require_full_coverage=False)
    if not validation["ready"]:
        raise ValueError("delivery reauthorization source violates E3.5 contract")
    if source_output.exists() or manifest_output.exists():
        raise ValueError("delivery reauthorization destinations are immutable")
    source_output.parent.mkdir(parents=True, exist_ok=True)
    source_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen), encoding="utf-8")
    prior_by_id = {row["sample_id"]: row for row in terminal}
    work = [{"work_id": f"R0:{row['sample_id']}:cascade", "round": "R0",
             "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
             "attempt_ordinal": 0, "predecessor_request_id": prior_by_id[row["sample_id"]]["request_id"],
             "prior_campaign_id": prior_summary.get("campaign_id"),
             "prior_round": "R2", "route_progression": list(ROUTES)} for row in chosen]
    payload = {"schema_version": "agrinet.e35-cascade-manifest/v1", "campaign_id": campaign_id,
               "round": "R0", "source": str(source_output), "source_sha256": sha256(source_output),
               "source_rows_expected": len(chosen), "audit_only": True,
               "delivery_reauthorization": True,
               "work_items": work, "workers": 4, "micu_intent_limit": 8000,
               "automatic_replay_allowed": False, "training_eligible": False,
               "training_authorized": False, "sft_may_start": False}
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload


def validate_source_rows(rows: list[dict[str, Any]], *, audit_ids: set[str] | None = None,
                         require_full_coverage: bool = True) -> dict[str, Any]:
    """Validate frozen full-pool assignment and three-key arm isolation."""
    audit_ids = audit_ids or set()
    seen = {key: set() for key in IDENTITIES}
    per_arm_class: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    per_arm_kind: dict[str, Counter[tuple[str, str]]] = {arm: Counter() for arm in ARMS}
    errors: list[str] = []
    for row in rows:
        arm, code = row.get("arm"), row.get("canonical_class_code")
        if arm not in ARMS or not isinstance(code, str):
            errors.append("invalid arm or canonical class"); continue
        for key in IDENTITIES:
            value = row.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"missing {key}")
            elif value in seen[key]:
                errors.append(f"identity overlap: {key}")
            else: seen[key].add(value)
        kind = row.get("question_type")
        if kind not in {"open", "option"}: errors.append("invalid question type")
        else: per_arm_kind[arm][(code, kind)] += 1
        per_arm_class[arm][code] += 1
        classifier = row.get("classifier", {})
        if not isinstance(classifier, dict) or any(not isinstance(classifier.get(key), str) or not classifier[key]
                                                  for key in ("checkpoint_sha256", "training_manifest_sha256",
                                                              "label_map_sha256", "registry_sha256")):
            errors.append("classifier card lacks frozen provenance hashes")
        top5 = classifier.get("top5") if isinstance(classifier, dict) else None
        if (not isinstance(top5, list) or len(top5) != 5 or
                any(not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                    not isinstance(item.get("score"), (float, int)) for item in top5)):
            errors.append("classifier card lacks public frozen Top-5")
        if arm == "known" and classifier.get("kind") != "oof": errors.append("Known row lacks OOF card")
        if arm == "known" and classifier.get("held_out_fold") not in {0, 1, 2}:
            errors.append("Known row lacks valid OOF fold")
        if arm == "simulated_unknown":
            if (classifier.get("kind") != "classfold" or
                    not isinstance(classifier.get("label_map_codes"), list) or
                    code in classifier.get("label_map_codes", [])):
                errors.append("simulated Unknown truth is visible to classifier")
        if row.get("sample_id") in audit_ids and row.get("phase") not in {"audit", "campaign"}:
            errors.append("audit membership is malformed")
    for arm in ARMS:
        if require_full_coverage and len(per_arm_class[arm]) != 107: errors.append(f"{arm} does not cover 107 classes")
        for code, count in per_arm_class[arm].items():
            if (require_full_coverage and
                    (count != 5 or per_arm_kind[arm][code, "open"] != 3 or per_arm_kind[arm][code, "option"] != 2)):
                errors.append(f"{arm}/{code} is not 3 Open + 2 Option")
    return {"ready": not errors, "errors": sorted(set(errors)),
            "rows": len(rows), "per_arm": {arm: sum(per_arm_class[arm].values()) for arm in ARMS}}


def next_route(*, current: str, parent_delivery: str, private_audit: str, route_contract_ok: bool) -> str | None:
    """Return the only legal same-attempt escalation; delivery never escalates."""
    if parent_delivery in DELIVERY_FAILURES:
        return None
    if current not in ROUTES or parent_delivery != "delivered": return None
    # A route-contract failure is recorded for audit, but never lets the
    # controller bypass the independent private decision.
    if private_audit != "reject": return None
    return {"direct": "classifier", "classifier": "rag", "rag": None}[current]


def recovery_attempt(*, prior_attempt_ordinal: int, status: str, predecessor_request_id: str) -> dict[str, Any]:
    if status not in DELIVERY_FAILURES: raise ValueError("only unresolved delivery can recover")
    if prior_attempt_ordinal not in {0, 1}:
        return {"state": "delivery_shortfall", "new_attempt": False}
    return {"state": "retry", "new_attempt": True,
            "attempt_ordinal": prior_attempt_ordinal + 1,
            "predecessor_request_id": predecessor_request_id}


def validate_rag_terminal(trajectory: dict[str, Any]) -> None:
    route, answer = trajectory.get("route"), trajectory.get("answer")
    if isinstance(answer, str):
        match = re.search(r"<answer>(.*?)</answer>", answer, flags=re.I | re.S)
        if match: answer = match.group(1).strip()
    calls = trajectory.get("tool_calls", [])
    if route != "rag": return
    rag_positions = [i for i, call in enumerate(calls) if call.get("name") == "agrinet_rag_search"]
    if answer == "INSUFFICIENT_EVIDENCE":
        if not rag_positions or not trajectory.get("private_rag_evidence_insufficient"):
            raise ValueError("RAG refusal requires actual RAG evidence and private audit")
        trace = trajectory.get("tool_trace")
        if (not isinstance(trace, list) or not any(isinstance(item, dict) and
                item.get("call", {}).get("name") == "agrinet_rag_search" and
                isinstance(item.get("response"), dict) for item in trace)):
            raise ValueError("RAG refusal requires a retained actual RAG response")
    names = [call.get("name") for call in calls]
    allowed = {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"}
    if any(name not in allowed for name in names):
        raise ValueError("RAG trajectory contains an unauthorized tool")
    if "agrinet_rag_search" in names and route != "rag":
        raise ValueError("RAG tool is unavailable outside RAG route")
    if trajectory.get("messages") is not None:
        normalize_training_messages(
            trajectory["messages"],
            tool_name={"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"},
            validate_arguments=lambda _: [], require_tool_calls=True)


def hermes_candidate(*, row: dict[str, Any], trajectory: dict[str, Any],
                     rewrite: dict[str, Any], rewrite_audit: str) -> dict[str, Any]:
    """Project one accepted winner to a public, image-bound Hermes row."""
    if rewrite_audit != "accept":
        raise ValueError("only a privately accepted rewrite may become an E3.5 candidate")
    route = trajectory.get("route")
    if route not in ROUTES or not isinstance(rewrite.get("reasoning"), str) or not isinstance(rewrite.get("final"), str):
        raise ValueError("candidate lacks route or rewritten conclusion")
    validate_rag_terminal({**trajectory, "route": route, "answer": rewrite["final"]})
    messages = rewrite.get("messages")
    if not isinstance(messages, list):
        raise ValueError("rewrite must supply public Hermes messages")
    allowed = {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"}
    normalized = normalize_training_messages(messages, tool_name=allowed, validate_arguments=lambda _: [],
                                             require_tool_calls=route == "rag")
    sft_system_prompt = row.get("system_prompt")
    if isinstance(sft_system_prompt, str) and sft_system_prompt.strip():
        if normalized[0].get("role") != "system":
            raise ValueError("Hermes candidate must begin with the fixed SFT system prompt")
        normalized[0] = {**normalized[0], "content": sft_system_prompt}
    if route == "direct" and any(message["role"] in {"tool_call", "tool"} for message in normalized):
        raise ValueError("Direct Hermes candidate must have no tools")
    rendered = json.dumps(normalized, ensure_ascii=False).casefold()
    if any(token in rendered for token in PRIVATE_TOKENS):
        raise ValueError("Hermes candidate leaks private source metadata")
    if rewrite["final"] == "INSUFFICIENT_EVIDENCE" and route == "rag":
        if not any(message["role"] == "tool" and "rag" in str(message["content"]).casefold()
                   for message in normalized):
            raise ValueError("RAG refusal candidate lacks retained RAG tool response")
    return {
        "schema_version": "agrinet.e35-hermes-sft-candidate/v1",
        "sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
        "route": route, "messages": normalized,
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }


def candidate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = validate_source_rows(rows)
    return {"schema_version": "agrinet.e35-candidate-report/v1", **source,
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}


def select_audit(rows: list[dict[str, Any]], *, seed: str = "e35-audit-v1", exclude_ids: set[str] | None = None,
                 rag_witnesses_per_arm: int = 0) -> list[dict[str, Any]]:
    """Deterministically select 4 per Open/Option x disease/pest cell per arm."""
    validation = validate_source_rows(rows)
    if not validation["ready"]:
        raise ValueError("cannot sample audit from invalid candidate pool")
    excluded = exclude_ids or set()
    if not isinstance(rag_witnesses_per_arm, int) or not 0 <= rag_witnesses_per_arm <= 2:
        raise ValueError("RAG audit witnesses per arm must be between zero and two")
    selected: list[dict[str, Any]] = []
    for arm in ARMS:
        for question_type in ("open", "option"):
            for domain in ("disease", "pest"):
                ranked = sorted((row for row in rows if row["sample_id"] not in excluded and row["arm"] == arm and row.get("question_type") == question_type and row.get("task_domain") == domain),
                                key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
                if len(ranked) < 4: raise ValueError(f"audit cell lacks four rows: {arm}/{question_type}/{domain}")
                # Prefer one item from each available fold before filling the cell.
                chosen, folds = [], set()
                for row in ranked:
                    fold = row["classifier"].get("held_out_fold")
                    if fold not in folds:
                        chosen.append(row); folds.add(fold)
                    if len(chosen) == 4: break
                for row in ranked:
                    if row not in chosen: chosen.append(row)
                    if len(chosen) == 4: break
                selected.extend(chosen)
    # The designation is a private audit-side coverage control.  It intentionally
    # has no effect on prompt construction, classifier cards, or public lineage.
    for arm in ARMS:
        ranked = sorted((row for row in selected if row["arm"] == arm),
                        key=lambda row: hashlib.sha256(f"{seed}:rag-witness:{row['sample_id']}".encode()).hexdigest())
        for row in ranked[:rag_witnesses_per_arm]:
            private = row.get("private")
            if not isinstance(private, dict):
                raise ValueError("audit witness lacks private source binding")
            private["audit_protocol"] = {"rag_witness": True, "purpose": "private_audit_route_coverage"}
    return selected


def campaign_report(outcomes: list[dict[str, Any]], *, source_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Public summary that keeps delivery, quality and RAG results distinct.

    ``outcomes`` is deliberately a projection: it must not contain truth or
    audit prose. The category values are fixed to prevent a caller from hiding
    an unresolved provider request inside a generic rejection bucket.
    """
    route_counts: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    rag_counts: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    shortfalls: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    per_arm_class: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    for outcome in outcomes:
        arm, code = outcome.get("arm"), outcome.get("canonical_class_code")
        if arm not in ARMS or not isinstance(code, str):
            raise ValueError("campaign outcome lacks public arm/class identity")
        delivery, route = outcome.get("delivery_status"), outcome.get("final_route")
        if route not in {*ROUTES, None}:
            raise ValueError("campaign outcome has invalid final route")
        if outcome.get("winner") is True:
            per_arm_class[arm][code] += 1
        if delivery in DELIVERY_FAILURES or outcome.get("state") == "delivery_shortfall":
            shortfalls[arm]["delivery_shortfall"] += 1
            continue
        if delivery != "delivered":
            raise ValueError("campaign outcome has invalid delivery classification")
        quality = str(outcome.get("quality_status") or "unclassified")
        if quality in {"route_contract_reject", "tool_shortfall", "quality_rejected", "provider_delivery"}:
            shortfalls[arm][quality] += 1
        elif route is not None:
            route_counts[arm][route] += 1
        if route == "rag":
            result = outcome.get("rag_result")
            if result not in {"correct_answer", "compliant_refusal", "incorrect_answer", None}:
                raise ValueError("invalid RAG terminal classification")
            if result is not None: rag_counts[arm][result] += 1
    source_codes: dict[str, set[str]] = {arm: set() for arm in ARMS}
    if source_rows is not None:
        source_validation = validate_source_rows(source_rows, require_full_coverage=False)
        if not source_validation["ready"]:
            raise ValueError("campaign report source violates E3.5 contract")
        for row in source_rows:
            source_codes[row["arm"]].add(row["canonical_class_code"])
    classes = {arm: {code: {"target": 5, "closed": per_arm_class[arm][code],
                            "shortfall": max(0, 5 - per_arm_class[arm][code])}
                     for code in sorted(source_codes[arm] or set(per_arm_class[arm]))}
               for arm in ARMS}
    return {
        "schema_version": "agrinet.e35-campaign-report/v1",
        "arms": {arm: {"final_routes": dict(route_counts[arm]), "rag_terminals": dict(rag_counts[arm]),
                        "shortfalls": dict(shortfalls[arm]), "per_class": classes[arm],
                        "closed_classes_at_target": sum(item["closed"] == 5 for item in classes[arm].values())}
                 for arm in ARMS},
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }


def audit_final_report(*, source_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Close the 32-image protocol audit without conflating delivery with quality."""
    if len(source_rows) != 32 or any(row.get("arm") not in ARMS for row in source_rows):
        raise ValueError("E3.5 audit final requires the exact 32-row dual-arm source")
    if [item.get("round") for item in summaries] != ["R0", "R1", "R2"]:
        raise ValueError("E3.5 audit final requires frozen R0/R1/R2 summaries")
    # Later recovery summaries are intentionally sparse: accepted/quality-
    # terminal rows are never replayed. Overlay each round by sample ID.
    latest: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for row in summary.get("rows", []):
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise ValueError("audit summary has malformed source row")
            latest[row["sample_id"]] = row
    if len(latest) != 32: raise ValueError("audit summaries do not close each source row")
    by_id = {row["sample_id"]: row for row in source_rows}
    if set(latest) != set(by_id): raise ValueError("audit source and terminal summary differ")
    result = {arm: Counter() for arm in ARMS}
    for sample_id, terminal in latest.items():
        arm = by_id[sample_id]["arm"]
        if terminal.get("delivery_status") in DELIVERY_FAILURES:
            result[arm]["delivery_shortfall"] += 1
        elif terminal.get("winner"):
            result[arm]["closed"] += 1
        else:
            result[arm][str(terminal.get("quality_status") or "quality_rejected")] += 1
    def has_retained_rag_evidence(terminal: dict[str, Any]) -> bool:
        if not (terminal.get("winner") and terminal.get("final_route") == "rag"):
            return False
        path = terminal.get("parent_path")
        if not isinstance(path, str) or not path:
            return False
        try:
            trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        trace = trajectory.get("tool_trace")
        return (trajectory.get("route") == "rag" and isinstance(trace, list) and any(
            isinstance(event, dict) and isinstance(event.get("call"), dict) and
            event["call"].get("name") == "agrinet_rag_search" and
            isinstance(event.get("response"), dict) for event in trace))

    rag_evidence_closed = any(has_retained_rag_evidence(row) for row in latest.values())
    protocol_observed = any(row.get("delivery_status") == "delivered" for row in latest.values())
    return {"schema_version": "agrinet.e35-audit-final-report/v1",
            "rows": 32, "rounds": ["R0", "R1", "R2"],
            "per_arm": {arm: dict(result[arm]) for arm in ARMS},
            "protocol_observed": protocol_observed, "rag_evidence_closed": rag_evidence_closed,
            "protocol_gate_passed": bool(protocol_observed and rag_evidence_closed),
            "full_campaign_expansion": "not_authorized" if not (protocol_observed and rag_evidence_closed) else "requires_separate_manifest",
            "training_eligible": False, "training_authorized": False, "sft_may_start": False}


def delivery_reauthorization_final_report(*, source_rows: list[dict[str, Any]], base_r0: dict[str, Any],
                                          reauthorization_summaries: list[dict[str, Any]],
                                          prior_reauthorization_groups: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Close a reauthorized delivery gap set against its immutable original R0.

    The base R0 establishes the full 32-image audit.  The new lineage may only
    replace its original delivery-failure rows, and its R0/R1/R2 summaries are
    overlaid before reusing the ordinary final-report validator.
    """
    if base_r0.get("round") != "R0" or [item.get("round") for item in reauthorization_summaries] != ["R0", "R1", "R2"]:
        raise ValueError("delivery reauthorization final requires base R0 plus fresh R0/R1/R2 summaries")
    base_rows = base_r0.get("rows", [])
    source_ids = {row.get("sample_id") for row in source_rows}
    base_ids = {row.get("sample_id") for row in base_rows}
    if len(base_rows) != 32 or base_ids != source_ids or None in base_ids:
        raise ValueError("delivery reauthorization base R0 does not bind the full audit source")
    gap_ids = {row.get("sample_id") for row in base_rows if row.get("delivery_status") in DELIVERY_FAILURES}
    if not gap_ids:
        raise ValueError("delivery reauthorization base R0 has no delivery gaps")
    latest = {row["sample_id"]: row for row in base_rows}
    groups = list(prior_reauthorization_groups or []) + [reauthorization_summaries]
    for index, group in enumerate(groups):
        if [item.get("round") for item in group] != ["R0", "R1", "R2"]:
            raise ValueError("delivery reauthorization lineage must contain R0/R1/R2")
        expected = {sample_id for sample_id in gap_ids
                    if latest[sample_id].get("delivery_status") in DELIVERY_FAILURES}
        observed = {row.get("sample_id") for row in group[0].get("rows", [])}
        if observed != expected:
            raise ValueError("delivery reauthorization R0 does not exactly cover current delivery gaps")
        for summary in group:
            for row in summary.get("rows", []):
                sample_id = row.get("sample_id")
                if sample_id not in expected:
                    raise ValueError("delivery reauthorization attempted a non-gap sample")
                latest[sample_id] = row
    synthesized = [{"round": "R0", "rows": base_rows},
                   {"round": "R1", "rows": [latest[key] for key in sorted(gap_ids)]},
                   {"round": "R2", "rows": []}]
    report = audit_final_report(source_rows=source_rows, summaries=synthesized)
    return {**report, "delivery_reauthorization": True,
            "base_campaign_id": base_r0.get("campaign_id"),
            "reauthorization_campaign_id": reauthorization_summaries[0].get("campaign_id"),
            "prior_reauthorization_campaign_ids": [group[0].get("campaign_id") for group in (prior_reauthorization_groups or [])]}


def full_campaign_manifest_after_audit(*, audit_report: dict[str, Any], campaign_id: str,
                                      source_path: Path, rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Freeze full collection only after an evidence-bearing audit gate passes."""
    if audit_report.get("schema_version") != "agrinet.e35-audit-final-report/v1":
        raise ValueError("full campaign requires an E3.5 audit final report")
    if audit_report.get("protocol_gate_passed") is not True or audit_report.get("rag_evidence_closed") is not True:
        raise ValueError("full campaign is not authorized until the audit protocol gate passes")
    return initial_manifest(campaign_id=campaign_id, source_path=source_path, rows=rows, output=output, audit_only=False)


V9_COLLECTION_CONTROLS = {
    "uncached_input_token_cap": 200000,
    "transport_image_max_side": 1024,
    "max_public_turns_per_route": 2,
    "reservation_uncached_tokens": {
        "generation": {"classifier": 2742, "rag": 2775},
        "private_audit": 13542,
    },
}


def v9_continuation_manifest(*, source_rows: list[dict[str, Any]], continuation: list[dict[str, Any]],
                             source_path: Path, continuation_path: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    """Freeze the only authorized budget-controlled v9 continuation scope."""
    if output.exists():
        raise ValueError("E3.5 v9 continuation manifest is immutable")
    by_id = {str(row.get("sample_id") or ""): row for row in source_rows}
    if len(by_id) != len(source_rows) or "" in by_id:
        raise ValueError("E3.5 v9 source has duplicate sample IDs")
    if len(continuation) not in {4, 6, 10}:
        raise ValueError("E3.5 continuation must contain four, six, or the original ten repair samples")
    seen, routes = set(), Counter()
    work = []
    for item in continuation:
        sample_id, route, predecessor = str(item.get("sample_id") or ""), str(item.get("resume_route") or ""), str(item.get("predecessor_request_id") or "")
        if sample_id not in by_id or sample_id in seen or route not in {"classifier", "rag"} or not predecessor:
            raise ValueError("E3.5 v9 continuation sidecar is malformed")
        state = str(item.get("prior_state") or "")
        if route == "classifier" and state != "direct_confirmed_private_reject":
            raise ValueError("E3.5 v9 classifier continuation lacks confirmed Direct rejection")
        if route == "rag" and state not in {"rag_unknown_delivery", "rag_route_contract_reject", "rag_local_tool_shortfall", "rag_budget_reservation_overrun"}:
            raise ValueError("E3.5 v9 RAG continuation has invalid prior state")
        seen.add(sample_id); routes[route] += 1
        row = by_id[sample_id]
        work.append({"work_id": f"R0:{sample_id}:continuation", "round": "R0", "sample_id": sample_id,
                     "image_group_id": row.get("image_group_id", row["image_sha256"]), "attempt_ordinal": 0,
                     "predecessor_request_id": predecessor, "resume_route": route, "prior_state": state,
                     "route_progression": list(ROUTES[ROUTES.index(route):])})
    if routes not in (Counter({"classifier": 4, "rag": 6}), Counter({"rag": 6}), Counter({"rag": 4})):
        raise ValueError("E3.5 continuation must be 4 Classifier plus 6 RAG, or four/six RAG-only repair samples")
    controls = V9_COLLECTION_CONTROLS if len(continuation) == 10 else {**V9_COLLECTION_CONTROLS,
        "reservation_uncached_tokens": {**V9_COLLECTION_CONTROLS["reservation_uncached_tokens"], "private_audit": 20000}}
    payload = {"schema_version": "agrinet.e35-cascade-manifest/v2", "campaign_id": campaign_id,
               "round": "R0", "audit_only": True, "continuation": True,
               "source": str(source_path), "source_sha256": sha256(source_path),
               "continuation_sidecar": str(continuation_path), "continuation_sidecar_sha256": sha256(continuation_path),
               "source_rows_expected": len(work), "work_items": sorted(work, key=lambda row: row["sample_id"]),
               "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
               "collection_controls": controls,
               "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--candidate-source", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build-source")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--known-pool", type=Path, required=True)
    build.add_argument("--simulated-unknown-pool", type=Path, required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--candidate-output", type=Path, required=True)
    build.add_argument("--audit-output", type=Path, required=True)
    prepare = sub.add_parser("prepare-scoring")
    prepare.add_argument("--oof-predictions", type=Path, required=True)
    prepare.add_argument("--e3-holdout", action="append", type=Path, required=True)
    prepare.add_argument("--images-manifest", type=Path, required=True)
    prepare.add_argument("--known-output", type=Path, required=True)
    prepare.add_argument("--simulated-unknown-output", type=Path, required=True)
    prepare.add_argument("--fold-manifest-dir", type=Path, required=True)
    materialize = sub.add_parser("materialize-fold-manifests")
    materialize.add_argument("--simulated-unknown-scope", type=Path, required=True)
    materialize.add_argument("--fold-manifest-dir", type=Path, required=True)
    audit_select = sub.add_parser("select-audit")
    audit_select.add_argument("--candidate-source", type=Path, required=True)
    audit_select.add_argument("--output", type=Path, required=True)
    audit_select.add_argument("--exclude-source", action="append", type=Path, default=[])
    audit_select.add_argument("--rag-witnesses-per-arm", type=int, default=0)
    cards = sub.add_parser("build-card-pools")
    cards.add_argument("--known-scope", type=Path, required=True)
    cards.add_argument("--simulated-unknown-scope", type=Path, required=True)
    cards.add_argument("--e3-prediction", action="append", type=Path, required=True)
    cards.add_argument("--e3-label-map", action="append", type=Path, required=True)
    cards.add_argument("--e3-checkpoint", action="append", type=Path, required=True)
    cards.add_argument("--e3-training-manifest", action="append", type=Path, required=True)
    cards.add_argument("--registry-sha256", required=True)
    cards.add_argument("--known-output", type=Path, required=True)
    cards.add_argument("--simulated-unknown-output", type=Path, required=True)
    plan = sub.add_parser("plan-r0")
    plan.add_argument("--candidate-source", type=Path, required=True)
    plan.add_argument("--audit", action="store_true")
    plan.add_argument("--source-is-audit", action="store_true")
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze-summary")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--outcomes", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    replenish = sub.add_parser("plan-replenishment")
    replenish.add_argument("--summary", type=Path, required=True)
    replenish.add_argument("--next-round", choices=("R1", "R2"), required=True)
    replenish.add_argument("--output", type=Path, required=True)
    audit_final = sub.add_parser("audit-final-report")
    audit_final.add_argument("--source", type=Path, required=True)
    audit_final.add_argument("--r0-summary", type=Path, required=True)
    audit_final.add_argument("--r1-summary", type=Path, required=True)
    audit_final.add_argument("--r2-summary", type=Path, required=True)
    audit_final.add_argument("--output", type=Path, required=True)
    full_plan = sub.add_parser("plan-full-after-audit")
    full_plan.add_argument("--audit-report", type=Path, required=True)
    full_plan.add_argument("--candidate-source", type=Path, required=True)
    full_plan.add_argument("--campaign-id", required=True)
    full_plan.add_argument("--output", type=Path, required=True)
    reauthorize = sub.add_parser("reauthorize-delivery")
    reauthorize.add_argument("--candidate-source", type=Path, required=True)
    reauthorize.add_argument("--prior-summary", type=Path, required=True)
    reauthorize.add_argument("--campaign-id", required=True)
    reauthorize.add_argument("--source-output", type=Path, required=True)
    reauthorize.add_argument("--output", type=Path, required=True)
    reauthorize_final = sub.add_parser("delivery-reauthorization-final-report")
    reauthorize_final.add_argument("--source", type=Path, required=True)
    reauthorize_final.add_argument("--base-r0-summary", type=Path, required=True)
    reauthorize_final.add_argument("--r0-summary", type=Path, required=True)
    reauthorize_final.add_argument("--r1-summary", type=Path, required=True)
    reauthorize_final.add_argument("--r2-summary", type=Path, required=True)
    reauthorize_final.add_argument("--prior-reauthorization-summary", type=Path, action="append", default=[])
    reauthorize_final.add_argument("--output", type=Path, required=True)
    v9 = sub.add_parser("plan-v9-continuation")
    v9.add_argument("--source", type=Path, required=True)
    v9.add_argument("--continuation", type=Path, required=True)
    v9.add_argument("--campaign-id", required=True)
    v9.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def read_rows(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.operation == "plan-v9-continuation":
        continuation = json.loads(args.continuation.read_text(encoding="utf-8"))
        if not isinstance(continuation, list):
            raise ValueError("E3.5 v9 continuation sidecar must be a JSON list")
        manifest = v9_continuation_manifest(source_rows=read_rows(args.source), continuation=continuation,
                                            source_path=args.source, continuation_path=args.continuation,
                                            campaign_id=args.campaign_id, output=args.output)
        print(json.dumps({"round": manifest["round"], "rows": len(manifest["work_items"]),
                          "uncached_input_token_cap": V9_COLLECTION_CONTROLS["uncached_input_token_cap"], "output": str(args.output)})); return 0
    if args.operation == "reauthorize-delivery":
        manifest = delivery_reauthorization_manifest(source_rows=read_rows(args.candidate_source),
            prior_summary=json.loads(args.prior_summary.read_text(encoding="utf-8")), campaign_id=args.campaign_id,
            source_output=args.source_output, manifest_output=args.output)
        print(json.dumps({"round": manifest["round"], "rows": manifest["source_rows_expected"],
                          "delivery_reauthorization": True, "output": str(args.output)})); return 0
    if args.operation == "delivery-reauthorization-final-report":
        if args.output.exists(): raise ValueError("E3.5 delivery reauthorization report destination is immutable")
        if len(args.prior_reauthorization_summary) % 3:
            raise ValueError("prior reauthorization summaries must be supplied as complete R0/R1/R2 groups")
        report = delivery_reauthorization_final_report(source_rows=read_rows(args.source),
            base_r0=json.loads(args.base_r0_summary.read_text(encoding="utf-8")),
            reauthorization_summaries=[json.loads(path.read_text(encoding="utf-8"))
                for path in (args.r0_summary, args.r1_summary, args.r2_summary)],
            prior_reauthorization_groups=[
                [json.loads(path.read_text(encoding="utf-8")) for path in args.prior_reauthorization_summary[index:index + 3]]
                for index in range(0, len(args.prior_reauthorization_summary), 3)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"protocol_gate_passed": report["protocol_gate_passed"],
                          "delivery_reauthorization": True, "output": str(args.output)})); return 0 if report["protocol_gate_passed"] else 2
    if args.operation == "freeze-summary":
        if args.output.exists(): raise ValueError("E3.5 summary destination is immutable")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = json.loads(args.outcomes.read_text(encoding="utf-8"))
        summary = freeze_round_summary(manifest=manifest, outcomes=payload.get("outcomes", []))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream: json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"round": summary["round"], "rows": len(summary["rows"])})); return 0
    if args.operation == "plan-replenishment":
        if args.output.exists(): raise ValueError("E3.5 replenishment destination is immutable")
        manifest = replenishment_manifest(summary=json.loads(args.summary.read_text(encoding="utf-8")), next_round=args.next_round)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream: json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"round": manifest["round"], "rows": len(manifest["work_items"])})); return 0
    if args.operation == "audit-final-report":
        if args.output.exists(): raise ValueError("E3.5 audit final report destination is immutable")
        report = audit_final_report(source_rows=read_rows(args.source), summaries=[json.loads(path.read_text(encoding="utf-8")) for path in (args.r0_summary, args.r1_summary, args.r2_summary)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream: json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"protocol_gate_passed": report["protocol_gate_passed"], "output": str(args.output)})); return 0 if report["protocol_gate_passed"] else 2
    if args.operation == "plan-full-after-audit":
        report = json.loads(args.audit_report.read_text(encoding="utf-8"))
        manifest = full_campaign_manifest_after_audit(audit_report=report, campaign_id=args.campaign_id,
            source_path=args.candidate_source, rows=read_rows(args.candidate_source), output=args.output)
        print(json.dumps({"round": manifest["round"], "rows": manifest["source_rows_expected"], "output": str(args.output)})); return 0
    if args.operation == "build-source":
        contract = yaml.safe_load(args.contract.read_text(encoding="utf-8")); validate_contract(contract)
        registry_rows = read_rows(args.registry)
        registry = {str(row.get("canonical_code") or row.get("canonical_class_code")): {
            "name": str(row.get("canonical_english_name") or row.get("english_name") or ""),
            "domain": str(row.get("domain") or "")} for row in registry_rows}
        if len(registry) != 211 or any(not item["name"] for item in registry.values()):
            raise ValueError("E3.5 source requires the full frozen public registry")
        rows = build_candidate_source({"known": read_rows(args.known_pool),
                                       "simulated_unknown": read_rows(args.simulated_unknown_pool)}, registry=registry)
        if args.candidate_output.exists() or args.audit_output.exists():
            raise ValueError("E3.5 source outputs are immutable and already exist")
        audit = select_audit(rows)
        for path, payload in ((args.candidate_output, rows), (args.audit_output, audit)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
        print(json.dumps({"candidate_rows": len(rows), "audit_rows": len(audit),
                          "candidate_output": str(args.candidate_output)})); return 0
    if args.operation == "prepare-scoring":
        if len(args.e3_holdout) != 3:
            raise ValueError("prepare-scoring requires exactly three E3 holdout manifests in fold order")
        metadata_rows = read_rows(args.images_manifest)
        metadata = {str(row.get("image_sha256")): row for row in metadata_rows}
        if len(metadata) != len(metadata_rows): raise ValueError("images manifest has duplicate SHA identities")
        scopes = prepare_scoring_scopes(oof_rows=read_rows(args.oof_predictions),
                                        e3_holdouts={fold: read_rows(path) for fold, path in enumerate(args.e3_holdout)},
                                        image_metadata=metadata)
        for path, payload in ((args.known_output, scopes["known"]),
                              (args.simulated_unknown_output, scopes["simulated_unknown"])):
            if path.exists(): raise ValueError("E3.5 scoring scopes are immutable and already exist")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
        if args.fold_manifest_dir.exists(): raise ValueError("E3.5 fold scoring manifest directory already exists")
        args.fold_manifest_dir.mkdir(parents=True)
        for fold in range(3):
            # ``label`` is a loader-required sentinel. score-only prediction
            # does not consume it for metrics or candidate selection.
            fold_rows = [{**row, "label": 0} for row in scopes["simulated_unknown"] if row["e3_fold"] == fold]
            if not fold_rows: raise ValueError(f"E3.5 fold {fold} scoring scope is empty")
            (args.fold_manifest_dir / f"fold-{fold}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in fold_rows), encoding="utf-8")
        print(json.dumps({"known_rows": len(scopes["known"]), "simulated_unknown_rows": len(scopes["simulated_unknown"])})); return 0
    if args.operation == "materialize-fold-manifests":
        scope = read_rows(args.simulated_unknown_scope)
        if len(scope) != 535 or any(row.get("e3_fold") not in {0, 1, 2} for row in scope):
            raise ValueError("simulated Unknown scope is not a frozen 535-row E3 scoring scope")
        if args.fold_manifest_dir.exists(): raise ValueError("E3.5 fold scoring manifest directory already exists")
        args.fold_manifest_dir.mkdir(parents=True)
        counts = {}
        for fold in range(3):
            rows = [{**row, "label": 0} for row in scope if row["e3_fold"] == fold]
            counts[fold] = len(rows)
            (args.fold_manifest_dir / f"fold-{fold}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"fold_rows": counts})); return 0
    if args.operation == "select-audit":
        source = read_rows(args.candidate_source)
        excluded_rows = [row for path in args.exclude_source for row in read_rows(path)]
        excluded_ids = {str(row.get("sample_id") or "") for row in excluded_rows}
        if len(excluded_ids) != len(excluded_rows) or "" in excluded_ids: raise ValueError("audit exclusion source has duplicate or missing sample ID")
        seed = "e35-audit-v3" if excluded_ids else "e35-audit-v1"
        audit = select_audit(source, seed=seed, exclude_ids=excluded_ids, rag_witnesses_per_arm=args.rag_witnesses_per_arm)
        serialized = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            # A transport/process failure can occur after the immutable source
            # sidecar reaches disk but before its paired report is written.
            # Continue only when rerunning deterministically reproduces it byte
            # for byte; a different selection can never overwrite the artifact.
            if args.output.read_text(encoding="utf-8") != serialized:
                raise ValueError("balanced E3.5 audit output already exists with different content")
        else:
            args.output.write_text(serialized, encoding="utf-8")
        report = {"schema_version": "agrinet.e35-balanced-audit-report/v2",
                  "selector_seed": seed, "candidate_source_sha256": sha256(args.candidate_source),
                  "audit_source_sha256": sha256(args.output), "rows": len(audit),
                  "excluded_source_sha256": [sha256(path) for path in args.exclude_source],
                  "cells": {arm: {f"{question_type}/{domain}": sum(1 for row in audit if row["arm"] == arm and row["question_type"] == question_type and row["task_domain"] == domain)
                                    for question_type in ("open", "option") for domain in ("disease", "pest")} for arm in ARMS},
                  "folds": {arm: dict(Counter(str(row["classifier"].get("held_out_fold")) for row in audit if row["arm"] == arm)) for arm in ARMS},
                  "private_rag_witnesses_per_arm": args.rag_witnesses_per_arm,
                  "training_eligible": False, "training_authorized": False, "sft_may_start": False}
        report_path = args.output.with_suffix(".report.json")
        with report_path.open("x", encoding="utf-8") as stream: json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
        print(json.dumps({"audit_rows": len(audit), "report": str(report_path)})); return 0
    if args.operation == "build-card-pools":
        sequences = (args.e3_prediction, args.e3_label_map, args.e3_checkpoint, args.e3_training_manifest)
        if any(len(items) != 3 for items in sequences): raise ValueError("build-card-pools requires exactly three fold-aligned inputs")
        pools = build_card_pools(known_scope=read_rows(args.known_scope), unknown_scope=read_rows(args.simulated_unknown_scope),
            e3_predictions={i: read_rows(path) for i, path in enumerate(args.e3_prediction)},
            label_maps={i: json.loads(path.read_text(encoding="utf-8")) for i, path in enumerate(args.e3_label_map)},
            checkpoint_paths={i: path for i, path in enumerate(args.e3_checkpoint)},
            training_manifests={i: path for i, path in enumerate(args.e3_training_manifest)}, registry_sha256=args.registry_sha256)
        for path, payload in ((args.known_output, pools["known"]), (args.simulated_unknown_output, pools["simulated_unknown"])):
            if path.exists(): raise ValueError("E3.5 card-pool outputs are immutable and already exist")
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
        print(json.dumps({"known_rows": len(pools["known"]), "simulated_unknown_rows": len(pools["simulated_unknown"])})); return 0
    rows = read_rows(args.candidate_source)
    if args.operation == "plan-r0":
        if args.source_is_audit and not args.audit:
            raise ValueError("--source-is-audit requires --audit")
        selected = rows if args.source_is_audit else select_audit(rows) if args.audit else rows
        result = initial_manifest(campaign_id=args.campaign_id, source_path=args.candidate_source, rows=selected,
                                  output=args.output, audit_only=bool(args.audit))
        print(json.dumps({"round": result["round"], "rows": result["source_rows_expected"], "output": str(args.output)})); return 0
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8")); validate_contract(contract)
    report = candidate_report(rows)
    report.update({"contract_sha256": sha256(args.contract), "candidate_source_sha256": sha256(args.candidate_source)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"ready": report["ready"], "output": str(args.output)})); return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
