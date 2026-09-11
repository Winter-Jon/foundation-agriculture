"""E2 dynamic-route smoke: source freeze, staged collection, and safe conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.classifier_ledger import DeliveryUnresolved, RequestLedger
from agrinet.rag.micu_classifier_hcv_v2 import _read_jsonl, sha256
from agrinet.rag.micu_classifier_hcv_v2_collect import (
    GlobalMicuBudget, _invoke_micu, _ledger_contract, _tool_schema, run_parent, run_private_audit,
)
from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.research.hcv.collector import image_url_content

CELLS = tuple(f"{q}-{l}-{d}" for q in ("open", "option") for l in ("en", "zh") for d in ("disease", "pest"))
ROUTES = {
    "direct": {"attempts": 3, "temperature": 0.5, "tools": set()},
    "classifier": {"attempts": 2, "temperature": 0.2, "tools": {"agrinet_classifier_predict", "agrinet_classifier_expand"}},
    "rag": {"attempts": 1, "temperature": 0.2, "tools": {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"}},
}

_REWRITE_SECTIONS = ("Visual observations:", "Candidate comparison:", "Evidence:", "Uncertainty:")
REWRITE_SCHEMA_V2 = "structured-observation-comparison-v2"
REWRITE_SCHEMA_V3 = "image-bound-compact-observation-comparison-v3"
_PRIVATE_TOKENS = (
    "private_truth", "truth_code", "sampling_arm", "target_pattern", "primary_pattern",
    "request_id", "simulated_unknown", "excluded_supervised_codes", "p6_group",
    "checkpoint_sha256", "label_map_sha256", "training_manifest_sha256", "held_out_fold",
    "artifact_id", "distance", "image_path", "collection",
)


def _cell(row: dict[str, Any]) -> str:
    return "-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_source(*, pool: Path, prior_source: Path, output: Path, seed: str) -> dict[str, Any]:
    """Freeze 32 new image groups, recording any unavailable random-arm quota."""
    candidates, prior = _read_jsonl(pool), _read_jsonl(prior_source)
    group_keys = ("image_group_id", "source_group_id", "near_duplicate_group_id")
    blocked = {key: {str(row[key]) for row in prior if row.get(key)} for key in group_keys}
    selected: list[dict[str, Any]] = []
    for cell in CELLS:
        eligible = [row for row in candidates if _cell(row) == cell and all(not row.get(key) or str(row[key]) not in blocked[key] for key in group_keys)]
        by_pattern: dict[str, list[dict[str, Any]]] = {}
        for row in eligible:
            pattern = str((row.get("private") or {}).get("target_pattern") or "")
            by_pattern.setdefault(pattern, []).append(row)
        for pattern in by_pattern:
            by_pattern[pattern].sort(key=lambda x: hashlib.sha256(f"{seed}:{x['sample_id']}".encode()).hexdigest())
        if any(not by_pattern.get(pattern) for pattern in ("P1", "P2", "P3", "P4")):
            raise ValueError(f"insufficient isolated {cell} candidates for P1--P4 dynamic smoke coverage")
        selected.extend(by_pattern[pattern][0] for pattern in ("P1", "P2", "P3", "P4"))
    if len(selected) != 32 or len({row["image_group_id"] for row in selected}) != 32:
        raise ValueError("dynamic smoke source must contain 32 independent image groups")
    payload = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-source/v1", "seed": seed,
               "rows": selected, "training_eligible": False, "sft_may_start": False}
    _write_json(output, payload)
    return {"rows": len(selected), "cells": dict(sorted(Counter(_cell(row) for row in selected).items())),
            "arms": dict(sorted(Counter((row.get("private") or {}).get("sampling_arm") for row in selected).items())),
            "pattern_targets": dict(sorted(Counter((row.get("private") or {}).get("target_pattern") for row in selected).items())),
            "random_arm_shortfall_recorded": True}


def _load_source(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("dynamic source must be frozen 32-row object")
    return rows


def _attempt_path(root: Path, route: str, sample_id: str, attempt: int) -> Path:
    return root / "public" / route / sample_id / f"attempt-{attempt}"


def _tool_count(parent: dict[str, Any]) -> int:
    return sum(1 for event in parent.get("trace", []) if isinstance(event, dict) and isinstance(event.get("action"), dict) and event["action"].get("type") == "tool")


def _actions(parent: dict[str, Any]) -> list[dict[str, Any]]:
    return [event["action"] for event in parent.get("trace", [])
            if isinstance(event, dict) and isinstance(event.get("action"), dict) and event["action"].get("type") == "tool"]


def route_contract_errors(parent: dict[str, Any], route: str, *, audit_status: str | None = None) -> list[str]:
    """Validate the positive tool-use requirements for a selected route."""
    if parent.get("status") != "closed":
        return ["parent_not_closed"]
    names = [str(action.get("name") or "") for action in _actions(parent)]
    errors: list[str] = []
    if route == "direct":
        if names:
            errors.append("direct_used_tool")
    elif route == "classifier":
        if "agrinet_classifier_predict" not in names:
            errors.append("classifier_predict_missing")
        if "agrinet_rag_search" in names:
            errors.append("classifier_used_rag")
    elif route == "rag":
        if "agrinet_classifier_predict" not in names:
            errors.append("rag_classifier_predict_missing")
        if "agrinet_rag_search" not in names:
            errors.append("rag_search_missing")
    else:
        errors.append("unknown_route")
    if parent.get("final") == "INSUFFICIENT_EVIDENCE" and audit_status != "accept":
        errors.append("p10_requires_accepted_audit")
    return errors


def _canary_gate(path: Path, *, endpoint: str, teacher_model: str) -> str:
    if not path.is_file():
        raise ValueError("dynamic smoke requires an existing canary report")
    report = json.loads(path.read_text(encoding="utf-8"))
    rounds = report.get("rounds")
    if (report.get("endpoint") != endpoint or report.get("model") != teacher_model
            or report.get("generation_ready_for_dynamic_smoke") is not True
            or not isinstance(rounds, list) or len(rounds) != 2
            or any(item.get("requests") != 10 or item.get("valid_deliveries") != 10 or not item.get("pass") for item in rounds)):
        raise ValueError("dynamic smoke canary is not a matching two-round 10/10 pass")
    return sha256(path)


def collect(*, source: Path, output_root: Path, rag_endpoint: str, teacher_model: str, timeout: int, max_tokens: int,
            canary_report: Path, teacher_endpoint: str) -> dict[str, Any]:
    """Run staged routes. A private verdict never feeds back into a teacher prompt."""
    canary_sha256 = _canary_gate(canary_report, endpoint=teacher_endpoint, teacher_model=teacher_model)
    if output_root.exists():
        raise ValueError("dynamic smoke output root already exists")
    output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for row in _load_source(source):
        sample_id, accepted = str(row["sample_id"]), None
        for route, policy in ROUTES.items():
            for attempt in range(1, int(policy["attempts"]) + 1):
                location = _attempt_path(output_root, route, sample_id, attempt)
                try:
                    parent = run_parent(source_row=row, output_root=location, rag_endpoint=rag_endpoint, max_tokens=max_tokens, timeout=timeout,
                                        teacher_model=teacher_model, temperature=float(policy["temperature"]), allowed_tools=set(policy["tools"]))
                    audit = run_private_audit(source_row=row, parent=parent, output_root=location, scope="parent", timeout=timeout,
                                              teacher_model=teacher_model, require_primary_pattern=True)
                    result = {"route": route, "attempt": attempt, "status": parent["status"], "audit": audit["status"],
                              "primary_pattern": audit.get("primary_pattern"), "tool_calls": _tool_count(parent),
                              "generation_calls": len([x for x in parent.get("trace", []) if x.get("action")])}
                    contract_errors = route_contract_errors(parent, route, audit_status=audit["status"])
                    result["route_contract_errors"] = contract_errors
                    if audit["status"] == "accept" and not contract_errors:
                        accepted = {**result, "parent_path": str(location / "public" / sample_id / "trajectory.json"),
                                    "audit_path": str(location / "private" / sample_id / "parent" / "audit.json")}
                        break
                except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
                    result = {"route": route, "attempt": attempt, "status": "not_completed", "audit": None,
                              "reason": type(exc).__name__}
                results.append({"sample_id": sample_id, **result})
            if accepted is not None:
                results.append({"sample_id": sample_id, "selected": True, **accepted})
                break
        if accepted is None:
            results.append({"sample_id": sample_id, "selected": False, "reason": "no_accepted_route"})
    summary = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-smoke/v1", "rows": results,
               "canary_report": str(canary_report), "canary_report_sha256": canary_sha256,
               "training_eligible": False, "sft_may_start": False, "automatic_replay_allowed": False}
    _write_json(output_root / "routing_summary.json", summary)
    return summary


def _public_tool(value: dict[str, Any]) -> dict[str, Any]:
    """Project raw tool payloads to the exact deployable student surface."""
    name = value.get("tool")
    if name in {"agrinet_classifier_predict", "agrinet_classifier_expand"}:
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("classifier result lacks candidates")
        return {"tool": name, "candidates": [{key: item.get(key) for key in ("rank", "name", "name_zh", "score")}
                                                       for item in candidates if isinstance(item, dict)]}
    if name == "agrinet_rag_search":
        raw = value.get("raw_response")
        evidence = raw.get("evidence") if isinstance(raw, dict) else None
        if not isinstance(evidence, list):
            raise ValueError("RAG result lacks evidence")
        cleaned = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            cleaned.append({key: item.get(key) for key in ("rank", "name", "name_zh", "summary") if item.get(key) is not None})
        return {"tool": name, "arguments": value.get("arguments"), "evidence": cleaned}
    raise ValueError("student conversion encountered unsupported tool payload")


def _assert_public(value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False).casefold()
    if any(token.casefold() in rendered for token in _PRIVATE_TOKENS):
        raise ValueError("student conversion encountered private or internal lineage")


def _answer_for_student(row: dict[str, Any], final: str) -> str:
    """Preserve final conclusion and make Option supervision semantic plus parseable."""
    if final == "INSUFFICIENT_EVIDENCE":
        return final
    if row.get("question_type") != "option":
        return final
    matches = [item for item in row.get("public_options") or [] if str(item.get("label") or "").upper() == final.upper()]
    if len(matches) != 1:
        raise ValueError("option final is not a visible option label")
    item = matches[0]
    name = item.get("name_zh") if row.get("language") == "zh" else item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("option lacks public canonical class name")
    return f"{name.strip()} — {final.upper()}"


def _public_registry_names(path: Path, *, expected_sha256: str) -> list[dict[str, str]]:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError("rewrite requires the matching frozen public registry")
    rows = _read_jsonl(path)
    names = []
    for item in rows:
        name = item.get("name") or item.get("canonical_name") or item.get("canonical_english_name")
        name_zh = item.get("name_zh") or item.get("canonical_name_zh") or item.get("canonical_chinese_name")
        if isinstance(name, str) and name.strip():
            names.append({"name": name.strip(), "name_zh": str(name_zh or "").strip()})
    if not names:
        raise ValueError("frozen public registry has no canonical names")
    return names


def _rewrite_prompt(*, row: dict[str, Any], route: str, parent: dict[str, Any], registry_names: list[dict[str, str]]) -> dict[str, Any]:
    """Build a public-only request for a constrained, auditable final CoT."""
    public_trace = []
    for event in parent.get("trace") or []:
        if isinstance(event, dict) and isinstance(event.get("tool"), dict):
            public_trace.append({"tool": _public_tool(event["tool"])})
        elif isinstance(event, dict) and isinstance(event.get("action"), dict):
            action = event["action"]
            public_trace.append({"action": {key: action.get(key) for key in ("type", "name", "arguments")}})
    context = {"route": route, "question": row["question"], "question_type": row["question_type"],
               "language": row["language"], "public_options": row.get("public_options") or [],
               "public_registry_names": registry_names,
               "public_trace": public_trace, "fixed_final": parent["final"]}
    _assert_public(context)
    language_limit = 900 if row.get("language") == "zh" else 1400
    return {"model": "gpt-5.6-sol", "temperature": 0.0, "top_p": 1.0, "max_tokens": 2048,
            "messages": [{"role": "system", "content":
                "Rewrite the supplied public trajectory into one concise, evidence-grounded final response for a small VLM. "
                "Do not add tools, facts, candidate names, or evidence not visible in the image/context. Preserve fixed_final exactly. "
                "Return JSON only with keys reasoning and final. reasoning must have exactly these English headings in order: "
                "Visual observations:, Candidate comparison:, Evidence:, Uncertainty:. Visual observations must give at least three independent visible facts spanning at least two visual dimensions. "
                "Candidate comparison must give selected-class support and two named nearby-class contrasts, each with its own visible counterevidence; never use rank or score as counterevidence. "
                "Evidence may cite only this route's visible image/question, classifier card, and RAG response. Uncertainty must name an unseen or unclear discriminating detail. "
                f"The reasoning length must not exceed {language_limit} characters."},
                         {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}}


def rewrite_prompt_v2(*, row: dict[str, Any], route: str, parent: dict[str, Any], registry_names: list[dict[str, str]]) -> dict[str, Any]:
    """Future-only structured rewrite prompt selected by an immutable patch."""
    request = _rewrite_prompt(row=row, route=route, parent=parent, registry_names=registry_names)
    request["messages"][0]["content"] = (
        "Return JSON only with final, visual_observations, candidate_comparisons, evidence, and uncertainty. "
        "final must preserve fixed_final exactly. visual_observations is an array of at least three independent visible facts. "
        "candidate_comparisons is an array of at least three objects, each with candidate, support, and counterevidence. "
        "Every support and counterevidence statement must use visible evidence; never use scores or ranks. "
        "Evidence and uncertainty must each be nonempty strings. "
        "Do not add tools, facts, candidate names, or evidence not visible in the supplied public context. "
        + ("For the Direct route, do not mention classifier, retrieval, RAG, scores, or ranks. " if route == "direct" else "")
        + "The rendered reasoning must use the fixed four public sections and respect its language character limit."
    )
    return request


def rewrite_prompt_v3(*, row: dict[str, Any], route: str, parent: dict[str, Any], registry_names: list[dict[str, str]]) -> dict[str, Any]:
    """Build a future-only compact rewrite request with its frozen image."""
    request = _rewrite_prompt(row=row, route=route, parent=parent, registry_names=registry_names)
    context = request["messages"][1]["content"]
    route_boundary = ("For the Direct route, do not mention classifier, retrieval, RAG, scores, or ranks. "
                      if route == "direct" else "")
    request["messages"][0]["content"] = (
        "Return JSON only with final, visual_observations, candidate_comparisons, evidence, and uncertainty. "
        "The attached image is available and must be inspected. final must preserve fixed_final exactly, but fixed_final is not evidence. "
        "Never claim the image is unavailable, unseen, missing, or unverified. "
        "visual_observations must be exactly three short image-grounded facts, each at most 120 characters. "
        "candidate_comparisons must be exactly three distinct objects with candidate, support, and counterevidence; each string at most 120 characters. "
        "Every support and counterevidence statement must use the image or supplied public tool context; never use scores or ranks. "
        "evidence and uncertainty must each be nonempty and at most 160 characters. "
        "Do not add tools, facts, candidate names, or evidence outside the image/public context. "
        + route_boundary
        + "The rendered reasoning must use the fixed four public sections and respect its language character limit."
    )
    request["messages"][1]["content"] = [
        {"type": "text", "text": context},
        image_url_content(Path(row["image_path"])),
    ]
    return request


def validate_rewrite(*, row: dict[str, Any], route: str, parent: dict[str, Any], rewrite: dict[str, Any]) -> list[str]:
    reasoning, final = rewrite.get("reasoning"), rewrite.get("final")
    if not isinstance(reasoning, str) or not isinstance(final, str):
        return ["rewrite_not_strings"]
    errors: list[str] = []
    if final != parent.get("final"):
        errors.append("rewrite_changed_final")
    if len(reasoning) > (900 if row.get("language") == "zh" else 1400):
        errors.append("rewrite_length_exceeded")
    positions = [reasoning.find(section) for section in _REWRITE_SECTIONS]
    if positions != sorted(positions) or any(position < 0 for position in positions):
        errors.append("rewrite_sections_invalid")
    observation = reasoning[positions[0]:positions[1]] if not errors or "rewrite_sections_invalid" not in errors else ""
    comparison = reasoning[positions[1]:positions[2]] if observation else ""
    if len(re.findall(r"(?m)^[-*•]|^\s*\d+[.)]", observation)) < 3:
        errors.append("rewrite_observations_under_three")
    if len(re.findall(r"(?m)^[-*•]", comparison)) < 3:
        errors.append("rewrite_comparisons_under_three")
    lowered = reasoning.casefold()
    if any(token.casefold() in lowered for token in _PRIVATE_TOKENS) or "private audit" in lowered:
        errors.append("rewrite_private_leakage")
    if re.search(r"(lower|higher) (score|rank)|top[- ]?\d", lowered):
        errors.append("rewrite_rank_as_evidence")
    if route == "direct" and any(token in lowered for token in ("classifier", "retriev", "rag")):
        errors.append("direct_rewrite_mentions_tool")
    return errors


def render_rewrite_reasoning_v2(rewrite: dict[str, Any]) -> str:
    """Render the future structured schema into the fixed public four-section form."""
    observations = rewrite.get("visual_observations") or []
    comparisons = rewrite.get("candidate_comparisons") or []
    observation_lines = "\n".join(f"- {item}" for item in observations)
    comparison_lines = "\n".join(
        f"- {item['candidate']}: {item['support']} Counterevidence: {item['counterevidence']}"
        for item in comparisons
    )
    return (f"Visual observations:\n{observation_lines}\n"
            f"Candidate comparison:\n{comparison_lines}\n"
            f"Evidence:\n- {rewrite.get('evidence', '')}\n"
            f"Uncertainty:\n- {rewrite.get('uncertainty', '')}")


def validate_rewrite_v2(*, row: dict[str, Any], route: str, parent: dict[str, Any], rewrite: dict[str, Any]) -> list[str]:
    """Validate the patch-v2 schema before rendering it to student-visible text."""
    errors: list[str] = []
    observations, comparisons = rewrite.get("visual_observations"), rewrite.get("candidate_comparisons")
    if not isinstance(observations, list) or len(observations) < 3 or not all(isinstance(item, str) and item.strip() for item in observations):
        errors.append("v2_observations_invalid")
    if not isinstance(comparisons, list) or len(comparisons) < 3:
        errors.append("v2_comparisons_invalid")
    elif any(not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip() for key in ("candidate", "support", "counterevidence")) for item in comparisons):
        errors.append("v2_comparisons_invalid")
    elif len({item["candidate"].casefold().strip() for item in comparisons}) < 3:
        errors.append("v2_comparisons_not_distinct")
    if not isinstance(rewrite.get("evidence"), str) or not rewrite["evidence"].strip():
        errors.append("v2_evidence_invalid")
    if not isinstance(rewrite.get("uncertainty"), str) or not rewrite["uncertainty"].strip():
        errors.append("v2_uncertainty_invalid")
    if not isinstance(rewrite.get("final"), str) or rewrite.get("final") != parent.get("final"):
        errors.append("rewrite_changed_final")
    if errors:
        return errors
    rendered = render_rewrite_reasoning_v2(rewrite)
    return validate_rewrite(row=row, route=route, parent=parent,
                            rewrite={"reasoning": rendered, "final": rewrite["final"]})


def validate_rewrite_v3(*, row: dict[str, Any], route: str, parent: dict[str, Any], rewrite: dict[str, Any]) -> list[str]:
    """Add compactness and visual-evidence assertions to the V2 contract."""
    errors = validate_rewrite_v2(row=row, route=route, parent=parent, rewrite=rewrite)
    observations = rewrite.get("visual_observations")
    comparisons = rewrite.get("candidate_comparisons")
    if isinstance(observations, list):
        if len(observations) != 3 or any(isinstance(item, str) and len(item) > 120 for item in observations):
            errors.append("v3_observation_length_or_count")
    if isinstance(comparisons, list):
        compact = len(comparisons) == 3
        for item in comparisons:
            if isinstance(item, dict):
                compact = compact and all(isinstance(item.get(key), str) and len(item[key]) <= 120
                                          for key in ("candidate", "support", "counterevidence"))
        if not compact:
            errors.append("v3_comparison_length_or_count")
    for key in ("evidence", "uncertainty"):
        if isinstance(rewrite.get(key), str) and len(rewrite[key]) > 160:
            errors.append("v3_evidence_or_uncertainty_length")
    fragments: list[str] = []
    if isinstance(observations, list):
        fragments.extend(item for item in observations if isinstance(item, str))
    if isinstance(comparisons, list):
        for item in comparisons:
            if isinstance(item, dict):
                fragments.extend(value for value in (item.get("support"), item.get("counterevidence"))
                                 if isinstance(value, str))
    fragments.extend(value for value in (rewrite.get("evidence"), rewrite.get("uncertainty"))
                     if isinstance(value, str))
    public_text = " ".join(fragments).casefold()
    banned = ("no image", "image unavailable", "cannot see the image", "did not see the image",
              "未见图像", "没有图像", "无法查看图像", "按固定答案", "固定答案作答", "fixed final")
    if any(token in public_text for token in banned):
        errors.append("v3_nonvisual_or_fixed_final_claim")
    return list(dict.fromkeys(errors))


def _parse_rewrite(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
        value = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("reasoning rewrite response is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("reasoning rewrite response must be an object")
    return value


def rewrite(*, source: Path, routing_summary: Path, output_root: Path, timeout: int, public_registry: Path,
            teacher_model: str = "gpt-5.6-sol") -> dict[str, Any]:
    """Create and independently audit public final reasoning for selected winners."""
    if output_root.exists():
        raise ValueError("dynamic rewrite output root already exists")
    source_by_id = {str(row["sample_id"]): row for row in _load_source(source)}
    registry_sha256 = str((next(iter(source_by_id.values()), {}).get("prediction") or {}).get("registry_sha256") or "")
    registry_names = _public_registry_names(public_registry, expected_sha256=registry_sha256)
    summary = json.loads(routing_summary.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for choice in [item for item in summary.get("rows", []) if item.get("selected") is True]:
        sample_id, route = str(choice["sample_id"]), str(choice["route"])
        row, parent_path = source_by_id.get(sample_id), Path(str(choice.get("parent_path") or ""))
        if row is None or not parent_path.is_file():
            rows.append({"sample_id": sample_id, "status": "excluded", "reason": "missing_parent"}); continue
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        errors = route_contract_errors(parent, route, audit_status="accept")
        if errors:
            rows.append({"sample_id": sample_id, "status": "excluded", "reason": "route_contract_violation", "errors": errors}); continue
        directory = output_root / "public" / sample_id
        ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                               image_group_id=row["image_group_id"], view="student_visible")
        request = _rewrite_prompt(row=row, route=route, parent=parent, registry_names=registry_names); request["model"] = teacher_model
        GlobalMicuBudget(output_root / "global_micu_events.jsonl").reserve(f"rewrite:{sample_id}:generation:1")
        raw = ledger.call("generation", "reasoning-rewrite-1",
                          {"operation": "reasoning_rewrite", "sample_id": sample_id, "route": route,
                           "parent_sha256": sha256(parent_path)},
                          lambda _summary: _invoke_micu(request, timeout=timeout))
        rewritten = _parse_rewrite(raw)
        errors = validate_rewrite(row=row, route=route, parent=parent, rewrite=rewritten)
        result = {"schema_version": "agrinet.micu-classifier-hcv-e2-rewrite/v1", "sample_id": sample_id,
                  "route": route, "parent_path": str(parent_path), "rewrite": rewritten,
                  "status": "closed" if not errors else "excluded", "errors": errors,
                  "training_eligible": False}
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "rewrite.json", result)
        if not errors:
            audit_subject = {**parent, "reasoning_rewrite": rewritten}
            audit = run_private_audit(source_row=row, parent=audit_subject, output_root=directory, scope="rewrite",
                                      timeout=timeout, teacher_model=teacher_model, require_primary_pattern=True)
            result["rewrite_audit"] = audit["status"]
            if audit["status"] != "accept":
                result["status"] = "excluded"; result["errors"] = ["rewrite_audit_not_accepted"]
            _write_json(directory / "rewrite.json", result)
        rows.append({key: result.get(key) for key in ("sample_id", "route", "status", "errors", "rewrite_audit")})
    report = {"schema_version": "agrinet.micu-classifier-hcv-e2-rewrite-report/v1", "rows": rows,
              "public_registry": str(public_registry), "public_registry_sha256": registry_sha256,
              "training_eligible": False, "sft_may_start": False}
    _write_json(output_root / "rewrite_summary.json", report)
    return report


def _tool_argument_errors(name: str, arguments: dict[str, Any]) -> list[str]:
    if name == "agrinet_classifier_predict":
        return [] if arguments == {} else ["classifier predict arguments must be empty"]
    if name == "agrinet_classifier_expand":
        return [] if isinstance(arguments.get("reason"), str) and arguments["reason"].strip() else ["expand reason required"]
    if name == "agrinet_rag_search":
        return [] if (arguments.get("retrieval_type") in {"visual", "semantic"}
                      and isinstance(arguments.get("query"), str) and arguments["query"].strip()
                      and isinstance(arguments.get("rationale"), str) and arguments["rationale"].strip()) else ["valid RAG arguments required"]
    return ["unsupported tool"]


def convert(*, source: Path, routing_summary: Path, rewrite_summary: Path, output_root: Path) -> dict[str, Any]:
    """Render selected public trajectories into non-authorized student candidates."""
    if output_root.exists():
        raise ValueError("dynamic conversion output root already exists")
    source_by_id = {str(row["sample_id"]): row for row in _load_source(source)}
    summary = json.loads(routing_summary.read_text(encoding="utf-8"))
    rewrite_by_id = {str(item.get("sample_id")): item for item in json.loads(rewrite_summary.read_text(encoding="utf-8")).get("rows", [])}
    selected = [row for row in summary.get("rows", []) if row.get("selected") is True]
    output_root.mkdir(parents=True)
    views: dict[str, list[dict[str, Any]]] = {"direct": [], "classifier": [], "rag": []}
    lineage: list[dict[str, Any]] = []; excluded: list[dict[str, Any]] = []
    for choice in selected:
        sample_id, route = str(choice["sample_id"]), str(choice["route"])
        row = source_by_id.get(sample_id)
        parent_path = Path(str(choice["parent_path"]))
        rewrite_row = rewrite_by_id.get(sample_id)
        rewrite_path = rewrite_summary.parent / "public" / sample_id / "rewrite.json"
        if row is None or not parent_path.is_file() or route not in views or rewrite_row is None or rewrite_row.get("status") != "closed" or not rewrite_path.is_file():
            excluded.append({"sample_id": sample_id, "reason": "missing_selected_public_lineage"}); continue
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        rewritten = json.loads(rewrite_path.read_text(encoding="utf-8"))
        trace = parent.get("trace")
        if parent.get("status") != "closed" or not isinstance(trace, list):
            excluded.append({"sample_id": sample_id, "reason": "selected_parent_not_closed"}); continue
        public_trace = []
        for event in trace:
            if not isinstance(event, dict):
                raise ValueError("public trace event is invalid")
            if event.get("tool", {}).get("tool") == "controller":
                raise ValueError("controller-denied trace cannot supervise a student")
            public_trace.append(event)
        question = str(row["question"]); options = row.get("public_options") or []
        user = "<image>\n" + question
        if options:
            user += "\n" + "\n".join(f"{item['label']}. {item['name_zh'] if row['language'] == 'zh' else item['name']}" for item in options)
        reasoning = (rewritten.get("rewrite") or {}).get("reasoning")
        final = (rewritten.get("rewrite") or {}).get("final")
        rewrite_errors = validate_rewrite(row=row, route=route, parent=parent, rewrite={"reasoning": reasoning, "final": final})
        if rewrite_errors:
            excluded.append({"sample_id": sample_id, "reason": "invalid_rewrite", "errors": rewrite_errors}); continue
        final_message = f"<think>\n{reasoning.strip()}\n</think>\n<answer>{_answer_for_student(row, final)}</answer>"
        if route == "direct":
            messages = [{"role": "system", "content": "Identify the agricultural disease or pest from visible image evidence. State INSUFFICIENT_EVIDENCE when needed."},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": final_message}]
        else:
            messages = [{"role": "system", "content": "Use tools only when needed. Classifier scores rank candidates and are not known probabilities. Base the final answer only on visible evidence and actual tool returns."},
                        {"role": "user", "content": user}]
            for event in public_trace:
                if "action" in event:
                    action = event["action"]
                    if action.get("type") == "tool":
                        messages.append({"role": "assistant", "content": "<think>Request the next public evidence needed to resolve the visible uncertainty.</think>"})
                        messages.append({"role": "tool_call", "content": json.dumps({"name": action["name"], "arguments": action["arguments"]}, ensure_ascii=False)})
                elif "tool" in event:
                    tool = event["tool"]
                    messages.append({"role": "tool", "content": json.dumps(_public_tool(tool), ensure_ascii=False)})
            messages.append({"role": "assistant", "content": final_message})
            messages = normalize_training_messages(messages, tool_name=set(ROUTES[route]["tools"]),
                                                   validate_arguments=lambda arguments: _tool_argument_errors("agrinet_rag_search" if "query" in arguments else ("agrinet_classifier_expand" if arguments else "agrinet_classifier_predict"), arguments),
                                                   require_tool_calls=True)
        views[route].append({"messages": messages, "images": [row["image_path"]],
                             "metadata": {"sample_id": sample_id, "image_group_id": row["image_group_id"],
                                          "route": route, "question_type": row["question_type"], "language": row["language"],
                                          "task_domain": row["task_domain"], "training_eligible": False}})
        lineage.append({"sample_id": sample_id, "image_group_id": row["image_group_id"], "route": route,
                        "parent_sha256": sha256(parent_path), "audit_path": choice["audit_path"],
                        "tool_calls": choice["tool_calls"], "generation_calls": choice["generation_calls"],
                        "teacher_model": "gpt-5.6-sol", "training_eligible": False})
    for route, rows in views.items():
        (output_root / f"{route}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    mask_report = {"schema_version": "agrinet.micu-classifier-hcv-e2-dynamic-mask/v1",
                   "views": {route: {"rows": len(rows), "assistant_turns": sum(sum(m["role"] == "assistant" for m in row["messages"]) for row in rows),
                                      "tool_turns": sum(sum(m["role"] == "tool" for m in row["messages"]) for row in rows)} for route, rows in views.items()},
                   "mask_policy": "assistant_and_tool_call_only; system,user,tool_unmasked", "training_eligible": False, "sft_may_start": False}
    _write_json(output_root / "lineage.json", {"rows": lineage, "excluded": excluded, "training_eligible": False})
    _write_json(output_root / "token_mask_report.json", mask_report)
    return mask_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    freeze = sub.add_parser("freeze-source")
    freeze.add_argument("--pool", type=Path, required=True); freeze.add_argument("--prior-source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True); freeze.add_argument("--seed", required=True)
    collect_cmd = sub.add_parser("collect")
    collect_cmd.add_argument("--source", type=Path, required=True); collect_cmd.add_argument("--output-root", type=Path, required=True)
    collect_cmd.add_argument("--rag-endpoint", required=True); collect_cmd.add_argument("--teacher-model", default="gpt-5.6-sol")
    collect_cmd.add_argument("--timeout", type=int, default=180); collect_cmd.add_argument("--max-tokens", type=int, default=8192)
    collect_cmd.add_argument("--canary-report", type=Path, required=True); collect_cmd.add_argument("--teacher-endpoint", required=True)
    rewrite_cmd = sub.add_parser("rewrite")
    rewrite_cmd.add_argument("--source", type=Path, required=True); rewrite_cmd.add_argument("--routing-summary", type=Path, required=True)
    rewrite_cmd.add_argument("--output-root", type=Path, required=True); rewrite_cmd.add_argument("--public-registry", type=Path, required=True); rewrite_cmd.add_argument("--timeout", type=int, default=180)
    rewrite_cmd.add_argument("--teacher-model", default="gpt-5.6-sol")
    convert_cmd = sub.add_parser("convert")
    convert_cmd.add_argument("--source", type=Path, required=True); convert_cmd.add_argument("--routing-summary", type=Path, required=True)
    convert_cmd.add_argument("--rewrite-summary", type=Path, required=True); convert_cmd.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "freeze-source": result = freeze_source(pool=args.pool, prior_source=args.prior_source, output=args.output, seed=args.seed)
    elif args.operation == "collect": result = collect(source=args.source, output_root=args.output_root, rag_endpoint=args.rag_endpoint, teacher_model=args.teacher_model, timeout=args.timeout, max_tokens=args.max_tokens, canary_report=args.canary_report, teacher_endpoint=args.teacher_endpoint)
    elif args.operation == "rewrite": result = rewrite(source=args.source, routing_summary=args.routing_summary, output_root=args.output_root, public_registry=args.public_registry, timeout=args.timeout, teacher_model=args.teacher_model)
    else: result = convert(source=args.source, routing_summary=args.routing_summary, rewrite_summary=args.rewrite_summary, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
