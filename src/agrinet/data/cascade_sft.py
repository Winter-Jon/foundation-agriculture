"""Build the immutable English E3.43 four-route SFT candidate dataset.

This is a local derivative only: it reads settled collection artifacts and never
contacts a provider, classifier, retriever, or training runtime.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from agrinet.data.io import DataError, write_jsonl_atomic

ARTIFACT_ID = "agrinet-e343-four-level-cascade-sft-candidate-v2"
FLAGS = {"training_eligible": False, "training_authorized": False, "sft_may_start": False}
ROUTES = ("direct", "classifier", "rag", "refusal")
TRAINING_ARTIFACT_ID = "agrinet-e343-three-route-sft-v3"


def approve_non_refusal(root: Path) -> dict[str, Any]:
    """Materialize the user's accepted non-refusal subset without launching SFT."""
    source = root / "outputs/artifacts/datasets" / ARTIFACT_ID
    destination = source.parent / TRAINING_ARTIFACT_ID
    if destination.exists():
        raise DataError(f"refuse existing immutable destination: {destination}")
    manifest = read_json(source / "artifact.yaml")
    for filename, key in (("data.jsonl", "data_sha256"), ("lineage.jsonl", "lineage_sha256"),
                          ("exclusions.jsonl", "exclusions_sha256"), ("source-admission.json", "source_admission_sha256")):
        if digest(source / filename) != manifest.get(key):
            raise DataError(f"parent artifact hash mismatch: {filename}")
    rows, lines = read_jsonl(source / "data.jsonl"), read_jsonl(source / "lineage.jsonl")
    if len(rows) != 1031 or len(lines) != len(rows):
        raise DataError("unexpected parent coverage")
    data, lineage = [], []
    exclusions = read_jsonl(source / "exclusions.jsonl")
    # User accepted the data for training; actual execution is a separate action.
    admitted = {"training_eligible": True, "training_authorized": False, "sft_may_start": False}
    for row, line in zip(rows, lines):
        if row["sample_id"] != line["sample_id"] or canonical_hash(row) != line["student_row_sha256"]:
            raise DataError("parent row lineage mismatch")
        if line["route"] == "refusal":
            exclusions.append({**{k: line[k] for k in ("sample_id", "image_sha256", "source_group_id", "near_duplicate_group_id")},
                               "reason": "refusal_excluded_by_user", **FLAGS})
            continue
        if line["route"] not in {"direct", "classifier", "rag"}:
            raise DataError("unexpected parent route")
        # Preserve training payload exactly; source_artifact_id identifies v2.
        data.append(row)
        lineage.append({**line, "admitted_artifact_id": TRAINING_ARTIFACT_ID, **admitted})
    counts = dict(Counter(line["route"] for line in lineage))
    if counts != {"direct": 530, "classifier": 255, "rag": 165} or len(exclusions) != 120:
        raise DataError("non-refusal conservation failed")
    if len({x["sample_id"] for x in data + exclusions}) != 1070:
        raise DataError("subset identity partition failed")
    stats = {"rows": len(data), "routes": {**counts, "refusal": 0}, "refusal_rows": 0,
             "tool_rows": sum(bool(row["tools"]) for row in data),
             "strata": {key: dict(Counter(str(line["strata"].get(key)) for line in lineage))
                        for key in ("arm", "question_type", "domain", "fold", "e327_overlap")}, **admitted}
    admission = {"artifact_id": TRAINING_ARTIFACT_ID, "parent_artifact_id": ARTIFACT_ID,
                 "parent_manifest_sha256": digest(source / "artifact.yaml"),
                 "decision": "User accepts all non-refusal v2 samples as training data; exclude all 81 refusal samples.",
                 "approval_basis": "explicit_user_acceptance", "training_execution_requested": False,
                 "rows": len(data), "route_counts": counts, "excluded_refusal_rows": 81,
                 "residual_count": len(exclusions),
                 "conservation": {"all_images": 1070, "training_rows": 950, "excluded_rows": 120, "holds": True},
                 **admitted}
    isolation = {key: {line[key]: [line["sample_id"]] for line in lineage}
                 for key in ("image_sha256", "source_group_id", "near_duplicate_group_id")}
    destination.mkdir(parents=True)
    for filename, values in (("data.jsonl", data), ("lineage.jsonl", lineage), ("exclusions.jsonl", exclusions)):
        write_jsonl_atomic(destination / filename, values)
    _write_json(destination / "source-admission.json", admission)
    _write_json(destination / "statistics.json", stats)
    _write_json(destination / "evaluation-isolation.json", {"schema_version": "agrinet.sft.evaluation-isolation/v1",
                "groups": isolation, "note": "Isolation metadata only; no train/validation split created.", **admitted})
    _write_json(destination / "artifact.yaml", {"schema_version": "agrinet.sft.frozen/v1",
                "artifact_id": TRAINING_ARTIFACT_ID, "artifact_type": "datasets", "immutable": True,
                "parent_artifact_id": ARTIFACT_ID, "statistics": stats,
                **{key: digest(destination / filename) for filename, key in
                   (("data.jsonl", "data_sha256"), ("lineage.jsonl", "lineage_sha256"),
                    ("exclusions.jsonl", "exclusions_sha256"), ("source-admission.json", "source_admission_sha256"))},
                "no_online_collection": True, **admitted})
    (destination / "README.md").write_text(
        f"# {TRAINING_ARTIFACT_ID}\n\n950 user-accepted training samples: Direct 530, Classifier 255, RAG 165.\n"
        "All 81 Refusal samples excluded. Frozen v2 payloads retained exactly, including public multilingual evidence.\n"
        "Data admission is approved; training execution has not been requested or started.\n", encoding="utf-8")
    return {"artifact_dir": str(destination), "rows": len(data), "routes": counts, "excluded_refusal_rows": 81, **admitted}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DataError(f"row must be object: {path}:{number}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _absolute(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _outcomes(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        value = read_json(path)
        for row in value.get("outcomes") or []:
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise DataError(f"invalid outcome in {path}")
            latest[row["sample_id"]] = {**row, "_outcome_path": str(path)}
    return latest


def _final_outcomes(root: Path, campaign: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    latest = _outcomes(campaign.glob("shards/shard-*/outcomes/*.json"))
    if set(latest) != expected_ids:
        missing, extra = expected_ids - set(latest), set(latest) - expected_ids
        raise DataError(f"terminal coverage mismatch missing={len(missing)} extra={len(extra)}")
    return latest


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"missing {field}")
    return value


def _public_user(question: str) -> dict[str, str]:
    return {"role": "user", "content": f"<image>\n{question}"}


def _public_question(source: dict[str, Any]) -> str:
    question = _safe_text(source.get("question"), "question")
    if source.get("question_type") == "option":
        options = source.get("public_options")
        if not isinstance(options, list) or len(options) != 4 or [x.get("label") for x in options if isinstance(x, dict)] != list("ABCD"):
            raise DataError("Option input requires four ordered public options A/B/C/D")
        question += "\n" + "\n".join(f"{x['label']}. {_safe_text(x.get('name'), 'option name')}" for x in options)
    return question


def _final_answer(messages: list[dict[str, Any]], answer: str) -> str:
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == answer:
        return answer
    if not re.fullmatch(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", answer, re.DOTALL):
        raise DataError("final public answer is not HCV answer format")
    return answer


def _check_no_private(value: Any, sample_id: str) -> None:
    rendered = json.dumps(value, ensure_ascii=False).lower()
    forbidden = ("truth_code", "label_map", "checkpoint_sha", "held_out_fold", "private_audit", "request_id", "provider_response")
    if any(token in rendered for token in forbidden):
        raise DataError(f"{sample_id}: private metadata leaked into student row")


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, str]:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))}


def _classifier_tools(trace: list[dict[str, Any]], sample_id: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    messages: list[dict[str, str]] = []; schemas: list[dict[str, Any]] = []
    calls = []
    for item in trace:
        call, response = item.get("call"), item.get("response")
        if not isinstance(call, dict) or not isinstance(response, dict):
            raise DataError(f"{sample_id}: malformed classifier trace")
        name, args = call.get("name"), call.get("arguments")
        if name not in {"agrinet_classifier_predict", "agrinet_classifier_expand"} or not isinstance(args, dict):
            raise DataError(f"{sample_id}: illegal classifier call")
        calls.append(str(name)); messages.extend([_tool_call(str(name), args), {"role": "tool", "content": json.dumps(response, ensure_ascii=False, separators=(",", ":"))}])
    if calls.count("agrinet_classifier_predict") != 1 or calls.count("agrinet_classifier_expand") > 1:
        raise DataError(f"{sample_id}: classifier call cardinality invalid")
    schemas = [
        {"type": "function", "function": {"name": "agrinet_classifier_predict", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "agrinet_classifier_expand", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}}},
    ]
    return messages, schemas


def _english_card(trace_item: dict[str, Any], sample_id: str) -> dict[str, Any]:
    response = trace_item.get("response")
    if not isinstance(response, dict) or response.get("tool") != "agrinet_rag_search":
        raise DataError(f"{sample_id}: missing RAG response")
    raw = response.get("raw_response")
    evidence = raw.get("evidence") if isinstance(raw, dict) else None
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise DataError(f"{sample_id}: RAG must retain exactly three slots")
    results: list[dict[str, Any]] = []
    for rank, item in enumerate(evidence, 1):
        metadata = item.get("metadata") if isinstance(item, dict) else None
        name = metadata.get("english_name") if isinstance(metadata, dict) else None
        descriptions = metadata.get("visual_descriptions") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise DataError(f"{sample_id}: RAG slot {rank} lacks public English name")
        english = next((text for text in descriptions if isinstance(text, str) and text.strip() and not re.search(r"[\u4e00-\u9fff]", text)), None) if isinstance(descriptions, list) else None
        if english is None and isinstance(metadata, dict):
            fallback = metadata.get("public_description")
            if isinstance(fallback, str) and fallback.strip() and not re.search(r"[\u4e00-\u9fff]", fallback):
                english = fallback
        if english is None or not isinstance(item.get("score"), (int, float)):
            raise DataError(f"{sample_id}: RAG slot {rank} lacks English morphology card")
        # Retain every teacher-visible description, not just the first snippet.
        results.append({"rank": rank, "name": name, "score": item["score"],
                        "public_description": metadata.get("public_description"),
                        "visual_descriptions": descriptions})
    call = trace_item.get("call")
    if not isinstance(call, dict) or call.get("name") != "agrinet_rag_search":
        raise DataError(f"{sample_id}: missing RAG call")
    args = dict(call.get("arguments") or {})
    if args.get("query") != "visual morphology" or args.get("retrieval_type") != "visual" or args.get("top_k") != 3 or "ranker" in args:
        raise DataError(f"{sample_id}: RAG call is not fixed visual Top-3")
    args["image"] = "query_image"
    return {"schema_version": "agrinet.sft.public-rag-card/v2", "tool": "agrinet_rag_search", "arguments": args, "results": results}


def _rag_schema() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "agrinet_rag_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "retrieval_type": {"enum": ["visual"]}, "image": {"enum": ["query_image"]}, "top_k": {"type": "integer"}, "rationale": {"type": "string"}}, "required": ["query", "retrieval_type", "image", "top_k", "rationale"], "additionalProperties": False}}}


def _row_from_trajectory(*, route: str, source: dict[str, Any], trajectory_path: Path, trajectory_sha: str, outcome: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = _safe_text(source.get("sample_id"), "sample_id")
    if not trajectory_path.is_file() or digest(trajectory_path) != trajectory_sha:
        raise DataError(f"{sample_id}: trajectory SHA binding failed")
    trace_doc = read_json(trajectory_path)
    if trace_doc.get("route") != route:
        raise DataError(f"{sample_id}: route mismatch")
    answer = _safe_text(trace_doc.get("answer"), "public answer")
    original = trace_doc.get("messages")
    if not isinstance(original, list):
        original = []
    system = next((m.get("content") for m in original if isinstance(m, dict) and m.get("role") == "system"), source.get("system_prompt", ""))
    if not isinstance(system, str) or not system.strip():
        raise DataError(f"{sample_id}: missing public system prompt")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, _public_user(_public_question(source))]
    tools: list[dict[str, Any]] = []
    trace = trace_doc.get("tool_trace")
    if not isinstance(trace, list):
        raise DataError(f"{sample_id}: tool trace missing")
    if route in {"classifier", "rag"}:
        pre_tool = [m for m in original if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip().startswith("<think>") and m["content"].strip().endswith("</think>")]
        if pre_tool:
            messages.append({"role": "assistant", "content": pre_tool[0]["content"]})
    if route == "direct":
        if trace:
            raise DataError(f"{sample_id}: Direct trace has tools")
    elif route == "classifier":
        tool_messages, tools = _classifier_tools(trace, sample_id)
        messages.extend(tool_messages)
    elif route == "rag":
        classifier_trace = [item for item in trace if isinstance(item, dict) and isinstance(item.get("call"), dict) and item["call"].get("name") in {"agrinet_classifier_predict", "agrinet_classifier_expand"}]
        rag_trace = [item for item in trace if isinstance(item, dict) and isinstance(item.get("call"), dict) and item["call"].get("name") == "agrinet_rag_search"]
        if len(rag_trace) != 1:
            raise DataError(f"{sample_id}: RAG call cardinality invalid")
        classifier_messages, classifier_schemas = _classifier_tools(classifier_trace, sample_id)
        card = _english_card(rag_trace[0], sample_id)
        messages.extend(classifier_messages)
        messages.extend([_tool_call("agrinet_rag_search", card["arguments"]), {"role": "tool", "content": json.dumps(card, ensure_ascii=False, separators=(",", ":"))}])
        tools = [*classifier_schemas, _rag_schema()]
    elif route == "refusal":
        if trace or answer.strip().split("<answer>")[-1].split("</answer>")[0].strip() != "INSUFFICIENT_EVIDENCE":
            raise DataError(f"{sample_id}: refusal route contract invalid")
        parent = trace_doc.get("parent_e341")
        if not isinstance(parent, dict):
            raise DataError(f"{sample_id}: missing refusal parent context")
        parent_path = _absolute(root, _safe_text(parent.get("trajectory_path"), "parent trajectory"))
        evidence_path = _absolute(root, _safe_text(parent.get("evidence_path"), "parent evidence"))
        for path, key in ((parent_path, "trajectory_sha256"), (evidence_path, "evidence_sha256")):
            if not path.is_file() or digest(path) != parent.get(key):
                raise DataError(f"{sample_id}: refusal parent SHA binding failed")
        parent_doc = read_json(parent_path)
        evidence = read_json(evidence_path)
        if parent_doc.get("route") != "rag":
            raise DataError(f"{sample_id}: refusal parent is not RAG")
        parent_trace = parent_doc.get("tool_trace") or []
        classifier_trace = [x for x in parent_trace if x.get("call", {}).get("name") in {"agrinet_classifier_predict", "agrinet_classifier_expand"}]
        tool_messages, classifier_schemas = _classifier_tools(classifier_trace, sample_id)
        rag_trace = [x for x in parent_trace if x.get("call", {}).get("name") == "agrinet_rag_search"]
        if len(rag_trace) != 1 or rag_trace[0].get("response") != evidence:
            raise DataError(f"{sample_id}: refusal parent evidence mismatch")
        card = _english_card(rag_trace[0], sample_id)
        messages.extend(tool_messages)
        messages.extend([_tool_call("agrinet_rag_search", card["arguments"]),
                         {"role": "tool", "content": json.dumps(card, ensure_ascii=False, separators=(",", ":"))}])
        # Preserve the prior answer the refusal teacher saw as context, not
        # another assistant target or an invented tool response.
        messages.append({"role": "user", "content": "Reassess the following prior assessment against the supplied public evidence. It may be incorrect.\n" + _safe_text(parent_doc.get("answer"), "parent answer")})
        tools = [*classifier_schemas, _rag_schema()]
    else:
        raise DataError(f"unsupported route {route}")
    messages.append({"role": "assistant", "content": _final_answer(messages, answer)})
    row = {"schema_version": "agrinet.sft.student/v1", "sample_id": sample_id, "messages": messages, "images": [_safe_text(source.get("image_path"), "image_path")], "tools": tools, "source_artifact_id": ARTIFACT_ID}
    _check_no_private(row, sample_id)
    lineage = {"sample_id": sample_id, "route": route, "source_path": str(trajectory_path.relative_to(root)), "trajectory_sha256": trajectory_sha, "outcome_path": str(Path(outcome["_outcome_path"]).relative_to(root)), "outcome_sha256": canonical_hash({key: value for key, value in outcome.items() if key != "_outcome_path"}), "image_sha256": source.get("image_sha256"), "source_group_id": source.get("source_group_id"), "near_duplicate_group_id": source.get("near_duplicate_group_id"), "strata": {key: source.get(key) for key in ("arm", "question_type", "domain", "fold", "e327_overlap")}, "student_row_sha256": canonical_hash(row), **FLAGS}
    return row, lineage


def _check_gate(path: Path, predicate: str) -> None:
    value = read_json(path)
    if value.get(predicate) is not True:
        raise DataError(f"required upstream gate failed: {path}")


def _write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise DataError(f"immutable output conflict: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def _winner_map(paths: Iterable[Path], disposition: str) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        for outcome in read_json(path).get("outcomes") or []:
            if not isinstance(outcome, dict) or outcome.get("disposition") != disposition or outcome.get("delivery_status") != "delivered":
                continue
            sample_id = outcome.get("sample_id")
            if isinstance(sample_id, str):
                winners[sample_id] = {**outcome, "_outcome_path": str(path)}
    return winners


def _source_map(*paths: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                if sample_id in result and result[sample_id].get("image_sha256") != row.get("image_sha256"):
                    raise DataError(f"inconsistent source identity: {sample_id}")
                result[sample_id] = row
    return result


def _trajectory_binding(outcome: dict[str, Any], root: Path) -> tuple[Path, str]:
    path = outcome.get("trajectory_path") or outcome.get("parent_path")
    if not isinstance(path, str):
        raise DataError(f"{outcome.get('sample_id')}: outcome has no trajectory path")
    resolved = _absolute(root, path)
    sha = outcome.get("trajectory_sha256")
    return resolved, str(sha) if isinstance(sha, str) else digest(resolved)


def build(root: Path, destination: Path) -> dict[str, Any]:
    """Build the immutable candidate dataset from the current audited cascade."""
    destination = destination.resolve()
    if destination.exists():
        raise DataError(f"refuse existing immutable destination: {destination}")
    e320_source = root / "outputs/artifacts/e320-four-stage-cascade/sources/e320-four-stage-cascade-source-v1.jsonl"
    e321_source = root / "outputs/artifacts/e321-direct-first-hcv-cascade/sources/e321-direct-full-source-v1.jsonl"
    e338_source = root / "outputs/artifacts/e338-classifier-full-v1/source.jsonl"
    e341_source = root / "outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/source.jsonl"
    e343_source = root / "outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/source.jsonl"
    required = [e320_source, e321_source, e338_source, e341_source, e343_source]
    if any(not path.is_file() for path in required):
        raise DataError("required cascade source is missing")
    # E3.41 and the corrected E3.43 gate are the downstream admission authority.
    _check_gate(root / "outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/campaign/final-gate-decision.json", "partial_rag_input_gate_passed")
    correction = read_json(root / "outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/final-gate-correction.json")
    if correction.get("refusal_gate_passed") is not True:
        raise DataError("E3.43 corrected refusal gate is not passed")
    sources = _source_map(e320_source, e321_source, e338_source, e341_source, e343_source)
    direct = _winner_map((root / "outputs/artifacts/e321-direct-first-hcv-cascade/outcomes").glob("*.json"), "semantic_correct")
    direct.update(_winner_map([root / "outputs/artifacts/e320-four-stage-cascade/outcomes/e320-four-stage-cascade-r1.json"], "semantic_correct"))
    # E3.41's frozen 261-row source is the authoritative Classifier→RAG
    # boundary after the sample-level remediation.  E3.38's historical
    # per-round disposition field is not sufficient on its own: it retains
    # pre-remediation values for some otherwise quality-pass terminals.
    classifier_all = _outcomes((root / "outputs/artifacts/e338-classifier-full-v1/campaign").glob("shards/shard-*/outcomes/*.json"))
    rag_input_ids = {row["sample_id"] for row in read_jsonl(e341_source)}
    classifier = {sample_id: outcome for sample_id, outcome in classifier_all.items() if sample_id not in rag_input_ids and outcome.get("delivery_status") == "delivered" and outcome.get("quality") == "pass" and isinstance(outcome.get("parent_path"), str)}
    rag = _winner_map((root / "outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/campaign").glob("shards/shard-*/outcomes/*.json"), "semantic_correct")
    refusal = _winner_map((root / "outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign").glob("shards/shard-*/outcomes/*.json"), "refusal_trajectory_complete")
    # Nine E3.38 Classifier terminal shortfalls have no quality-pass public
    # successor.  They remain residuals; do not convert them merely to make
    # the historical 1,040 summary add up.
    # E3.22 v3's 16 reused Classifier winners have their own audited outcome
    # ledger rather than an E3.38 provider outcome; bind them explicitly.
    e322_reused = _winner_map((root / "outputs/artifacts/e322-classifier-presample-v3/campaign/outcomes").glob("*.json"), "semantic_correct")
    for sample_id in list(classifier):
        if sources[sample_id].get("e338_provenance") == "reused_e322_v3":
            classifier.pop(sample_id)
    classifier.update(e322_reused)
    expected = {"direct": 530, "classifier": 255, "rag": 165, "refusal": 81}
    selected = {"direct": direct, "classifier": classifier, "rag": rag, "refusal": refusal}
    counts = {route: len(value) for route, value in selected.items()}
    if counts != expected:
        raise DataError(f"unexpected terminal winner counts: {counts}")
    all_ids = [sample_id for values in selected.values() for sample_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise DataError("sample appears in more than one final route")
    residual_direct = set(sources) - set(all_ids)
    if len(sources) != 1070 or len(residual_direct) != 39:
        raise DataError(f"cascade conservation invalid sources={len(sources)} residuals={len(residual_direct)}")
    data: list[dict[str, Any]] = []; lineage: list[dict[str, Any]] = []
    for route in ROUTES:
        for sample_id, outcome in sorted(selected[route].items()):
            source = sources.get(sample_id)
            if source is None:
                raise DataError(f"source absent for {sample_id}")
            path, sha = _trajectory_binding(outcome, root)
            row, line = _row_from_trajectory(route=route, source=source, trajectory_path=path, trajectory_sha=sha, outcome=outcome, root=root)
            data.append(row); lineage.append(line)
    identities = ("sample_id", "image_sha256", "source_group_id", "near_duplicate_group_id")
    for key in identities:
        values = [line.get(key) for line in lineage]
        if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
            raise DataError(f"candidate identity is not unique: {key}")
    classifier_shortfalls = {item["sample_id"] for item in read_json(root / "outputs/artifacts/e338-classifier-full-v1/campaign/final-report.json").get("classifier_terminal_shortfalls") or []}
    rag_shortfalls = set(read_json(root / "outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/campaign/final-report.json").get("terminal_shortfalls") or [])
    # Earlier E3.43 unknown deliveries can be repaired.  Only the final report
    # names the one terminal R2 unknown residual.
    e343_unknown = set(read_json(root / "outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/final-report.json").get("terminal_unknown_delivery") or [])
    exclusions = []
    for sample_id in sorted(residual_direct):
        source = sources[sample_id]
        reason = "direct_frozen_residual"
        if sample_id in classifier_shortfalls:
            reason = "classifier_terminal_shortfall"
        elif sample_id in rag_shortfalls:
            reason = "rag_quality_reject"
        elif sample_id in e343_unknown:
            reason = "refusal_delivery_unknown"
        exclusions.append({"sample_id": sample_id, "reason": reason, "image_sha256": source.get("image_sha256"), "source_group_id": source.get("source_group_id"), "near_duplicate_group_id": source.get("near_duplicate_group_id"), **FLAGS})
    if Counter(item["reason"] for item in exclusions) != Counter({"direct_frozen_residual": 15, "classifier_terminal_shortfall": 9, "rag_quality_reject": 14, "refusal_delivery_unknown": 1}):
        raise DataError("residual categories do not match current protocol")
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(destination / "data.jsonl", data); write_jsonl_atomic(destination / "lineage.jsonl", lineage); write_jsonl_atomic(destination / "exclusions.jsonl", exclusions)
    isolation = {group: {str(value): sorted(line["sample_id"] for line in lineage if line.get(group) == value) for value in sorted({line.get(group) for line in lineage})} for group in ("image_sha256", "source_group_id", "near_duplicate_group_id")}
    _write_json(destination / "evaluation-isolation.json", {"schema_version": "agrinet.sft.evaluation-isolation/v1", "groups": isolation, "note": "Groups are isolation metadata only; no train/validation split is authorized.", **FLAGS})
    source_admission = {"schema_version": "agrinet.sft.cascade-admission/v1", "artifact_id": ARTIFACT_ID, "rows": len(data), "route_counts": counts, "residual_count": len(exclusions), "conservation": {"all_images": 1070, "candidate_rows": len(data), "residuals": len(exclusions), "holds": len(data) + len(exclusions) == 1070}, "inputs": {str(path.relative_to(root)): digest(path) for path in required}, **FLAGS}
    _write_json(destination / "source-admission.json", source_admission)
    stats = {"rows": len(data), "routes": counts, "strata": {key: dict(sorted(Counter(str(line["strata"].get(key)) for line in lineage).items())) for key in ("arm", "question_type", "domain", "fold", "e327_overlap")}, "tool_rows": sum(bool(row["tools"]) for row in data), "refusal_rows": counts["refusal"], **FLAGS}
    _write_json(destination / "statistics.json", stats)
    manifest = {"schema_version": "agrinet.sft.frozen/v1", "artifact_id": ARTIFACT_ID, "artifact_type": "datasets", "immutable": True, "data_sha256": digest(destination / "data.jsonl"), "lineage_sha256": digest(destination / "lineage.jsonl"), "exclusions_sha256": digest(destination / "exclusions.jsonl"), "source_admission_sha256": digest(destination / "source-admission.json"), "statistics": stats, "no_online_collection": True, **FLAGS}
    _write_json(destination / "artifact.yaml", manifest)
    (destination / "README.md").write_text(f"# {ARTIFACT_ID}\n\nImmutable English four-route SFT candidate dataset. It is not training-authorized.\n", encoding="utf-8")
    return {"artifact_dir": str(destination), "rows": len(data), "routes": counts, "residuals": len(exclusions), **FLAGS}
