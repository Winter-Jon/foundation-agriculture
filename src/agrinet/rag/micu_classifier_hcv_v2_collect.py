"""Public-boundary helpers for the Scheme-B Micu tool-policy smoke.

The live loop is intentionally kept behind v2 readiness.  This module owns
the parts that must be stable before an image can be sent to Micu: projection
of a private source row, classifier visibility, and typed local RAG execution.
"""
from __future__ import annotations

import json
import re
import argparse
from argparse import Namespace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.classifier_ledger import DeliveryUnresolved, RequestLedger
from agrinet.rag.classifier_distill import validate_contract as validate_v1_contract
from agrinet.rag.micu_classifier_hcv_v2 import _read_contract, local_rag_health, readiness
from agrinet.research.hcv.collector import image_url_content
from agrinet.research.hcv.collector import post_teacher_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read the fixed smoke source without silently accepting malformed rows."""
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"source row {line_number} must be a JSON object")
        rows.append(value)
    return rows


def _has_public_intent(output_root: Path, sample_id: str) -> bool:
    """Never replay a source image once any parent provider intent was persisted."""
    events = Path(output_root) / "parent" / "public" / sample_id / "ledger" / "events.jsonl"
    if not events.is_file():
        return False
    for raw in events.read_text(encoding="utf-8").splitlines():
        if raw.strip() and json.loads(raw).get("event") == "intent":
            return True
    return False


def _closed_local_parent(output_root: Path, source_row: dict[str, Any]) -> bool:
    """Recognize an exact already-closed local parent during safe resume."""
    path = Path(output_root) / "parent" / "public" / str(source_row["sample_id"]) / "trajectory.json"
    if not path.is_file():
        return False
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (trajectory.get("status") == "closed"
            and trajectory.get("sample_id") == source_row.get("sample_id")
            and trajectory.get("image_sha256") == source_row.get("image_sha256"))


def _closed_predecessor_parent(predecessor: Path | None, source_row: dict[str, Any]) -> bool:
    """Accept only an exact, closed predecessor trajectory by reference."""
    if predecessor is None:
        return False
    path = predecessor / "parent" / "public" / str(source_row["sample_id"]) / "trajectory.json"
    if not path.is_file():
        return False
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (trajectory.get("status") == "closed"
            and trajectory.get("sample_id") == source_row.get("sample_id")
            and trajectory.get("image_sha256") == source_row.get("image_sha256"))


class ProgressBudget:
    """Public-only cumulative RAG controller for a single trajectory."""

    def __init__(self, schedule: tuple[int, ...] = (0, 1, 3, 5)) -> None:
        if schedule != (0, 1, 3, 5):
            raise ValueError("v2 retrieval schedule is fixed at 0,1,3,5")
        self.schedule, self.limit, self.used = schedule, 0, 0
        self.seen: set[tuple[str, str, str]] = set()
        self.empty_progress = 0

    def permit(self, *, query: str, retrieval_type: str, image_sha256: str,
               rationale: str, generation_remaining: int) -> dict[str, Any]:
        if generation_remaining < 2:
            return {"allowed": False, "reason": "reserve_final_generation", "limit": self.limit, "used": self.used}
        normalized = re.sub(r"\s+", " ", query.strip().casefold())
        key = (normalized, retrieval_type, image_sha256 if retrieval_type == "visual" else "none")
        if not rationale.strip():
            return {"allowed": False, "reason": "empty_rationale", "limit": self.limit, "used": self.used}
        if key in self.seen:
            return {"allowed": False, "reason": "duplicate_query", "limit": self.limit, "used": self.used}
        if self.empty_progress >= 2:
            return {"allowed": False, "reason": "two_nonprogress_retrievals", "limit": self.limit, "used": self.used}
        if self.used >= self.limit:
            next_limit = next((value for value in self.schedule if value > self.limit), None)
            if next_limit is None:
                return {"allowed": False, "reason": "rag_budget_exhausted", "limit": self.limit, "used": self.used}
            self.limit = next_limit
        self.seen.add(key); self.used += 1
        return {"allowed": True, "limit": self.limit, "used": self.used,
                "remaining": self.limit - self.used}

    def observe(self, evidence: list[dict[str, Any]]) -> None:
        identifiers = {str(item.get("artifact_id") or "") for item in evidence if isinstance(item, dict)} - {""}
        if identifiers:
            self.empty_progress = 0
        else:
            self.empty_progress += 1


def parse_teacher_action(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize exactly one native teacher tool call or a text final."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("teacher response has no first message") from exc
    calls = message.get("tool_calls") or []
    if calls:
        if len(calls) != 1 or not isinstance(calls[0], dict):
            raise ValueError("teacher must make exactly one tool call")
        function = calls[0].get("function") or {}
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise ValueError("teacher tool arguments are not JSON") from exc
        if function.get("name") not in {"agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"} or not isinstance(arguments, dict):
            raise ValueError("teacher requested an unsupported tool")
        call_id = calls[0].get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("teacher tool call lacks a tool_call_id")
        return {"type": "tool", "name": function["name"], "arguments": arguments,
                "tool_call_id": call_id}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("teacher must return text or one tool call")
    return {"type": "final", "content": content.strip()}


class ToolState:
    """Validate the public action sequence without inferring a hidden answer."""

    def __init__(self, *, image_sha256: str, max_generations: int = 7) -> None:
        self.image_sha256 = image_sha256
        self.max_generations = max_generations
        self.generations = 0
        self.classifier_called = False
        self.classifier_expanded = False
        self.closed = False
        self.rag = ProgressBudget()

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise ValueError("trajectory is already closed")
        self.generations += 1
        if self.generations > self.max_generations:
            raise ValueError("generation budget exhausted")
        if action.get("type") == "final":
            self.closed = True
            return {"allowed": True, "event": "final", "generations": self.generations}
        if action.get("type") != "tool":
            raise ValueError("action must be final or tool")
        name, arguments = action.get("name"), action.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if name == "agrinet_classifier_predict":
            if self.classifier_called:
                raise ValueError("classifier may be called once per trajectory")
            self.classifier_called = True
            return {"allowed": True, "event": "classifier_predict", "generations": self.generations}
        if name == "agrinet_classifier_expand":
            if not self.classifier_called or self.classifier_expanded:
                raise ValueError("classifier expansion requires one prior prediction and may occur once")
            if not isinstance(arguments.get("reason"), str) or not arguments["reason"].strip():
                raise ValueError("classifier expansion requires a non-empty public reason")
            self.classifier_expanded = True
            return {"allowed": True, "event": "classifier_expand", "generations": self.generations}
        if name == "agrinet_rag_search":
            return self.rag.permit(query=str(arguments.get("query") or ""),
                                   retrieval_type=str(arguments.get("retrieval_type") or ""),
                                   image_sha256=self.image_sha256,
                                   rationale=str(arguments.get("rationale") or ""),
                                   generation_remaining=self.max_generations - self.generations)
        raise ValueError("unsupported v2 tool")


def _ledger_contract(teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    """Frozen ledger shape shared with the v1 durable request primitive."""
    # This is a transport-accounting contract, not the research/route contract.
    # It intentionally keeps the exact validator-compatible bounds.
    return {
        "schema_version": "agrinet.hcv-classifier-distill/v1",
        "teacher": {"service": "micu_slb", "model": teacher_model,
                    "service_version": "micu_slb-v1", "model_version": teacher_model,
                    "prompt_version": "classifier-hcv-v2-unified-router-v1",
                    "parameters_version": "classifier-hcv-v2-runtime-v1"},
        "dataset": {"version": "open_agri_v3", "known_classes": 107, "unknown_classes": 104,
                    "registry_sha256": "4fa6426203c64631315a2d754f3d53d6c3a7639e26c1c8617b2cbf92caf00cb8",
                    "prediction_kinds": ["fresh", "out_of_fold"], "oof_folds": 3,
                    "grouping": ["image_group_id", "source_group_id", "near_duplicate_group_id"],
                    "formal_unknown_sft": False, "final_test_selection": False},
        "candidates": {"initial_top_k": 3, "stored_top_k": 5, "expansion_requires_event": True,
                       "retrieval_outside_top5": True, "scores_are_known_probability": False,
                       "add_retrieval_and_classifier_scores": False},
        "routes": {"A": "candidate_judgment", "B": "real_tool_policy",
                   "C": "disagreement_sampling_for_A_or_B", "D": "regenerate_under_direct_or_rag_student_visibility"},
        "views": ["without_candidates", "with_candidates"],
        "stages": {"smoke": {"independent_images": 32, "per_cell": 4},
                   "exploration": {"independent_images": 160, "target_per_pattern": 16}},
        "budgets": {"rag_calls_per_trajectory": 5, "micu_requests_per_trajectory": 7,
                    "private_audits_per_trajectory": 1},
        "patterns": {f"P{i}": "runtime" for i in range(1, 11)},
        "sampling": {"ordinary_to_hard": [1, 1], "compare_equal_size_random": True,
                     "duplicate_padding": False, "target_pattern_is_private": True, "interventions_separate": True},
        "gates": {"pilot_training_eligible": False, "stable_rounds_required": 2,
                  "human_confirmation_required": True, "independent_public_private_requests": True,
                  "private_outcomes": ["accept", "reject", "human_review"],
                  "private_truth_teacher_repair": False,
                  "exclude": ["rejected", "retired", "contacted", "unknown_delivery"],
                  "ambiguous_delivery_auto_replay": False, "actual_token_mask_check_required": True},
        "evaluation": {"confidence_interval_unit": "image_group_id"},
    }


def _tool_schema(allowed_tools: set[str] | None = None) -> list[dict[str, Any]]:
    tools = [{"type": "function", "function": {"name": "agrinet_classifier_predict",
             "description": "Obtain Top-3 classifier candidates for the current image.",
             "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "agrinet_classifier_expand",
             "description": "Expand the existing candidate card to stored Top-5.",
             "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                            "required": ["reason"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "agrinet_rag_search",
             "description": "Retrieve local public AgriNet evidence.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"}, "retrieval_type": {"enum": ["visual", "semantic"]},
                 "rationale": {"type": "string"}}, "required": ["query", "retrieval_type", "rationale"],
                            "additionalProperties": False}}}]
    return [tool for tool in tools if allowed_tools is None or tool["function"]["name"] in allowed_tools]


SYSTEM_PROMPT = """You are an agricultural visual diagnostician. Use only the image, user question, and actual tool results.
You may call one listed tool at a time when it would resolve uncertainty. Classifier scores rank known-class candidates only; do not treat them as a known/unknown probability and never add them to retrieval scores.
Compare visible support and counterevidence before choosing. You may retrieve a class outside classifier Top-5. Stop once public evidence supports a conclusion, or state INSUFFICIENT_EVIDENCE if it does not.
For Option questions answer only a visible option letter or INSUFFICIENT_EVIDENCE; for Open questions answer a public class name or INSUFFICIENT_EVIDENCE."""

PATTERN_GUIDE = """P1 evidence-based correct candidate confirmation; P2 low-confidence candidate confirmation; P3 candidate reranking; P4 high-confidence candidate correction; P5 discovery outside classifier Top-5; P6 rejection of an incomplete closed-set classifier with recovery allowed from full RAG; P7 classifier/retrieval evidence conflict; P8 noisy or non-discriminative retrieval; P9 naming, alias, life-stage, symptom/disease, or granularity control; P10 justified insufficient-evidence stopping."""


_GLOBAL_MICU_LIMIT = 340
_GLOBAL_RAG_LIMIT = 200


class GlobalMicuBudget:
    """Atomic whole-goal generation accounting across independent trajectories."""

    def __init__(self, path: Path, *, limit: int | None = None) -> None:
        self.path, self.limit = Path(path), _GLOBAL_MICU_LIMIT if limit is None else limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(self, key: str) -> None:
        if not key:
            raise ValueError("global budget key is required")
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                stream.seek(0)
                events = [json.loads(line) for line in stream if line.strip()]
                seen = {str(event.get("key") or "") for event in events}
                if key in seen:
                    return
                if len(events) >= self.limit:
                    raise ValueError("global Micu budget exhausted")
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps({"key": key, "time": _stamp()}, sort_keys=True) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class GlobalRagBudget(GlobalMicuBudget):
    """Atomic whole-goal local retrieval accounting across parent and G2."""

    def __init__(self, path: Path, *, limit: int | None = None) -> None:
        super().__init__(path, limit=_GLOBAL_RAG_LIMIT if limit is None else limit)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _goal_root(output_root: Path) -> Path:
    """Resolve the shared smoke root from a parent or artifact-root caller."""
    root = Path(output_root)
    return root.parent if root.name == "parent" else root


def _generation_summary(*, sample: dict[str, Any], turn: int, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Durable public request record, deliberately excluding image bytes."""
    rendered = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            public_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    public_parts.append({"type": "image_reference", "image_sha256": sample["image_sha256"]})
                else:
                    public_parts.append(part)
            content = public_parts
        rendered.append({"role": message.get("role"), "content": content})
    return {"operation": "parent_generation", "sample_id": sample["sample_id"],
            "image_sha256": sample["image_sha256"], "turn": turn,
            "public_messages": rendered,
            "system_prompt_version": "classifier-hcv-v2-unified-router-v1"}


def _invoke_micu(request: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    env = yunwu_environment(profile="micu_slb")
    headers = {"Authorization": f"Bearer {env['YUNWU_API_KEY']}", "Content-Type": "application/json"}
    url = env["YUNWU_API_BASE_URL"].rstrip("/") + "/chat/completions"
    return post_teacher_json(url, request, headers,
                             Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0))


def public_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Return the only source fields allowed into a public trajectory.

    In particular this excludes truth, target pattern, classifier output and
    canonical IDs in option records.  The classifier may reveal candidates
    later only through ``classifier_tool_result``.
    """
    options = []
    for item in row.get("public_options") or []:
        if not isinstance(item, dict):
            raise ValueError("public option must be an object")
        options.append({"label": item.get("label"), "name": item.get("name"),
                        "name_zh": item.get("name_zh")})
    result = {
        "sample_id": row["sample_id"], "image_path": row["image_path"],
        "image_sha256": row["image_sha256"], "question": row["question"],
        "question_type": row["question_type"], "language": row["language"],
        "task_domain": row["task_domain"], "public_options": options,
    }
    return result


def normalize_final_answer(*, source_row: dict[str, Any], final: str) -> dict[str, Any]:
    """Create a stable student-side answer envelope without changing teacher text."""
    if not isinstance(final, str) or not final.strip():
        raise ValueError("final answer must be non-empty")
    value = final.strip()
    result: dict[str, Any] = {
        "schema_version": "agrinet.micu-classifier-hcv-answer/v1",
        "answer_mode": str(source_row.get("question_type") or ""),
        "raw_answer": value,
    }
    if value == "INSUFFICIENT_EVIDENCE":
        result["answer_status"] = "insufficient_evidence"
        return result
    if result["answer_mode"] == "open":
        result.update({"answer_status": "answered", "selected_class_name": value})
        return result
    if result["answer_mode"] != "option":
        raise ValueError("answer mode must be open or option")
    label = value.upper()
    options = source_row.get("public_options") or []
    matches = [item for item in options if str(item.get("label") or "").upper() == label]
    if len(matches) != 1:
        raise ValueError("option final must be one visible option label")
    selected = matches[0]
    result.update({
        "answer_status": "answered",
        "selected_option": label,
        "selected_class_name": selected.get("name"),
        "selected_class_name_zh": selected.get("name_zh"),
        "selected_class_code": selected.get("code"),
    })
    return result


def teacher_first_request(public: dict[str, Any], *, system_prompt: str, max_tokens: int,
                          teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    if not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be within the v2 contract")
    image = Path(public["image_path"])
    return {
        "model": teacher_model, "temperature": 0.0, "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": public["question"]}, image_url_content(image),
            ]},
        ],
    }


def _append_native_tool_exchange(messages: list[dict[str, Any]], raw: dict[str, Any],
                                 action: dict[str, Any], tool: dict[str, Any]) -> None:
    """Preserve the provider's native tool-call protocol for the next turn."""
    try:
        message = raw["choices"][0]["message"]
        calls = message["tool_calls"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("tool action has no native assistant tool-call message") from exc
    if not isinstance(calls, list) or len(calls) != 1 or action.get("tool_call_id") != calls[0].get("id"):
        raise ValueError("native tool-call identity changed unexpectedly")
    messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": action["tool_call_id"],
                     "content": json.dumps(tool, ensure_ascii=False)})


def _append_reconstructed_tool_exchange(messages: list[dict[str, Any]], action: dict[str, Any],
                                        tool: dict[str, Any]) -> None:
    """Rebuild an auditable native exchange from a stored public trace."""
    call_id = action.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("stored tool action lacks a tool_call_id")
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": action.get("name"),
                     "arguments": json.dumps(action.get("arguments"), ensure_ascii=False)},
    }]})
    messages.append({"role": "tool", "tool_call_id": call_id,
                     "content": json.dumps(tool, ensure_ascii=False)})


def classifier_tool_result(row: dict[str, Any], *, expand: bool = False) -> dict[str, Any]:
    """Reveal stored audited candidates only when the classifier tool is called."""
    prediction = row.get("prediction")
    if not isinstance(prediction, dict) or prediction.get("kind") not in {"out_of_fold", "p6_class_holdout"}:
        raise ValueError("classifier tool requires an audited OOF or P6 prediction")
    limit = 5 if expand else 3
    top5 = prediction.get("top5")
    if not isinstance(top5, list) or len(top5) != 5:
        raise ValueError("classifier tool requires stored Top-5")
    reference = {
        key: prediction[key] for key in ("kind", "checkpoint_sha256", "label_map_sha256",
                                         "training_manifest_sha256", "p6_group")
        if key in prediction
    }
    if prediction.get("kind") == "out_of_fold":
        reference.update({"folds": prediction["folds"], "held_out_fold": prediction["held_out_fold"]})
    return {
        "tool": "agrinet_classifier_expand" if expand else "agrinet_classifier_predict",
        "prediction_reference": reference,
        "candidates": [{"rank": index, "name": item.get("name"), "name_zh": item.get("name_zh"),
                        "score": item["score"]}
                       for index, item in enumerate(top5[:limit], 1)],
    }


def execute_rag(endpoint: str, public: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Call only the local typed RAG HTTP surface and retain its raw response."""
    retrieval_type = arguments.get("retrieval_type")
    if retrieval_type not in {"visual", "semantic", "balanced"}:
        raise ValueError("RAG exposes visual, semantic, and balanced retrieval")
    rationale = arguments.get("rationale")
    query = arguments.get("query")
    if not isinstance(rationale, str) or not rationale.strip() or not isinstance(query, str) or not query.strip():
        raise ValueError("RAG requires non-empty query and rationale")
    top_k=arguments.get("top_k",3)
    if not isinstance(top_k,int) or not 1 <= top_k <= 10:
        raise ValueError("RAG top_k must be an integer from 1 to 10")
    body: dict[str, Any] = {"retrieval_type": retrieval_type, "text": query, "top_k": top_k}
    if retrieval_type in {"visual","balanced"}:
        body["image_path"] = public["image_path"]
    for key in ("ranker","text_weight","image_weight","sparse_weight"):
        if key in arguments: body[key]=arguments[key]
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/search", encoded,
                                     {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local RAG request failed: {type(exc).__name__}") from exc
    if raw.get("schema_version") != "agrinet.rag.search/v1" or not isinstance(raw.get("evidence"), list):
        raise ValueError("local RAG returned an invalid typed response")
    for item in raw["evidence"]:
        score = item.get("score") if isinstance(item, dict) else None
        if not isinstance(score, (float, int)) or isinstance(score, bool):
            raise ValueError("local RAG returned an invalid typed response")
        item["score"] = float(Decimal(str(score)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    # This is a public projection of the typed response, not a model guess.
    # Later RAG protocols may require final answers to quote one of these
    # service-returned standard names exactly.
    returned_standard_class_names=[]
    for item in raw["evidence"]:
        metadata=item.get("metadata") if isinstance(item, dict) else None
        name=metadata.get("english_name") if isinstance(metadata, dict) else None
        # Preserve one public class-name entry per returned rank slot.  Visual
        # Top-3 protocols bind R1/R2/R3 to response positions, including when
        # two backing artifacts share a canonical class name.
        if isinstance(name, str) and name.strip():
            returned_standard_class_names.append(name.strip())
    if retrieval_type == "visual" and top_k == 3 and len(returned_standard_class_names) != 3:
        raise ValueError("fixed visual Top-3 retrieval must return exactly three named slots")
    public_arguments={"query":query,"retrieval_type":retrieval_type,"top_k":top_k,"rationale":rationale}
    for key in ("ranker","text_weight","image_weight","sparse_weight"):
        if key in arguments: public_arguments[key]=arguments[key]
    return {"tool": "agrinet_rag_search", "arguments": public_arguments, "raw_response": raw,
            "returned_standard_class_names": returned_standard_class_names}


def run_parent(*, source_row: dict[str, Any], output_root: Path, rag_endpoint: str,
               max_tokens: int = 8192, timeout: int = 180,
               invoke_teacher: Any | None = None, teacher_model: str = "gpt-5.6-terra",
               temperature: float = 0.0, allowed_tools: set[str] | None = None,
               campaign_root: Path | None = None, budget_key_prefix: str | None = None) -> dict[str, Any]:
    """Run one v2 public parent trajectory, retaining every request outcome."""
    validate_v1_contract(_ledger_contract(teacher_model))
    public = public_sample(source_row)
    directory = Path(output_root) / "public" / public["sample_id"]
    budget_root = Path(campaign_root) if campaign_root is not None else _goal_root(Path(output_root))
    global_budget = GlobalMicuBudget(budget_root / "global_micu_events.jsonl")
    ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                           image_group_id=source_row["image_group_id"], view="without_candidates")
    initial = teacher_first_request(public, system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens, teacher_model=teacher_model)
    messages = initial["messages"]
    state = ToolState(image_sha256=public["image_sha256"])
    trace: list[dict[str, Any]] = []
    teacher = invoke_teacher or (lambda request: _invoke_micu(request, timeout=timeout))
    for turn in range(1, 8):
        request = {"model": teacher_model, "temperature": temperature, "top_p": 1.0,
                   "max_tokens": max_tokens, "messages": messages}
        # A Direct rung must be a genuinely tool-free provider request, rather
        # than an empty tools list with an ambiguous tool-choice directive.
        if allowed_tools != set():
            request.update({"tools": _tool_schema(allowed_tools), "tool_choice": "auto"})
        prefix = budget_key_prefix or f"parent:{public['sample_id']}"
        global_budget.reserve(f"{prefix}:generation:{turn}")
        raw = ledger.call("generation", f"generation-{turn}",
                          _generation_summary(sample=public, turn=turn, messages=messages),
                          lambda _summary: teacher(request))
        action = parse_teacher_action(raw)
        if action["type"] == "tool" and allowed_tools is not None and action["name"] not in allowed_tools:
            raise ValueError(f"tool {action['name']} is unavailable in this route")
        transition = state.act(action)
        trace.append({"time": _stamp(), "turn": turn, "action": action, "controller": transition})
        if action["type"] == "final":
            normalized = normalize_final_answer(source_row=source_row, final=action["content"])
            result = {"schema_version": "agrinet.micu-classifier-hcv-v2-parent/v1",
                      "sample_id": public["sample_id"], "image_sha256": public["image_sha256"],
                      "status": "closed", "training_eligible": False,
                      "public_context": {key: public[key] for key in ("question", "question_type", "language", "task_domain", "public_options")},
                      "final": action["content"], "normalized_final": normalized, "trace": trace}
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "trajectory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result
        if transition.get("allowed"):
            if action["name"] == "agrinet_classifier_predict":
                tool = classifier_tool_result(source_row)
            elif action["name"] == "agrinet_classifier_expand":
                tool = classifier_tool_result(source_row, expand=True)
            else:
                GlobalRagBudget(budget_root / "global_rag_events.jsonl").reserve(
                    f"{prefix}:rag:{turn}")
                tool = ledger.call("rag", f"rag-{turn}",
                                   {"operation": "rag", "sample_id": public["sample_id"],
                                    "turn": turn, "arguments": action["arguments"]},
                                   lambda _summary: execute_rag(rag_endpoint, public, action["arguments"]))
                state.rag.observe(tool["raw_response"].get("evidence") or [])
        else:
            tool = {"tool": "controller", "result": transition}
        trace.append({"time": _stamp(), "turn": turn, "tool": tool})
        _append_native_tool_exchange(messages, raw, action, tool)
    raise ValueError("trajectory exhausted generation budget without final answer")


def private_audit_payload(*, source_row: dict[str, Any], parent: dict[str, Any],
                          teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    """Build an isolated audit request; the result never returns to generation."""
    private = source_row.get("private")
    if not isinstance(private, dict) or not isinstance(private.get("truth_code"), str):
        raise ValueError("private audit needs a local truth sidecar")
    return {
        "model": teacher_model, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
        "messages": [{"role": "system", "content":
                      "You are an isolated private auditor. Evaluate whether the public trajectory is supported by its image and public tool evidence. Return JSON only with decision (accept, reject, or human_review), concise reason, and exactly one primary_pattern (P1 through P10) describing the dominant behavior actually observed. Do not propose a corrected answer. Pattern definitions: " + PATTERN_GUIDE},
                     {"role": "user", "content": [
                         {"type": "text", "text": json.dumps({
                             "image_sha256": source_row["image_sha256"],
                             "private_truth_code": private["truth_code"],
                             "private_truth_name": private.get("truth_name"),
                             "private_correct_option": private.get("correct_option"),
                             "classifier_condition": {
                                 "kind": (source_row.get("prediction") or {}).get("kind"),
                                 "excluded_supervised_codes": (source_row.get("prediction") or {}).get("excluded_supervised_codes") or [],
                                 "simulated_unknown": private.get("simulated_unknown"),
                             },
                             "public_question": source_row.get("question"),
                             "public_options": source_row.get("public_options") or [],
                             "public_parent": parent,
                         }, ensure_ascii=False)},
                         image_url_content(Path(source_row["image_path"])),
                     ]}],
        "response_format": {"type": "json_object"},
    }


def parse_private_audit(response: dict[str, Any]) -> dict[str, str]:
    try:
        content = response["choices"][0]["message"]["content"]
        payload = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("private audit response is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("decision") not in {"accept", "reject", "human_review"}:
        raise ValueError("private audit decision is invalid")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("private audit reason is missing")
    pattern = payload.get("primary_pattern")
    if pattern is not None and pattern not in {f"P{i}" for i in range(1, 11)}:
        raise ValueError("private audit primary pattern is invalid")
    result = {"decision": payload["decision"], "reason": reason.strip()}
    if pattern is not None:
        result["primary_pattern"] = pattern
    return result


def g1_rewrite_payload(parent: dict[str, Any], *, teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    """Ask Micu to complete a natural assistant response from the full public trace."""
    trace = parent.get("trace")
    if not isinstance(trace, list) or parent.get("status") != "closed":
        raise ValueError("G1 requires a closed parent trajectory")
    return {"model": teacher_model, "temperature": 0.0, "top_p": 1.0, "max_tokens": 4096,
            "messages": [{"role": "system", "content":
                          "Complete one natural assistant response from the full public trajectory. Rewrite the reasoning as a coherent concise explanation suitable for a small vision-language model, then preserve the original final answer exactly. Use only facts in the supplied trajectory; do not add tools or evidence. Return JSON with assistant_reasoning and final."},
                         {"role": "user", "content": json.dumps({
                             "public_context": parent.get("public_context"),
                             "public_trace": trace, "original_final": parent.get("final")},
                             ensure_ascii=False)}],
            "response_format": {"type": "json_object"}}


def parse_g1_rewrite(response: dict[str, Any], *, original_final: str) -> dict[str, str]:
    """Validate model-written reasoning while freezing the parent's conclusion."""
    try:
        content = response["choices"][0]["message"]["content"]
        payload = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("G1 response is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("G1 response must be an object")
    reasoning, final = payload.get("assistant_reasoning"), payload.get("final")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("G1 response lacks assistant reasoning")
    if final != original_final:
        raise ValueError("G1 may not change the parent final answer")
    return {"assistant_reasoning": reasoning.strip(), "final": final}


def _cell(row: dict[str, Any]) -> str:
    return "-".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))


def _private_audit_summary(source_row: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Private durable ledger payload with an image reference instead of bytes."""
    private = source_row.get("private") or {}
    return {"operation": "private_audit", "sample_id": source_row["sample_id"],
            "image_sha256": source_row["image_sha256"],
            "private_truth_code": private.get("truth_code"), "public_parent": parent}


def run_private_audit(*, source_row: dict[str, Any], parent: dict[str, Any], output_root: Path,
                      scope: str = "parent", timeout: int = 180, invoke_teacher: Any | None = None,
                      require_primary_pattern: bool = False, teacher_model: str = "gpt-5.6-terra",
                      campaign_root: Path | None = None, budget_key_prefix: str | None = None) -> dict[str, Any]:
    """Run the one allowed isolated audit without exposing its outcome to generation."""
    if scope not in {"parent", "g1", "g2", "rewrite"}:
        raise ValueError("private audit scope is invalid")
    directory = Path(output_root) / "private" / source_row["sample_id"] / scope
    ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                           image_group_id=source_row["image_group_id"], view="without_candidates",
                           channel="private")
    request = private_audit_payload(source_row=source_row, parent=parent, teacher_model=teacher_model)
    teacher = invoke_teacher or (lambda payload: _invoke_micu(payload, timeout=timeout))
    budget_root = Path(campaign_root) if campaign_root is not None else _goal_root(Path(output_root))
    GlobalMicuBudget(budget_root / "global_micu_events.jsonl").reserve(
        f"{budget_key_prefix or f'{scope}:{source_row["sample_id"]}'}:private-audit:1")
    raw = ledger.call("audit", "private-audit-1", _private_audit_summary(source_row, parent),
                      lambda _summary: teacher(request))
    verdict = parse_private_audit(raw)
    if require_primary_pattern and verdict.get("primary_pattern") not in {f"P{i}" for i in range(1, 11)}:
        raise ValueError("E2 parent audit requires exactly one observed primary pattern")
    result = {"schema_version": "agrinet.micu-classifier-hcv-v2-private-audit/v1",
              "sample_id": source_row["sample_id"], "image_sha256": source_row["image_sha256"],
              "scope": scope, "status": verdict["decision"], "reason": verdict["reason"],
              "primary_pattern": verdict.get("primary_pattern"),
              "training_eligible": False}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def select_derivation_parents(*, source_rows: list[dict[str, Any]], output_root: Path,
                              max_total: int = 8, max_per_pattern: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose one parent per cell for smoke, or capped observed patterns for E2."""
    selected: list[dict[str, Any]] = []; exclusions: list[dict[str, Any]] = []
    pattern_counts: dict[str, int] = {}
    cell_counts: dict[str, int] = {}
    boundary_path = Path(output_root) / "public_boundary_exclusions.json"
    boundary_payload = json.loads(boundary_path.read_text(encoding="utf-8")) if boundary_path.is_file() else {}
    boundary_excluded = set(boundary_payload.get("sample_ids") or [])
    for row in sorted(source_rows, key=lambda item: str(item["sample_id"])):
            cell = _cell(row)
            if row["sample_id"] in boundary_excluded:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell,
                                   "reason": "public_boundary_excluded"}); continue
            base = Path(output_root) / "parent" / "public" / row["sample_id"]
            parent_path = base / "trajectory.json"
            audit_path = Path(output_root) / "private" / row["sample_id"] / "parent" / "audit.json"
            if not parent_path.is_file() or not audit_path.is_file():
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "reason": "missing_parent_or_audit"}); continue
            parent, audit = json.loads(parent_path.read_text(encoding="utf-8")), json.loads(audit_path.read_text(encoding="utf-8"))
            if parent.get("status") != "closed" or audit.get("status") != "accept":
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "reason": "parent_not_accepted"}); continue
            pattern = audit.get("primary_pattern")
            if max_per_pattern is not None and pattern not in {f"P{i}" for i in range(1, 11)}:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "reason": "missing_primary_pattern"}); continue
            try:
                prefix = g2_prefix(parent)
            except ValueError:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "reason": "no_completed_tool_prefix"}); continue
            if len(selected) >= max_total:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "primary_pattern": pattern, "reason": "global_derivation_cap"}); continue
            if max_per_pattern is None and cell_counts.get(cell, 0) >= 1:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "reason": "fixed_order_not_selected"}); continue
            if max_per_pattern is not None and pattern_counts.get(str(pattern), 0) >= max_per_pattern:
                exclusions.append({"sample_id": row["sample_id"], "cell": cell, "primary_pattern": pattern, "reason": "pattern_derivation_cap"}); continue
            selected.append({"source_row": row, "parent": parent, "prefix": prefix,
                             "cell": cell, "primary_pattern": pattern})
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if pattern is not None:
                pattern_counts[str(pattern)] = pattern_counts.get(str(pattern), 0) + 1
    return selected, exclusions


def g2_prefix(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Select a deterministic public prefix after first actual tool result."""
    trace = parent.get("trace")
    if not isinstance(trace, list):
        raise ValueError("G2 requires a parent trace")
    for index, event in enumerate(trace):
        tool = event.get("tool") if isinstance(event, dict) else None
        if isinstance(tool, dict) and tool.get("tool") in {
            "agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"
        }:
            return trace[:index + 1]
    raise ValueError("G2 skips parents without a completed tool result")


def _prefix_messages(source_row: dict[str, Any], prefix: list[dict[str, Any]], *, max_tokens: int) -> tuple[list[dict[str, Any]], ToolState]:
    """Reconstruct G2's visible state and controller state from its public prefix."""
    public = public_sample(source_row)
    messages = teacher_first_request(public, system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens)["messages"]
    state = ToolState(image_sha256=public["image_sha256"])
    pending: dict[str, Any] | None = None
    for event in prefix:
        if not isinstance(event, dict):
            raise ValueError("G2 prefix event is invalid")
        if isinstance(event.get("action"), dict):
            if pending is not None:
                raise ValueError("G2 prefix has two actions without a result")
            pending = event["action"]
            state.act(pending)
        elif isinstance(event.get("tool"), dict):
            if pending is None:
                raise ValueError("G2 prefix has tool result without action")
            _append_reconstructed_tool_exchange(messages, pending, event["tool"])
            if pending.get("name") == "agrinet_rag_search":
                state.rag.observe((event["tool"].get("raw_response") or {}).get("evidence") or [])
            pending = None
        else:
            raise ValueError("G2 prefix event is neither action nor tool result")
    if pending is not None or state.closed:
        raise ValueError("G2 prefix is not an open post-tool state")
    return messages, state


def _derivation_summary(*, kind: str, sample_id: str, image_sha256: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist only public, image-byte-free derivation request context."""
    return {"operation": kind, "sample_id": sample_id, "image_sha256": image_sha256,
            "payload": payload}


def run_g1(*, source_row: dict[str, Any], parent: dict[str, Any], output_root: Path, timeout: int = 180,
           invoke_teacher: Any | None = None, teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    """One public expression rewrite with fixed parent facts and durable lineage."""
    directory = Path(output_root) / "derivations" / "g1" / source_row["sample_id"]
    ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                           image_group_id=source_row["image_group_id"], view="without_candidates")
    request = g1_rewrite_payload(parent, teacher_model=teacher_model)
    teacher = invoke_teacher or (lambda payload: _invoke_micu(payload, timeout=timeout))
    key = f"g1:{source_row['sample_id']}:generation:1"
    GlobalMicuBudget(_goal_root(Path(output_root)) / "global_micu_events.jsonl").reserve(key)
    raw = ledger.call("generation", "g1-rewrite-1",
                      _derivation_summary(kind="g1_rewrite", sample_id=source_row["sample_id"],
                                          image_sha256=source_row["image_sha256"], payload={"parent": parent}),
                      lambda _summary: teacher(request))
    rewrite = parse_g1_rewrite(raw, original_final=str(parent["final"]))
    result = {"schema_version": "agrinet.micu-classifier-hcv-v2-g1/v1", "sample_id": source_row["sample_id"],
              "parent_sample_id": source_row["sample_id"], "status": "closed",
              "public_context": parent.get("public_context"), "trace": parent["trace"],
              "assistant_reasoning": rewrite["assistant_reasoning"], "final": rewrite["final"],
              "normalized_final": normalize_final_answer(source_row=source_row, final=rewrite["final"]),
              "training_eligible": False}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "derivation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_g2(*, source_row: dict[str, Any], parent: dict[str, Any], output_root: Path, rag_endpoint: str,
           max_tokens: int = 8192, timeout: int = 180, invoke_teacher: Any | None = None,
           teacher_model: str = "gpt-5.6-terra") -> dict[str, Any]:
    """Continue from the fixed prefix; every new tool result is executed afresh."""
    prefix = g2_prefix(parent)
    public = public_sample(source_row)
    messages, state = _prefix_messages(source_row, prefix, max_tokens=max_tokens)
    directory = Path(output_root) / "derivations" / "g2" / public["sample_id"]
    ledger = RequestLedger(directory / "ledger", contract=_ledger_contract(teacher_model),
                           image_group_id=source_row["image_group_id"], view="without_candidates")
    teacher = invoke_teacher or (lambda payload: _invoke_micu(payload, timeout=timeout))
    trace: list[dict[str, Any]] = []
    while state.generations < state.max_generations:
        turn = state.generations + 1
        request = {"model": teacher_model, "temperature": 0.0, "top_p": 1.0,
                   "max_tokens": max_tokens, "messages": messages, "tools": _tool_schema(),
                   "tool_choice": "auto"}
        key = f"g2:{public['sample_id']}:generation:{turn}"
        GlobalMicuBudget(_goal_root(Path(output_root)) / "global_micu_events.jsonl").reserve(key)
        raw = ledger.call("generation", f"generation-{turn}",
                          _generation_summary(sample=public, turn=turn, messages=messages),
                          lambda _summary: teacher(request))
        action = parse_teacher_action(raw)
        transition = state.act(action)
        trace.append({"time": _stamp(), "turn": turn, "action": action, "controller": transition})
        if action["type"] == "final":
            normalized = normalize_final_answer(source_row=source_row, final=action["content"])
            result = {"schema_version": "agrinet.micu-classifier-hcv-v2-g2/v1",
                      "sample_id": public["sample_id"], "parent_sample_id": public["sample_id"],
                      "status": "closed", "prefix": prefix, "continuation": trace, "trace": prefix + trace,
                      "public_context": {key: public[key] for key in ("question", "question_type", "language", "task_domain", "public_options")},
                      "final": action["content"], "normalized_final": normalized,
                      "training_eligible": False}
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "derivation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result
        if transition.get("allowed"):
            if action["name"] == "agrinet_classifier_predict":
                tool = classifier_tool_result(source_row)
            elif action["name"] == "agrinet_classifier_expand":
                tool = classifier_tool_result(source_row, expand=True)
            else:
                GlobalRagBudget(_goal_root(Path(output_root)) / "global_rag_events.jsonl").reserve(
                    f"g2:{public['sample_id']}:rag:{turn}")
                tool = ledger.call("rag", f"rag-{turn}",
                                   {"operation": "rag", "sample_id": public["sample_id"],
                                    "turn": turn, "arguments": action["arguments"]},
                                   lambda _summary: execute_rag(rag_endpoint, public, action["arguments"]))
                state.rag.observe(tool["raw_response"].get("evidence") or [])
        else:
            tool = {"tool": "controller", "result": transition}
        trace.append({"time": _stamp(), "turn": turn, "tool": tool})
        _append_native_tool_exchange(messages, raw, action, tool)
    raise ValueError("G2 exhausted inherited generation budget without final answer")


def run_derivation_batch(*, source_rows: list[dict[str, Any]], output_root: Path, rag_endpoint: str,
                         max_tokens: int = 8192, timeout: int = 180,
                         max_images: int = 8, max_per_pattern: int | None = None,
                         require_primary_pattern: bool = False, max_concurrency: int = 1) -> dict[str, Any]:
    """Audit closed parents, then make one fixed-order G1/G2 pair per eligible cell."""
    root = Path(output_root)
    statuses: list[dict[str, Any]] = []
    by_id = {str(row["sample_id"]): row for row in source_rows}
    boundary_path = root / "public_boundary_exclusions.json"
    boundary_payload = json.loads(boundary_path.read_text(encoding="utf-8")) if boundary_path.is_file() else {}
    boundary_excluded = set(boundary_payload.get("sample_ids") or [])
    pending_audits: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sample_id, row in sorted(by_id.items()):
        if sample_id in boundary_excluded:
            statuses.append({"sample_id": sample_id, "stage": "parent_audit",
                             "status": "skipped", "reason": "public_boundary_excluded"})
            continue
        parent_path = root / "parent" / "public" / sample_id / "trajectory.json"
        if not parent_path.is_file():
            statuses.append({"sample_id": sample_id, "stage": "parent_audit", "status": "skipped", "reason": "parent_not_closed"})
            continue
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        existing_audit = root / "private" / sample_id / "parent" / "audit.json"
        if existing_audit.is_file():
            audit = json.loads(existing_audit.read_text(encoding="utf-8"))
            if (audit.get("sample_id") == sample_id and audit.get("scope") == "parent"
                    and audit.get("status") in {"accept", "reject", "human_review"}):
                statuses.append({"sample_id": sample_id, "stage": "parent_audit",
                                 "status": audit["status"], "referenced": True})
                continue
        pending_audits.append((sample_id, row, parent))

    def audit_one(item: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        sample_id, row, parent = item
        try:
            audit = run_private_audit(source_row=row, parent=parent, output_root=root, scope="parent",
                                      timeout=timeout, require_primary_pattern=require_primary_pattern)
            return {"sample_id": sample_id, "stage": "parent_audit", "status": audit["status"]}
        except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
            return {"sample_id": sample_id, "stage": "parent_audit", "status": "not_completed",
                    "reason": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        statuses.extend(pool.map(audit_one, pending_audits))
    selected, exclusions = select_derivation_parents(
        source_rows=source_rows, output_root=root, max_total=max_images,
        max_per_pattern=max_per_pattern)
    for item in selected:
        row, parent, cell = item["source_row"], item["parent"], item["cell"]
        sample_id = row["sample_id"]
        for stage, runner in (("g1", lambda: run_g1(source_row=row, parent=parent, output_root=root, timeout=timeout)),
                              ("g2", lambda: run_g2(source_row=row, parent=parent, output_root=root, rag_endpoint=rag_endpoint, max_tokens=max_tokens, timeout=timeout))):
            try:
                result = runner()
                audit = run_private_audit(source_row=row, parent=result, output_root=root, scope=stage, timeout=timeout)
                statuses.append({"sample_id": sample_id, "cell": cell, "stage": stage,
                                 "status": result["status"], "audit": audit["status"]})
            except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
                statuses.append({"sample_id": sample_id, "cell": cell, "stage": stage,
                                 "status": "not_completed", "reason": type(exc).__name__})
    result = {"schema_version": "agrinet.micu-classifier-hcv-v2-derivation-batch/v1",
              "selected": [{"sample_id": item["source_row"]["sample_id"], "cell": item["cell"],
                            "primary_pattern": item.get("primary_pattern")} for item in selected],
              "selection_exclusions": exclusions, "statuses": statuses, "training_eligible": False}
    destination = root / "derivations" / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    global _GLOBAL_MICU_LIMIT, _GLOBAL_RAG_LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--phase", choices=("parent", "derivations"), default="parent")
    args = parser.parse_args(argv)
    contract = _read_contract(args.contract)
    _GLOBAL_MICU_LIMIT = int(contract["budgets"]["total_micu"])
    _GLOBAL_RAG_LIMIT = int(contract["budgets"]["rag_total"])
    report = readiness(contract_path=args.contract, dataset_root=args.dataset_root,
                       rag_health=local_rag_health(str(contract["retrieval"]["endpoint"])))
    if not report.get("ready_for_live_collection"):
        raise RuntimeError("v2 readiness failed; parent collection will not contact Micu")
    root = args.contract.parents[3]
    source = root / contract["data"]["source"]
    rows = _read_jsonl(source)
    expected_rows = int(contract["data"]["independent_images"])
    limit = expected_rows if args.limit is None else args.limit
    if limit != expected_rows or len(rows) != expected_rows:
        raise ValueError(f"collection requires the complete {expected_rows}-row source")
    if args.phase == "derivations":
        exploration = str(contract["data"].get("stage") or "smoke") == "exploration"
        result = run_derivation_batch(source_rows=rows, output_root=args.output_root,
                                      rag_endpoint=str(contract["retrieval"]["endpoint"]),
                                      max_tokens=int(contract["runtime"]["micu_max_output_tokens"]),
                                      timeout=int(contract["runtime"]["micu_timeout_seconds"]),
                                      max_images=int(contract["routing"]["g1_g2_max_images"]),
                                      max_per_pattern=2 if exploration else None,
                                      require_primary_pattern=exploration,
                                      max_concurrency=int(contract["runtime"].get("micu_max_concurrency", 1)))
        print(json.dumps({"selected": len(result["selected"]), "statuses": len(result["statuses"])}, ensure_ascii=False))
        return 0
    output = args.output_root / "parent"
    predecessor_value = contract.get("data", {}).get("predecessor")
    predecessor = root / predecessor_value if isinstance(predecessor_value, str) and predecessor_value else None
    statuses: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    for row in rows:
        if _closed_local_parent(args.output_root, row):
            statuses.append({"sample_id": row["sample_id"], "status": "closed_existing",
                             "reason": "closed_local_trajectory"})
            continue
        if _closed_predecessor_parent(predecessor, row):
            statuses.append({"sample_id": row["sample_id"], "status": "closed_referenced",
                             "reason": "closed_predecessor_trajectory"})
            continue
        if _has_public_intent(args.output_root, str(row["sample_id"])):
            statuses.append({"sample_id": row["sample_id"], "status": "unresolved_not_replayed",
                             "reason": "existing_public_intent"})
            continue
        pending_rows.append(row)

    def collect_one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            result = run_parent(source_row=row, output_root=output,
                                rag_endpoint=str(contract["retrieval"]["endpoint"]),
                                max_tokens=int(contract["runtime"]["micu_max_output_tokens"]),
                                timeout=int(contract["runtime"]["micu_timeout_seconds"]))
            return {"sample_id": row["sample_id"], "status": result["status"]}
        except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
            return {"sample_id": row["sample_id"], "status": "not_closed",
                    "reason": type(exc).__name__}

    concurrency = int(contract["runtime"].get("micu_max_concurrency", 1))
    for start in range(0, len(pending_rows), concurrency):
        batch = pending_rows[start:start + concurrency]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(collect_one, batch))
        statuses.extend(results)
        failures = [item.get("reason") for item in results if item["status"] == "not_closed"]
        if (len(failures) >= 2 and len(set(failures)) == 1
                and failures[0] != "DeliveryUnresolved"):
            statuses.append({"sample_id": None, "status": "batch_stopped",
                             "reason": f"systemic_{failures[0]}"})
            break
    summary = {"schema_version": "agrinet.micu-classifier-hcv-v2-parent-batch/v1",
               "rows": len(rows), "statuses": statuses, "training_eligible": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "closed": sum(item["status"] == "closed" for item in statuses)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
