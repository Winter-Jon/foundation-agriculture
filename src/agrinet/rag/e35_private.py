"""Isolated E3.5 audit and rewrite adapters.

Private prompts may see the truth sidecar, but their text is never returned to
the public controller.  Public rewrite prompts receive only the closed public
trajectory and are independently audited before export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agrinet.research.hcv.collector import image_url_content
from agrinet.rag.e35_transport import transport_image

JsonCaller = Callable[[dict[str, Any]], dict[str, Any]]
_MAX_PRIVATE_TRAJECTORY_CHARS = 24000


def is_rag_protocol_witness(row: dict[str, Any]) -> bool:
    """Return the private-only audit coverage designation, if present.

    This is deliberately read only by the isolated auditor.  It is an audit
    protocol control, never a teacher instruction or a public quality label.
    """
    protocol = (row.get("private") or {}).get("audit_protocol")
    return isinstance(protocol, dict) and protocol.get("rag_witness") is True

def _content(response: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = response["choices"][0]["message"]["content"]
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("private response must be a JSON object") from exc
    if not isinstance(value, dict): raise ValueError("private response must be a JSON object")
    return value


def private_trajectory_projection(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return public evidence without serializing image bytes as text.

    The isolated auditor receives the image exactly once as a distinct multimodal
    content item.  The public conversation must retain tool protocol state but
    never duplicate a data URL or a local image path inside JSON text.
    """
    if not isinstance(trajectory, dict):
        raise ValueError("private audit trajectory is invalid")
    projected = {key: trajectory.get(key) for key in ("route", "answer", "tool_calls", "tool_trace")}
    messages = trajectory.get("messages")
    if isinstance(messages, list):
        safe_messages = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("private audit trajectory message is invalid")
            content = message.get("content")
            if isinstance(content, list):
                content = [{"type": "image_reference", "image_sha256": "bound"}
                           if isinstance(item, dict) and item.get("type") == "image_url"
                           else item for item in content]
            safe = {"role": message.get("role"), "content": content}
            if "tool_calls" in message: safe["tool_calls"] = message["tool_calls"]
            if "tool_call_id" in message: safe["tool_call_id"] = message["tool_call_id"]
            safe_messages.append(safe)
        projected["messages"] = safe_messages
    rendered = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    if "data:image" in rendered or len(rendered) > _MAX_PRIVATE_TRAJECTORY_CHARS:
        raise ValueError("private audit trajectory projection exceeds public text limit")
    return projected

def private_parent_request(row: dict[str, Any], trajectory: dict[str, Any], *, model: str, transport_max_side: int = 0) -> dict[str, Any]:
    private = row.get("private")
    image_path = row.get("image_path")
    if (not isinstance(private, dict) or not isinstance(private.get("truth_code"), str) or
            not isinstance(private.get("truth_name"), str) or not private["truth_name"].strip() or
            not isinstance(image_path, str)):
        raise ValueError("private parent audit lacks frozen truth/image binding")
    witness = is_rag_protocol_witness(row)
    image, _ = transport_image(Path(image_path), max_side=transport_max_side) if transport_max_side else (image_url_content(Path(image_path)), {"version": "original"})
    public_trajectory = private_trajectory_projection(trajectory)
    protocol_name = {"agrinet.e39-hcv-cascade/v1": "E3.9", "agrinet.e310-hcv-cascade/v1": "E3.10", "agrinet.e311-hcv-cascade/v1": "E3.11", "agrinet.e312-hcv-cascade/v1": "E3.12", "agrinet.e313-hcv-cascade/v1": "E3.13", "agrinet.e314-hcv-cascade/v1": "E3.14", "agrinet.e315-option-format-repair/v1": "E3.15", "agrinet.e316-rag-discriminator/v1": "E3.16", "agrinet.e316-rag-discriminator-canary/v1": "E3.16 canary", "agrinet.e317-all-unknown-rag-audit/v1": "E3.17", "agrinet.e318-all-unknown-512-rag-audit/v1": "E3.18", "agrinet.e319-rag-closure-audit/v1": "E3.19"}.get(row.get("e39_protocol"))
    e39_rule = (f" {protocol_name}: reject unless the public final is legal Hermes and its HCV comparison is grounded in visible facts and actual public tools. For an {protocol_name} Option final in `class name — letter` form, verify both pieces against the public choices and supplied correct letter. "
                f"For an {protocol_name} private route-coverage designation, reject every terminal before that designated route; at the designated route apply normal truth/evidence/HCV rules only. " if protocol_name else "")
    return {"model": model, "temperature": 0.0, "max_tokens": 1024, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "You are an isolated private auditor. Return JSON only: decision (accept or reject), reason, requires_rag (boolean), rag_evidence_insufficient (boolean). Check the final against the supplied private canonical truth: Open must name that canonical class; Option must choose the supplied correct letter. Accept only a truth-correct final supported by the image and actual public tool evidence. Set requires_rag=true only when a rejected Classifier terminal has a specific unresolved public visual discriminator that actual retrieval could resolve; otherwise false. For INSUFFICIENT_EVIDENCE, accept only after an actual RAG tool response is insufficient for any specific conclusion. If private audit_protocol.rag_witness is true, reject Direct and Classifier terminals solely as an audit coverage control and set requires_rag=true for its Classifier terminal; at RAG, accept only if the normal truth/evidence rules hold and the trajectory contains an actual agrinet_rag_search call with response. Never provide a corrected answer or reveal audit protocol." + e39_rule},
                         {"role": "user", "content": [{"type": "text", "text": json.dumps({"truth_code": private["truth_code"], "truth_name": private["truth_name"], "correct_option": private.get("correct_option"), "audit_protocol": {"rag_witness": witness}, "public_options": row.get("public_options") or [], "public_trajectory": public_trajectory}, ensure_ascii=False)}, image]}]}

def validate_parent_protocol(row: dict[str, Any], trajectory: dict[str, Any], audit: dict[str, Any]) -> None:
    """Validate private-only route-coverage controls after provider delivery."""
    if audit.get("decision") not in {"accept", "reject"}:
        raise ValueError("private parent audit decision is invalid")
    if is_rag_protocol_witness(row):
        route = trajectory.get("route")
        if route in {"direct", "classifier"} and audit["decision"] != "reject":
            raise ValueError("private RAG witness must reject pre-RAG terminal")
        if route == "rag":
            trace = trajectory.get("tool_trace")
            if not isinstance(trace, list) or not any(isinstance(item, dict) and
                    isinstance(item.get("call"), dict) and item["call"].get("name") == "agrinet_rag_search" and
                    isinstance(item.get("response"), dict) for item in trace):
                raise ValueError("private RAG witness requires actual RAG call and response")
    designation = (row.get("private") or {}).get("e39_route_coverage") or (row.get("private") or {}).get("e310_route_coverage") or (row.get("private") or {}).get("e311_route_coverage") or (row.get("private") or {}).get("e312_route_coverage") or (row.get("private") or {}).get("e313_route_coverage") or (row.get("private") or {}).get("e314_route_coverage") or (row.get("private") or {}).get("e316_route_coverage") or (row.get("private") or {}).get("e317_route_coverage") or (row.get("private") or {}).get("e318_route_coverage") or (row.get("private") or {}).get("e319_route_coverage")
    if row.get("e39_protocol") in {"agrinet.e39-hcv-cascade/v1", "agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} and designation in {"direct", "classifier", "rag"}:
        route = trajectory.get("route")
        ordering = {"direct": 0, "classifier": 1, "rag": 2}
        if route in ordering and ordering[route] < ordering[designation] and audit["decision"] != "reject":
            raise ValueError("private E3.9 route coverage must reject pre-target terminal")


def run_private_parent(row: dict[str, Any], trajectory: dict[str, Any], *, call: JsonCaller, model: str,
                       enforce_protocol: bool = True, transport_max_side: int = 0) -> dict[str, Any]:
    value = _content(call(private_parent_request(row, trajectory, model=model, transport_max_side=transport_max_side)))
    if value.get("decision") not in {"accept", "reject"} or not isinstance(value.get("reason"), str):
        raise ValueError("private parent audit decision is invalid")
    result = {"decision": value["decision"], "requires_rag": bool(value.get("requires_rag")), "rag_evidence_insufficient": bool(value.get("rag_evidence_insufficient")),
              "private_audit_recorded": True}
    if enforce_protocol:
        validate_parent_protocol(row, trajectory, result)
    return result

def rewrite_request(trajectory: dict[str, Any], *, model: str) -> dict[str, Any]:
    if trajectory.get("route") not in {"direct", "classifier", "rag"} or not isinstance(trajectory.get("answer"), str):
        raise ValueError("rewrite requires a closed public E3.5 trajectory")
    return {"model": model, "temperature": 0.0, "max_tokens": 4096, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "Rewrite only the supplied public trajectory into Hermes messages. Return JSON with reasoning, final, messages. Preserve final exactly. Use no private facts. Every assistant final must be <think>...</think><answer>...</answer>. Preserve actual tool calls and tool responses."},
                         {"role": "user", "content": json.dumps(trajectory, ensure_ascii=False)}]}

def run_rewrite(trajectory: dict[str, Any], *, call: JsonCaller, model: str) -> dict[str, Any]:
    value = _content(call(rewrite_request(trajectory, model=model)) )
    if value.get("final") != trajectory.get("answer") or not isinstance(value.get("reasoning"), str) or not isinstance(value.get("messages"), list):
        raise ValueError("rewrite changed final or lacks Hermes payload")
    return {"reasoning": value["reasoning"], "final": value["final"], "messages": value["messages"]}

def rewrite_audit_request(rewrite: dict[str, Any], trajectory: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {"model": model, "temperature": 0.0, "max_tokens": 512, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "You are an isolated rewrite auditor. Return JSON only with decision accept or reject. Accept only if final is unchanged, all claims come from the supplied public trajectory, and no private metadata appears."},
                         {"role": "user", "content": json.dumps({"trajectory": trajectory, "rewrite": rewrite}, ensure_ascii=False)}]}

def run_rewrite_audit(rewrite: dict[str, Any], trajectory: dict[str, Any], *, call: JsonCaller, model: str) -> str:
    decision = _content(call(rewrite_audit_request(rewrite, trajectory, model=model))).get("decision")
    if decision not in {"accept", "reject"}: raise ValueError("private rewrite audit decision is invalid")
    return str(decision)
