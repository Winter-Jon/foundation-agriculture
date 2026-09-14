"""Isolated structured private audit for E3.20."""
from __future__ import annotations

import json
from typing import Any


def e320_private_audit_request(*, truth_name: str, correct_option: str | None,
                              public_options: list[dict[str, Any]], trajectory: dict[str, Any],
                              model: str, image: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an audit request whose public-safe output has no explanation."""
    return {
        "model": model, "temperature": 0.0, "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content":
             "You are an isolated private auditor. Return JSON only with semantic (correct, incorrect, or evidence_unavailable) and quality (pass or fail). semantic compares the final with private truth. quality checks Hermes/HCV, Option terminal form, tool order/schema, and RAG evidence grounding. Do not give reasons, corrections, truth, or a route recommendation. Classify independently: an incorrect but well-formed answer is semantic=incorrect, quality=pass; a correct malformed answer is semantic=correct, quality=fail."},
            {"role": "user", "content": ([{"type": "text", "text": json.dumps({
                "private_truth_name": truth_name, "private_correct_option": correct_option,
                "public_options": public_options, "public_trajectory": trajectory,
            }, ensure_ascii=False)}] + ([image] if image else []))},
        ],
    }


def parse_e320_private_audit(response: dict[str, Any]) -> dict[str, str]:
    try:
        raw = response["choices"][0]["message"]["content"]
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("E3.20 private audit response is invalid") from exc
    if (not isinstance(value, dict)
            or value.get("semantic") not in {"correct", "incorrect", "evidence_unavailable"}
            or value.get("quality") not in {"pass", "fail"}):
        raise ValueError("E3.20 private audit classification is invalid")
    return {"semantic": value["semantic"], "quality": value["quality"]}
