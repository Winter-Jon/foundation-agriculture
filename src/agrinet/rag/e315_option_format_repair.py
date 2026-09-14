"""E3.15: isolated one-shot repair of E3.14 Option terminal formatting.

This is a new five-image pre-collection, not a replay of terminal E3.14
records. It retains a private pointer to the superseded classifier response
for audit lineage while every live request receives a new request ID.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e313_hcv_cascade import BASE, validate_e313_trajectory

E315_PROTOCOL = "agrinet.e315-option-format-repair/v1"
E315_EXPECTED = 5

# Label-agnostic: demonstrate only the answer wire format, never a target.
CLASSIFIER_OPTION_PROMPT = (
    BASE
    + "First output one standalone pure <think> planning turn and do not answer or call a tool. "
      "Next make exactly one native agrinet_classifier_predict call. After the card, answer or make at most one agrinet_classifier_expand call for one stated ambiguity. "
      "For Option, Candidate comparison: must contain exactly four labeled clauses `A. public class name: ...` through `D. public class name: ...`; assess every option before answering. "
      "The final <answer> is mechanically checked. It must contain exactly the selected public class name, one space, an em dash (—), one space, and its uppercase option letter; do not use `A. class`, `class (A)`, `A`, or added words. "
      "One format-only example: if option B is selected and its public name is `example leaf blight`, write exactly `<answer>example leaf blight — B</answer>`."
)
PROMPTS = {
    "direct": BASE + "No tools. This E3.15 repair never starts Direct.",
    "classifier": CLASSIFIER_OPTION_PROMPT,
    "rag": BASE + "This E3.15 repair never starts RAG.",
}


def validate_e315_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    """Use frozen E3.14-equivalent HCV and Option validators."""
    try:
        validate_e313_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.13", "E3.15")) from exc
    if row.get("question_type") != "option":
        raise ValueError("E3.15 repair source must contain only Option rows")


def _classifier_request_id(ledger_path: Path) -> str:
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "result" and str(event.get("key", "")).startswith("classifier:generation"):
            return str(event["request_id"])
    raise ValueError(f"E3.14 classifier request is absent: {ledger_path}")


def materialize_e315_source(*, e314_source: Path, e314_outcome_paths: list[Path],
                            e314_campaign_root: Path) -> list[dict[str, Any]]:
    """Select exactly the five confirmed E3.14 format-only terminal records."""
    source_by_id = {row["sample_id"]: row for row in (
        json.loads(line) for line in e314_source.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    selected: dict[str, dict[str, Any]] = {}
    for path in e314_outcome_paths:
        for outcome in json.loads(path.read_text(encoding="utf-8"))["outcomes"]:
            if (outcome.get("delivery_status") != "delivered"
                    or outcome.get("quality_status") != "route_contract_reject"
                    or outcome.get("contract_error") != "option_terminal_format"):
                continue
            sample_id = str(outcome["work_id"]).split(":", 2)[1]
            if sample_id in selected:
                raise ValueError(f"duplicate E3.14 format terminal: {sample_id}")
            selected[sample_id] = outcome
    if len(selected) != E315_EXPECTED:
        raise ValueError(f"E3.15 needs exactly {E315_EXPECTED} E3.14 Option-format terminals")
    rows=[]
    for sample_id, outcome in sorted(selected.items()):
        prior_work_id = str(outcome["work_id"])
        ledger = e314_campaign_root / "ledgers" / prior_work_id.replace(":", "_") / "events.jsonl"
        old = source_by_id.get(sample_id)
        if old is None or old.get("question_type") != "option":
            raise ValueError(f"E3.15 source binding is not an Option row: {sample_id}")
        private = dict(old.get("private") or {})
        private.pop("e314_route_coverage", None)
        private["e315_superseded"] = {
            "protocol": "agrinet.e314-hcv-cascade/v1",
            "terminal_code": "option_terminal_format",
            "classifier_request_id": _classifier_request_id(ledger),
        }
        rows.append({**old, "e39_protocol": E315_PROTOCOL,
                     "teacher_system_prompts": dict(PROMPTS), "private": private})
    return rows


def write_e315_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.15 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != E315_EXPECTED or any(
            row.get("e39_protocol") != E315_PROTOCOL or row.get("question_type") != "option" for row in rows):
        raise ValueError("E3.15 source binding invalid")
    payload = {
        "schema_version": "agrinet.e315-option-format-repair-manifest/v1",
        "protocol": E315_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source),
        "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected": E315_EXPECTED, "audit_only": True,
        "work_items": [{
            "work_id": f"R0:{row['sample_id']}:e315-option-format", "round": "R0",
            "sample_id": row["sample_id"],
            "image_group_id": row.get("image_group_id", row["image_sha256"]),
            "attempt_ordinal": 0, "predecessor_request_id": None,
            "supersedes_classifier_request_id": row["private"]["e315_superseded"]["classifier_request_id"],
            "resume_route": "classifier", "route_progression": ["classifier", "rag"],
        } for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "collection_controls": {
            "uncached_input_token_cap": 300000, "transport_image_max_side": 1024,
            "max_public_turns_per_route": 12, "max_rag_searches": 3,
            "reservation_uncached_tokens": {
                "generation": {"direct": 5000, "classifier": 8000, "rag": 12000},
                "private_audit": 18000,
            },
        },
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e314-source", type=Path, required=True)
    parser.add_argument("--e314-outcome", action="append", type=Path, required=True)
    parser.add_argument("--e314-campaign-root", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args=parser.parse_args(argv)
    if args.source_output.exists() or args.manifest_output.exists():
        raise ValueError("E3.15 preparation destinations are immutable")
    rows=materialize_e315_source(
        e314_source=args.e314_source, e314_outcome_paths=args.e314_outcome,
        e314_campaign_root=args.e314_campaign_root)
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest=write_e315_manifest(source=args.source_output, campaign_id=args.campaign_id, output=args.manifest_output)
    print(json.dumps({"rows": len(rows), "manifest_rows": len(manifest["work_items"]), "training_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
