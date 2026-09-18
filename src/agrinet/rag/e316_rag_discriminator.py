"""E3.16: future identity-disjoint RAG audit with discriminative retrieval.

This module intentionally does not alter E3.14/E3.15 sources, prompts,
outcomes, or request lineages.  It defines the teacher contract for a new audit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agrinet.rag.e310_hcv_cascade import select_e310_coverage
from agrinet.rag.e311_hcv_cascade import select_e311_audit
from agrinet.rag.e313_hcv_cascade import BASE, PROMPTS as E313_PROMPTS, validate_e313_trajectory
from agrinet.rag.e315_option_format_repair import CLASSIFIER_OPTION_PROMPT
from agrinet.rag.e35_classifier_cascade import DELIVERY_FAILURES, recovery_attempt

E316_PROTOCOL = "agrinet.e316-rag-discriminator/v1"
E316_CANARY_PROTOCOL = "agrinet.e316-rag-discriminator-canary/v1"

RAG_DISCRIMINATOR_PROMPT = (
    BASE
    + "First output one standalone pure <think> planning turn and then make exactly one native agrinet_classifier_predict call. "
      "After its actual card, identify the leading image-grounded candidate and its nearest visually confusable alternative. Do not treat a classifier rank as evidence. "
      "Before every RAG call, output one standalone pure <think> plan naming one concrete visible discriminator that could separate those two candidates. The first retrieval must compare the leading candidate against its nearest alternative; later retrievals, if any, must test a different unresolved discriminator. "
      "Call only agrinet_rag_search with every required argument. Label-agnostic schema example: agrinet_rag_search({\"query\":\"candidate A versus candidate B: diagnostic visible leaf-margin and lesion-surface traits\",\"retrieval_type\":\"visual\",\"rationale\":\"Verify whether the image-visible margin and surface traits distinguish candidate A from candidate B.\"}). `retrieval_type` is required and must be exactly `visual` or `semantic`; never omit it. "
      "Do not use retrieval to confirm a favorite candidate in isolation and never merge classifier and retrieval scores. "
      "In the final RAG <think>, Candidate comparison: must explicitly identify `Nearest alternative:`. Evidence: must contain these literal labels once: `Visible trait:`, `RAG evidence:`, and `Discriminator:`. Tie a trait actually visible in the image to a statement in the actual RAG response, then explain why that trait excludes the nearest alternative. "
      "Under Rejected alternatives:, include the nearest alternative using `candidate name: rejected because visible trait conflicts with ...`. "
      "If the decisive trait is not visible, or actual RAG does not establish it, do not choose a concrete class merely because a retrieved entry mentions it. After at least one actual RAG response, answer `INSUFFICIENT_EVIDENCE`. "
      "For Option, still compare A/B/C/D explicitly in the final Candidate comparison:."
)

PROMPTS = {**E313_PROMPTS, "classifier": CLASSIFIER_OPTION_PROMPT, "rag": RAG_DISCRIMINATOR_PROMPT}


def _section(answer: str, name: str, following: str) -> str:
    match = re.search(rf"{re.escape(name)}(.*?){re.escape(following)}", answer, re.I | re.S)
    return match.group(1).strip() if match else ""


def validate_e316_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    """Retain E3.14 HCV contract and add inspectable RAG evidence linkage."""
    try:
        validate_e313_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.13", "E3.16")) from exc
    if trajectory.get("route") != "rag":
        return
    trace = trajectory.get("tool_trace") or []
    rag_calls = [item for item in trace if isinstance(item, dict) and item.get("call", {}).get("name") == "agrinet_rag_search"]
    if not rag_calls or any(not isinstance(item.get("response"), dict) for item in rag_calls):
        raise ValueError("E3.16 RAG requires an actual search response")
    for item in rag_calls:
        args=item.get("call", {}).get("arguments")
        if not isinstance(args, dict) or args.get("retrieval_type") not in {"visual", "semantic", "balanced"}:
            raise ValueError("E3.16 RAG retrieval_type is required")
    answer=str(trajectory.get("answer") or "")
    body=re.search(r"<answer>(.*?)</answer>", answer, re.I | re.S).group(1).strip()
    comparison=_section(answer, "Candidate comparison:", "Evidence:")
    evidence=_section(answer, "Evidence:", "Rejected alternatives:")
    rejected=_section(answer, "Rejected alternatives:", "Uncertainty:")
    if "Nearest alternative:" not in comparison:
        raise ValueError("E3.16 RAG nearest alternative is missing")
    if any(label not in evidence for label in ("Visible trait:", "RAG evidence:", "Discriminator:")):
        raise ValueError("E3.16 RAG evidence linkage is incomplete")
    nearest_line = comparison.split("Nearest alternative:", 1)[1].strip().splitlines()[0].strip()
    # The prompt permits `nearest name: explanation` and Option prose such as
    # `B, a nearest name, because ...`.  Extract stable candidate tokens only;
    # never require the explanatory tail to be copied verbatim into rejection.
    nearest_head = nearest_line.split(":", 1)[0].strip()
    label = re.match(r"([A-D])(?:\s|,|—|$)", nearest_head)
    names=[nearest_head]
    if label:
        names.append(label.group(1))
        options={str(item.get("label")):str(item.get("name")) for item in row.get("public_options") or [] if isinstance(item, dict)}
        if label.group(1) in options:
            names.append(options[label.group(1)])
    if body == "INSUFFICIENT_EVIDENCE":
        # Honest abstention must retain the candidate comparison and explain why
        # retrieval leaves the discriminator unresolved; it need not falsely
        # claim that the nearest candidate was ruled out.
        uncertainty=_section(answer, "Uncertainty:", "</think>")
        limitation_text=f"{evidence} {uncertainty}"
        if not re.search(r"(?:not visible|cannot|insufficient|unresolved|(?:did )?not .*?(?:establish|provide|confirm))", limitation_text, re.I):
            raise ValueError("E3.16 abstention lacks evidence limitation")
    elif not any(name and name.lower() in rejected.lower() for name in names):
        raise ValueError("E3.16 RAG nearest alternative is not rejected")


def materialize_e316_source(rows: list[dict[str, Any]], coverage: dict[str, str]) -> list[dict[str, Any]]:
    if len(rows) != 32 or len(coverage) != 6:
        raise ValueError("E3.16 needs 32 rows and six private route witnesses")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        if row["sample_id"] in coverage:
            private["e316_route_coverage"] = coverage[row["sample_id"]].rsplit(":", 1)[1]
        result.append({**row, "e39_protocol": E316_PROTOCOL, "teacher_system_prompts": dict(PROMPTS), "private": private})
    return result


def select_e316_canary(rows: list[dict[str, Any]], *, prior_rows: list[dict[str, Any]],
                       seed: str = "e316-rag-discriminator-canary-v1") -> list[dict[str, Any]]:
    """Freeze four novel cells for a RAG-only behavioral canary."""
    blocked={key:{str(row.get(key)) for row in prior_rows} for key in
             ("sample_id", "image_sha256", "source_group_id", "near_duplicate_group_id")}
    cells=(("known", "open", "disease"), ("known", "option", "pest"),
           ("simulated_unknown", "open", "pest"), ("simulated_unknown", "option", "disease"))
    chosen=[]
    for arm, kind, domain in cells:
        candidates=[row for row in rows if row.get("arm") == arm and row.get("question_type") == kind
                    and row.get("task_domain") == domain
                    and all(str(row.get(key)) not in values for key, values in blocked.items())]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
        if not candidates:
            raise ValueError(f"E3.16 canary lacks identity-disjoint {arm}/{kind}/{domain}")
        chosen.append(candidates[0])
    if len({row["sample_id"] for row in chosen}) != 4:
        raise ValueError("E3.16 canary identity collision")
    return chosen


def materialize_e316_canary_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 4:
        raise ValueError("E3.16 canary needs exactly four rows")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        private["e316_route_coverage"] = "rag"
        private["audit_protocol"] = {"rag_witness": True}
        result.append({**row, "e39_protocol": E316_CANARY_PROTOCOL,
                       "teacher_system_prompts": dict(PROMPTS), "private": private})
    return result


def write_e316_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.16 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 32 or any(row.get("e39_protocol") != E316_PROTOCOL for row in rows):
        raise ValueError("E3.16 source binding invalid")
    payload={
        "schema_version": "agrinet.e316-rag-discriminator-manifest/v1",
        "protocol": E316_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source), "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected": 32, "audit_only": True,
        "work_items": [{
            "work_id": f"R0:{row['sample_id']}:e316-rag-discriminator", "round": "R0",
            "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
            "attempt_ordinal": 0, "predecessor_request_id": None,
            "route_progression": ["direct", "classifier", "rag"],
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


def write_e316_canary_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.16 canary manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    cells={(row.get("arm"), row.get("question_type"), row.get("task_domain")) for row in rows}
    required={("known", "open", "disease"), ("known", "option", "pest"),
              ("simulated_unknown", "open", "pest"), ("simulated_unknown", "option", "disease")}
    if len(rows) != 4 or cells != required or any(row.get("e39_protocol") != E316_CANARY_PROTOCOL for row in rows):
        raise ValueError("E3.16 canary source binding invalid")
    payload={
        "schema_version": "agrinet.e316-rag-discriminator-canary-manifest/v1",
        "protocol": E316_CANARY_PROTOCOL, "campaign_id": campaign_id, "round": "R0",
        "source": str(source), "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected": 4, "audit_only": True, "canary_only": True,
        "work_items": [{
            "work_id": f"R0:{row['sample_id']}:e316-rag-canary", "round": "R0",
            "sample_id": row["sample_id"], "image_group_id": row.get("image_group_id", row["image_sha256"]),
            "attempt_ordinal": 0, "predecessor_request_id": None,
            "route_progression": ["direct", "classifier", "rag"],
        } for row in rows],
        "workers": 4, "micu_intent_limit": 8000, "automatic_replay_allowed": False,
        "collection_controls": {
            "uncached_input_token_cap": 120000, "transport_image_max_side": 1024,
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


def write_e316_canary_replenishment_manifest(*, summary: dict[str, Any], next_round: str, output: Path) -> dict[str, Any]:
    prior={"R1": "R0", "R2": "R1"}
    if next_round not in prior or summary.get("round") != prior[next_round]:
        raise ValueError("E3.16 canary recovery must proceed R0 -> R1 -> R2")
    if output.exists():
        raise ValueError("E3.16 canary replenishment destination is immutable")
    controls=summary.get("collection_controls")
    if not isinstance(controls, dict) or controls.get("uncached_input_token_cap") != 120000:
        raise ValueError("E3.16 canary recovery has no frozen controls")
    work=[]
    for row in summary.get("rows", []):
        if row.get("delivery_status") not in DELIVERY_FAILURES:
            continue
        rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal", -1)),
                             status=str(row["delivery_status"]), predecessor_request_id=str(row.get("request_id") or ""))
        if rec.get("new_attempt"):
            work.append({
                "work_id": f"{next_round}:{row['sample_id']}:e316-rag-canary", "round": next_round,
                "sample_id": row["sample_id"], "image_group_id": row["image_group_id"],
                "route_progression": ["direct", "classifier", "rag"], **rec,
            })
    payload={
        "schema_version": "agrinet.e316-rag-discriminator-canary-manifest/v1",
        "protocol": E316_CANARY_PROTOCOL, "campaign_id": summary.get("campaign_id"), "round": next_round,
        "audit_only": True, "canary_only": True, "work_items": work, "workers": 4,
        "micu_intent_limit": 8000, "automatic_replay_allowed": False, "collection_controls": controls,
        "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def freeze_e316_canary_summary(*, manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Bind one completed canary round before any recovery planning."""
    if output.exists():
        raise ValueError("E3.16 canary summary is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8"))
    result=json.loads(outcomes.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "agrinet.e316-rag-discriminator-canary-manifest/v1":
        raise ValueError("E3.16 canary summary has invalid manifest")
    actual_sha=hashlib.file_digest(manifest.open("rb"), "sha256").hexdigest()
    if result.get("manifest_sha256") != actual_sha or result.get("campaign_id") != plan.get("campaign_id") or result.get("round") != plan.get("round"):
        raise ValueError("E3.16 canary outcome lineage mismatch")
    by_work={str(row.get("work_id")):row for row in result.get("outcomes", [])}
    items=plan.get("work_items", [])
    if len(by_work) != len(items) or set(by_work) != {str(item.get("work_id")) for item in items}:
        raise ValueError("E3.16 canary summary scope mismatch")
    rows=[]
    for item in items:
        outcome=by_work[str(item["work_id"])]
        rows.append({
            **{key:item.get(key) for key in ("work_id", "round", "sample_id", "image_group_id", "attempt_ordinal", "predecessor_request_id", "route_progression")},
            **{key:outcome.get(key) for key in ("delivery_status", "request_id", "winner", "final_route", "quality_status", "contract_error", "parent_path")},
        })
    payload={
        "schema_version": "agrinet.e316-rag-discriminator-canary-summary/v1",
        "campaign_id": plan["campaign_id"], "round": plan["round"],
        "manifest_sha256": actual_sha, "source_sha256": plan.get("source_sha256"),
        "collection_controls": plan["collection_controls"], "automatic_replay_allowed": False,
        "rows": rows, "training_eligible": False, "training_authorized": False, "sft_may_start": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare-canary":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--candidate-source", type=Path, required=True)
        parser.add_argument("--prior", action="append", type=Path, required=True)
        parser.add_argument("--source-output", type=Path, required=True)
        parser.add_argument("--campaign-id", required=True)
        parser.add_argument("--manifest-output", type=Path, required=True)
        args=parser.parse_args(argv[1:])
        if args.source_output.exists() or args.manifest_output.exists():
            raise ValueError("E3.16 canary preparation destinations are immutable")
        candidates=[json.loads(line) for line in args.candidate_source.read_text(encoding="utf-8").splitlines() if line.strip()]
        prior=[json.loads(line) for path in args.prior for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        source=materialize_e316_canary_source(select_e316_canary(candidates, prior_rows=prior))
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source), encoding="utf-8")
        manifest=write_e316_canary_manifest(source=args.source_output, campaign_id=args.campaign_id, output=args.manifest_output)
        print(json.dumps({"rows": len(source), "prior_rows": len(prior), "manifest_rows": len(manifest["work_items"]), "training_authorized": False}))
        return 0
    if argv and argv[0] == "freeze-canary-summary":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--outcomes", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args=parser.parse_args(argv[1:])
        result=freeze_e316_canary_summary(manifest=args.manifest, outcomes=args.outcomes, output=args.output)
        print(json.dumps({"round": result["round"], "rows": len(result["rows"])}))
        return 0
    if argv and argv[0] == "plan-canary-replenishment":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--summary", type=Path, required=True)
        parser.add_argument("--next-round", choices=("R1", "R2"), required=True)
        parser.add_argument("--output", type=Path, required=True)
        args=parser.parse_args(argv[1:])
        result=write_e316_canary_replenishment_manifest(summary=json.loads(args.summary.read_text(encoding="utf-8")), next_round=args.next_round, output=args.output)
        print(json.dumps({"round": result["round"], "rows": len(result["work_items"])}))
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--prior", action="append", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args=parser.parse_args(argv)
    if any(path.exists() for path in (args.coverage_output, args.source_output, args.manifest_output)):
        raise ValueError("E3.16 preparation destinations are immutable")
    candidates=[json.loads(line) for line in args.candidate_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior=[json.loads(line) for path in args.prior for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows=select_e311_audit(candidates, prior_rows=prior, seed="e316-rag-discriminator-v1")
    coverage=select_e310_coverage(rows, seed="e316-rag-discriminator-coverage-v1")
    source=materialize_e316_source(rows, coverage)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.write_text(json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source), encoding="utf-8")
    manifest=write_e316_manifest(source=args.source_output, campaign_id=args.campaign_id, output=args.manifest_output)
    print(json.dumps({"rows": len(source), "prior_rows": len(prior), "manifest_rows": len(manifest["work_items"]), "training_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
