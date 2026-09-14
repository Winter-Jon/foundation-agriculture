"""E3.20 four-stage semantic cascade contract.

This module deliberately owns the new state machine rather than changing the
historical E3.5/E3.19 collector.  Its public state contains no truth or audit
reason: private evaluation is reduced to the safe disposition below.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e316_rag_discriminator import PROMPTS as E316_PROMPTS, validate_e316_trajectory
from agrinet.rag.e317_all_unknown_rag_audit import CELLS, FOLD_TARGETS, IDENTITY_KEYS, _fold, select_e317_all_unknown
from agrinet.rag.e318_all_unknown_512_audit import E318_TOKEN_CAP

E320_PROTOCOL = "agrinet.e320-four-stage-cascade/v1"
STAGES = ("direct", "classifier", "rag", "reject")
QUALITY_REPAIR_LIMIT = 1

_RAG_RULE = (
    "RAG grounding rule: after every actual agrinet_rag_search, inspect public `returned_standard_class_names`. "
    "If any are returned, the final must select exactly one such standard class name; do not answer INSUFFICIENT_EVIDENCE. "
    "For Option, it must also be an offered class and use exactly `class name — LETTER`. "
    "Only when every actual RAG response returns an empty list may you answer INSUFFICIENT_EVIDENCE."
)
_REJECT_RULE = (
    "Reject stage: use the supplied actual public RAG trace. Do not call tools and do not invent a class. "
    "Return legal Hermes HCV reasoning followed by exactly <answer>INSUFFICIENT_EVIDENCE</answer>; this stage is available only after a RAG semantic failure or a RAG trace with no returned standard names."
)
PROMPTS = {**E316_PROMPTS, "rag": E316_PROMPTS["rag"] + _RAG_RULE, "reject": E316_PROMPTS["direct"] + _REJECT_RULE}

def next_stage(stage: str, disposition: str) -> str | None:
    """Semantic errors advance exactly one stage; quality never advances."""
    if stage not in STAGES: raise ValueError("E3.20 invalid stage")
    if disposition == "semantic_correct": return None
    if disposition == "semantic_wrong":
        return STAGES[STAGES.index(stage) + 1] if stage != "reject" else None
    if disposition == "quality_reject": return stage
    if disposition == "delivery_unknown": return None
    raise ValueError("E3.20 invalid disposition")

def validate_transition(*, stage: str, disposition: str, quality_attempt_ordinal: int) -> None:
    successor=next_stage(stage, disposition)
    if disposition == "quality_reject" and quality_attempt_ordinal >= QUALITY_REPAIR_LIMIT:
        raise ValueError("E3.20 same-stage quality repair limit reached")
    if disposition == "quality_reject" and successor != stage: raise ValueError("E3.20 quality rejection must not advance")
    if disposition == "semantic_wrong" and stage != "reject" and successor is None: raise ValueError("E3.20 semantic error must advance")


def safe_private_disposition(audit: dict[str, Any], *, contract_error: str | None = None) -> str:
    """Reduce a private audit to a public-safe controller disposition."""
    if contract_error:
        return "quality_reject"
    if audit.get("quality") != "pass":
        return "quality_reject"
    if audit.get("semantic") == "correct":
        return "semantic_correct"
    if audit.get("semantic") in {"incorrect", "evidence_unavailable"}:
        return "semantic_wrong"
    raise ValueError("E3.20 private audit lacks a classified disposition")

def returned_standard_names(trajectory: dict[str, Any]) -> set[str]:
    names=set()
    for event in trajectory.get("tool_trace") or []:
        if not isinstance(event, dict) or event.get("call", {}).get("name") != "agrinet_rag_search": continue
        response=event.get("response") or {}
        for name in response.get("returned_standard_class_names") or []:
            if isinstance(name, str) and name.strip(): names.add(name.strip())
    return names

def validate_e320_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    route=trajectory.get("route")
    if route == "reject":
        answer=str(trajectory.get("answer") or "")
        if not re.fullmatch(r"\s*<think>.+</think><answer>INSUFFICIENT_EVIDENCE</answer>\s*", answer, re.S):
            raise ValueError("E3.20 Reject must be legal Hermes INSUFFICIENT_EVIDENCE")
        if not trajectory.get("prior_rag_trajectory"): raise ValueError("E3.20 Reject requires prior public RAG trace")
        return
    try: validate_e316_trajectory(row, trajectory)
    except ValueError as exc: raise ValueError(str(exc).replace("E3.16", "E3.20")) from exc
    if route != "rag": return
    body=re.search(r"<answer>(.*?)</answer>", str(trajectory.get("answer") or ""), re.S | re.I).group(1).strip()
    names=returned_standard_names(trajectory)
    if body == "INSUFFICIENT_EVIDENCE" and names: raise ValueError("E3.20 RAG abstention despite returned standard class")
    if body != "INSUFFICIENT_EVIDENCE":
        chosen=body.rsplit(" — ", 1)[0].strip()
        if chosen not in names: raise ValueError("E3.20 RAG answer is not a returned standard class")

def materialize_e320_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 32 or Counter(_fold(row) for row in rows) != Counter(FOLD_TARGETS): raise ValueError("E3.20 requires fold balance 11/11/10")
    if Counter((r.get("question_type"), r.get("task_domain")) for r in rows) != Counter({c:8 for c in CELLS}): raise ValueError("E3.20 requires eight rows per cell")
    out=[]
    for row in rows:
        private=dict(row.get("private") or {}); private["e320_route_coverage"]="rag"; private["audit_protocol"]={"rag_witness":True,"purpose":"e320_four_stage"}
        out.append({**row,"e39_protocol":E320_PROTOCOL,"teacher_system_prompts":dict(PROMPTS),"private":private})
    return out

def write_e320_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists(): raise ValueError("E3.20 manifest is immutable")
    rows=[json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows)!=32 or any(r.get("e39_protocol") != E320_PROTOCOL for r in rows): raise ValueError("E3.20 source binding invalid")
    p={"schema_version":"agrinet.e320-four-stage-cascade-manifest/v1","protocol":E320_PROTOCOL,"campaign_id":campaign_id,"round":"R0","source":str(source),"source_sha256":hashlib.file_digest(source.open("rb"),"sha256").hexdigest(),"source_rows_expected":32,"audit_only":True,"all_simulated_unknown":True,"classifier_fold_targets":FOLD_TARGETS,"work_items":[{"work_id":f"R0:{r['sample_id']}:e320-four-stage","round":"R0","sample_id":r["sample_id"],"image_group_id":r.get("image_group_id",r["image_sha256"]),"attempt_ordinal":0,"predecessor_request_id":None,"route_progression":list(STAGES),"quality_attempt_ordinal":0,"prompt_revision":"base"} for r in rows],"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":{"uncached_input_token_cap":E318_TOKEN_CAP,"transport_image_max_side":512,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000,"reject":5000},"private_audit":18000}},"quality_repair":{"max_per_stage":1,"new_request_id":True,"separate_from_delivery_recovery":True},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return p


def write_e320_continuation_manifest(*, prior_manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Freeze semantic advancement or a same-stage quality repair.

    This intentionally excludes provider ambiguity: unknown delivery stays on
    the immutable R0/R1/R2 recovery path rather than becoming a quality retry.
    """
    if output.exists():
        raise ValueError("E3.20 continuation manifest is immutable")
    prior=json.loads(prior_manifest.read_text(encoding="utf-8"))
    result=json.loads(outcomes.read_text(encoding="utf-8"))
    digest=hashlib.file_digest(prior_manifest.open("rb"), "sha256").hexdigest()
    if prior.get("protocol") != E320_PROTOCOL or result.get("manifest_sha256") != digest:
        raise ValueError("E3.20 continuation lineage mismatch")
    original={str(item.get("work_id")): item for item in prior.get("work_items") or []}
    work=[]
    for outcome in result.get("outcomes") or []:
        item=original.get(str(outcome.get("work_id")))
        if item is None: raise ValueError("E3.20 continuation outcome is out of scope")
        if outcome.get("delivery_status") == "unknown_delivery": continue
        stage=str(outcome.get("final_route") or outcome.get("stage") or "")
        disposition=str(outcome.get("disposition") or "")
        ordinal=int(outcome.get("quality_attempt_ordinal", item.get("quality_attempt_ordinal", 0)))
        validate_transition(stage=stage, disposition=disposition, quality_attempt_ordinal=ordinal)
        successor=next_stage(stage, disposition)
        if successor is None: continue
        repair=disposition == "quality_reject"
        work.append({
            **{key:item[key] for key in ("sample_id", "image_group_id", "route_progression")},
            "work_id": (f"Q{ordinal + 1}:{item['sample_id']}:e320-{successor}" if repair else f"S:{item['sample_id']}:e320-{successor}"),
            "round": "Q1" if repair else "S1", "resume_route": successor,
            "attempt_ordinal": int(item.get("attempt_ordinal", 0)),
            "predecessor_request_id": outcome.get("request_id"),
            "quality_attempt_ordinal": ordinal + 1 if repair else 0,
            "prompt_revision": "repair_v1" if repair else "base",
            "transition_reason": disposition,
            **({"prior_rag_parent_path": outcome.get("parent_path")} if successor == "reject" else {}),
        })
    payload={"schema_version":prior["schema_version"],"protocol":E320_PROTOCOL,"campaign_id":prior.get("campaign_id"),"round":"continuation","continuation":True,"source":prior.get("source"),"source_sha256":prior.get("source_sha256"),"predecessor_manifest_sha256":digest,"work_items":work,"workers":prior.get("workers"),"micu_intent_limit":prior.get("micu_intent_limit"),"automatic_replay_allowed":False,"collection_controls":prior.get("collection_controls"),"quality_repair":prior.get("quality_repair"),"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return payload


def write_e320_delivery_recovery_manifest(*, prior_manifest: Path, outcomes: Path,
                                          next_round: str, output: Path) -> dict[str, Any]:
    """Materialize only R1/R2 replacements for unresolved delivery."""
    if next_round not in {"R1", "R2"}:
        raise ValueError("E3.20 delivery recovery must be R1 or R2")
    if output.exists():
        raise ValueError("E3.20 delivery recovery destination is immutable")
    prior=json.loads(prior_manifest.read_text(encoding="utf-8"))
    result=json.loads(outcomes.read_text(encoding="utf-8"))
    digest=hashlib.file_digest(prior_manifest.open("rb"), "sha256").hexdigest()
    if prior.get("protocol") != E320_PROTOCOL or result.get("manifest_sha256") != digest:
        raise ValueError("E3.20 delivery recovery lineage mismatch")
    expected_prior={"R1": "R0", "R2": "R1"}[next_round]
    if prior.get("round") != expected_prior:
        raise ValueError("E3.20 delivery recovery round order is invalid")
    items={str(item.get("work_id")):item for item in prior.get("work_items") or []}
    work=[]
    for outcome in result.get("outcomes") or []:
        if outcome.get("delivery_status") != "unknown_delivery": continue
        item=items.get(str(outcome.get("work_id")))
        if item is None or not outcome.get("request_id"):
            raise ValueError("E3.20 unknown delivery lacks bound predecessor")
        work.append({**{key:item[key] for key in ("sample_id","image_group_id","route_progression")},
                     "work_id":f"{next_round}:{item['sample_id']}:e320-{item.get('resume_route','direct')}-delivery",
                     "round":next_round,"resume_route":item.get("resume_route","direct"),
                     "attempt_ordinal":int(item.get("attempt_ordinal",0))+1,
                     "predecessor_request_id":outcome["request_id"],
                     "quality_attempt_ordinal":int(item.get("quality_attempt_ordinal",0)),
                     "prompt_revision":item.get("prompt_revision","base"),
                     **({"prior_rag_parent_path":item["prior_rag_parent_path"]} if item.get("prior_rag_parent_path") else {})})
    payload={"schema_version":prior["schema_version"],"protocol":E320_PROTOCOL,"campaign_id":prior.get("campaign_id"),"round":next_round,"delivery_recovery":True,"source":prior.get("source"),"source_sha256":prior.get("source_sha256"),"predecessor_manifest_sha256":digest,"work_items":work,"workers":prior.get("workers"),"micu_intent_limit":prior.get("micu_intent_limit"),"automatic_replay_allowed":False,"collection_controls":prior.get("collection_controls"),"quality_repair":prior.get("quality_repair"),"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload

def select_e320_all_unknown(rows: list[dict[str, Any]], *, prior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return select_e317_all_unknown(rows, prior_rows=prior_rows, seed="e320-four-stage-cascade-v1")


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args=parser.parse_args(argv)
    if args.source_output.exists() or args.manifest_output.exists():
        raise ValueError("E3.20 preparation destinations are immutable")
    candidates=[json.loads(x) for x in args.candidate_source.read_text(encoding="utf-8").splitlines() if x.strip()]
    prior=[json.loads(x) for path in args.prior for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    source=materialize_e320_source(select_e320_all_unknown(candidates, prior_rows=prior))
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in source), encoding="utf-8")
    manifest=write_e320_manifest(source=args.source_output, campaign_id=args.campaign_id, output=args.manifest_output)
    print(json.dumps({"rows":len(source),"folds":dict(Counter(_fold(r) for r in source)),"transport_image_max_side":512,"uncached_input_token_cap":manifest["collection_controls"]["uncached_input_token_cap"],"training_authorized":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
