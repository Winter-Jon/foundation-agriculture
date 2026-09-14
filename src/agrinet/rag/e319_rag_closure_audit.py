"""E3.19: prospective RAG-closure repair after the terminal E3.18 audit.

It deliberately freezes a new source line.  Its public teacher contract makes
the evidence-to-answer decision explicit: retrieve to falsify a near neighbour,
then answer concretely when that discriminator is established.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agrinet.rag.e316_rag_discriminator import PROMPTS as E316_PROMPTS, validate_e316_trajectory
from agrinet.rag.e317_all_unknown_rag_audit import FOLD_TARGETS
from agrinet.rag.e318_all_unknown_512_audit import E318_TOKEN_CAP, select_e318_all_unknown

E319_PROTOCOL = "agrinet.e319-rag-closure-audit/v1"

_CLOSURE = (
    "RAG closure rule: after an actual RAG response, choose a concrete canonical class whenever the visible trait and the RAG evidence establish the stated discriminator and exclude the nearest alternative. Do not turn residual uncertainty into INSUFFICIENT_EVIDENCE. "
    "INSUFFICIENT_EVIDENCE is permitted only when both conditions hold: (1) name one decisive missing trait under the literal heading `Decisive missing trait:`, and state it is not visible in this image; (2) under `RAG limitation:`, state that the actual RAG response does not establish that same trait. If either condition is absent, output the best evidence-supported concrete answer. "
    "For Option, the final answer is checked byte-for-byte after trimming: `<answer>selected public class name — UPPERCASE LETTER</answer>`. Never emit `B. name`, `B) name`, a bare letter, a parenthesized letter, or explanatory text in <answer>. Example only: `<answer>example leaf blight — B</answer>`. "
)

PROMPTS = {**E316_PROMPTS, "rag": E316_PROMPTS["rag"] + _CLOSURE,
           "classifier": E316_PROMPTS["classifier"] + _CLOSURE}


def select_e319_all_unknown(rows: list[dict[str, Any]], *, prior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return select_e318_all_unknown(rows, prior_rows=prior_rows)


def materialize_e319_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 32 or any(row.get("arm") != "simulated_unknown" for row in rows):
        raise ValueError("E3.19 requires 32 simulated-Unknown rows")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        for key in tuple(private):
            if key.endswith("_route_coverage"): private.pop(key)
        private["e319_route_coverage"] = "rag"
        private["audit_protocol"] = {"rag_witness": True, "purpose": "e319_concrete_rag_closure"}
        result.append({**row, "e39_protocol": E319_PROTOCOL, "teacher_system_prompts":dict(PROMPTS), "private":private})
    return result


def validate_e319_trajectory(row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    try:
        validate_e316_trajectory(row, trajectory)
    except ValueError as exc:
        raise ValueError(str(exc).replace("E3.16", "E3.19")) from exc
    if trajectory.get("route") != "rag": return
    answer=str(trajectory.get("answer") or "")
    match=re.search(r"<answer>(.*?)</answer>", answer, re.I | re.S)
    if not match: raise ValueError("E3.19 final must be legal Hermes")
    body=match.group(1).strip()
    if body == "INSUFFICIENT_EVIDENCE":
        if not re.search(r"Decisive missing trait:\s*[^\n]+(?:not visible|cannot be seen|unresolved)", answer, re.I):
            raise ValueError("E3.19 abstention lacks missing visible trait")
        if not re.search(r"RAG limitation:\s*[^\n]+(?:does not|did not|cannot|insufficient|unresolved)", answer, re.I):
            raise ValueError("E3.19 abstention lacks matching RAG limitation")


def write_e319_manifest(*, source: Path, campaign_id: str, output: Path) -> dict[str, Any]:
    if output.exists(): raise ValueError("E3.19 manifest is immutable")
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 32 or any(row.get("e39_protocol") != E319_PROTOCOL for row in rows):
        raise ValueError("E3.19 source binding invalid")
    payload={
        "schema_version":"agrinet.e319-rag-closure-audit-manifest/v1", "protocol":E319_PROTOCOL,
        "campaign_id":campaign_id, "round":"R0", "source":str(source),
        "source_sha256":hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        "source_rows_expected":32, "audit_only":True, "all_simulated_unknown":True,
        "classifier_fold_targets":FOLD_TARGETS, "transport_resolution_calibration":True,
        "work_items":[{"work_id":f"R0:{row['sample_id']}:e319-rag-closure", "round":"R0", "sample_id":row["sample_id"], "image_group_id":row.get("image_group_id",row["image_sha256"]), "attempt_ordinal":0, "predecessor_request_id":None, "route_progression":["direct","classifier","rag"]} for row in rows],
        "workers":4, "micu_intent_limit":8000, "automatic_replay_allowed":False,
        "collection_controls":{"uncached_input_token_cap":E318_TOKEN_CAP, "transport_image_max_side":512, "max_public_turns_per_route":12, "max_rag_searches":3, "reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000},"private_audit":18000}},
        "training_eligible":False, "training_authorized":False, "sft_may_start":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


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
        raise ValueError("E3.19 preparation destinations are immutable")
    candidates=[json.loads(line) for line in args.candidate_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior=[json.loads(line) for path in args.prior for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source=materialize_e319_source(select_e319_all_unknown(candidates, prior_rows=prior))
    args.source_output.parent.mkdir(parents=True,exist_ok=True)
    args.source_output.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in source),encoding="utf-8")
    manifest=write_e319_manifest(source=args.source_output,campaign_id=args.campaign_id,output=args.manifest_output)
    print(json.dumps({"rows":len(source),"prior_rows":len(prior),"transport_image_max_side":512,"uncached_input_token_cap":manifest["collection_controls"]["uncached_input_token_cap"],"training_authorized":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
