"""E3.21 Direct-First HCV Cascade preparation and immutable manifests.

The module intentionally materializes only the Direct work that is authorized
now.  It nevertheless freezes the four-stage successor contract in every
manifest so later controllers cannot reinterpret a Direct semantic failure.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e320_four_stage_cascade import PROMPTS, STAGES

E321_PROTOCOL = "agrinet.e321-direct-first-hcv-cascade/v1"
SCHEMA = "agrinet.e321-direct-first-hcv-cascade-manifest/v1"
CAP = 16_000_000
RESERVATION = {"generation": {"direct": 1000, "classifier": 8000, "rag": 12000, "reject": 5000}, "private_audit": 2500}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def materialize_direct_source(*, candidates: Path, e320_source: Path, e320_r1_outcomes: Path, output: Path) -> dict[str, Any]:
    """Build 1,039 authorized Direct items without replaying settled E3.20 rows."""
    if output.exists():
        raise ValueError("E3.21 source destination is immutable")
    pool = _read_jsonl(candidates)
    audit = {str(row["sample_id"]): row for row in _read_jsonl(e320_source)}
    outcomes = json.loads(e320_r1_outcomes.read_text(encoding="utf-8")).get("outcomes") or []
    by_id = {str(row["sample_id"]): row for row in pool}
    if len(pool) != 1070 or len(by_id) != 1070 or len(audit) != 32:
        raise ValueError("E3.21 frozen candidate/audit scope is invalid")
    disposition = {str(row.get("sample_id")): str(row.get("disposition")) for row in outcomes}
    predecessor = {str(row.get("sample_id")): str(row.get("request_id") or "") for row in outcomes}
    if Counter(disposition.values()) != Counter({"semantic_correct": 25, "semantic_wrong": 6, "quality_reject": 1}):
        raise ValueError("E3.21 requires the verified E3.20 R1 partition")
    if set(disposition) != set(audit):
        raise ValueError("E3.21 E3.20 outcome/source binding is incomplete")
    rows=[]
    for row in pool:
        sample_id=str(row["sample_id"])
        if sample_id in audit:
            continue
        rows.append({**row, "e39_protocol": E321_PROTOCOL, "teacher_system_prompts": dict(PROMPTS)})
    repair_ids=[sample_id for sample_id, value in disposition.items() if value == "quality_reject"]
    for sample_id in repair_ids:
        row=audit[sample_id]
        rows.append({**row, "e39_protocol": E321_PROTOCOL, "teacher_system_prompts": dict(PROMPTS),
                     "e321_origin": "e320_q1_repair", "e321_predecessor_request_id":predecessor[sample_id]})
    if len(rows) != 1039 or len({str(row["sample_id"]) for row in rows}) != 1039:
        raise ValueError("E3.21 Direct scope must contain 1,039 unique items")
    if Counter(str(row.get("arm")) for row in rows) != Counter({"known":535, "simulated_unknown":504}):
        raise ValueError("E3.21 Direct arm scope is invalid")
    rows.sort(key=lambda row: str(row["sample_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in rows), encoding="utf-8")
    return {"rows":len(rows), "source_sha256":_digest(output), "reused_direct_winners":25,
            "deferred_classifier":6, "direct_q1_repairs":repair_ids}


def _controls() -> dict[str, Any]:
    return {"uncached_input_token_cap":CAP, "transport_image_max_side":512,
            "max_public_turns_per_route":12, "max_rag_searches":3,
            "reservation_uncached_tokens":RESERVATION, "recovery_rounds":["R0","R1","R2"]}


def write_direct_manifest(*, source: Path, sample_ids: list[str], campaign_id: str, output: Path, shard: str, repair: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("E3.21 manifest destination is immutable")
    rows={str(row["sample_id"]):row for row in _read_jsonl(source)}
    if not sample_ids or len(sample_ids) != len(set(sample_ids)) or any(sample_id not in rows for sample_id in sample_ids):
        raise ValueError("E3.21 manifest sample scope is invalid")
    if repair != (len(sample_ids) == 1 and rows[sample_ids[0]].get("e321_origin") == "e320_q1_repair"):
        raise ValueError("E3.21 Q1 repair scope is invalid")
    work=[]
    for sample_id in sorted(sample_ids):
        row=rows[sample_id]
        work.append({"work_id":(f"Q1:{sample_id}:e321-direct" if repair else f"R0:{sample_id}:e321-direct"),
                     "round":"Q1" if repair else "R0", "sample_id":sample_id,
                     "image_group_id":row.get("image_group_id",row["image_sha256"]),
                     "attempt_ordinal":1 if repair else 0,
                     "predecessor_request_id":(str(row.get("e321_predecessor_request_id") or "") if repair else None),
                     "route_progression":list(STAGES), "resume_route":"direct",
                     "quality_attempt_ordinal":1 if repair else 0,
                     "prompt_revision":"repair_v1" if repair else "base",
                     "direct_only_authorized":True})
    payload={"schema_version":SCHEMA, "protocol":E321_PROTOCOL, "campaign_id":campaign_id,
             "round":"Q1" if repair else "R0", "shard":shard, "direct_only":True,
             "source":str(source), "source_sha256":_digest(source), "source_rows_expected":1039,
             "work_items":work, "workers":4, "micu_intent_limit":8000,
             "automatic_replay_allowed":False, "collection_controls":_controls(),
             "quality_repair":{"max_per_stage":1,"new_request_id":True,"separate_from_delivery_recovery":True},
             "future_stage_plan":{"stages":list(STAGES),"semantic_wrong_transition":{"direct":"classifier","classifier":"rag","rag":"reject"},"executed_now":["direct"],"frozen_not_executed":["classifier","rag","reject"]},
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def shard_ids(source: Path, *, shards: int = 8) -> list[list[str]]:
    if shards != 8: raise ValueError("E3.21 freezes eight base shards")
    rows=[row for row in _read_jsonl(source) if row.get("e321_origin") != "e320_q1_repair"]
    result=[[] for _ in range(shards)]
    for row in rows:
        index=int(hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(),16) % shards
        result[index].append(str(row["sample_id"]))
    if sum(map(len,result)) != 1038 or not all(result): raise ValueError("E3.21 shard partition invalid")
    return result


def _manifest(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA or value.get("protocol") != E321_PROTOCOL:
        raise ValueError("E3.21 manifest protocol is invalid")
    if not value.get("direct_only") or any(item.get("resume_route") != "direct" for item in value.get("work_items") or []):
        raise ValueError("E3.21 manifest is not Direct-only")
    return value


def write_direct_quality_repair_manifest(*, prior_manifest: Path, outcomes: Path, output: Path) -> dict[str, Any]:
    """Freeze exactly the allowed Direct Q1 repairs; semantic errors remain deferred."""
    if output.exists(): raise ValueError("E3.21 continuation destination is immutable")
    prior=_manifest(prior_manifest); result=json.loads(outcomes.read_text(encoding="utf-8"))
    if result.get("manifest_sha256") != _digest(prior_manifest): raise ValueError("E3.21 continuation lineage mismatch")
    items={str(item["work_id"]):item for item in prior.get("work_items") or []}
    work=[]; deferred=[]
    for outcome in result.get("outcomes") or []:
        item=items.get(str(outcome.get("work_id")))
        if item is None: raise ValueError("E3.21 continuation has out-of-scope outcome")
        disposition=str(outcome.get("disposition") or "")
        if outcome.get("delivery_status") != "delivered": continue
        if disposition == "semantic_wrong":
            deferred.append({"sample_id":item["sample_id"],"predecessor_request_id":outcome.get("request_id"),"future_route":"classifier"})
        elif disposition == "quality_reject" and int(item.get("quality_attempt_ordinal",0)) == 0:
            work.append({**{key:item[key] for key in ("sample_id","image_group_id","route_progression")},
                         "work_id":f"Q1:{item['sample_id']}:e321-direct","round":"Q1","resume_route":"direct",
                         "attempt_ordinal":int(item.get("attempt_ordinal",0)),
                         "predecessor_request_id":outcome.get("request_id"),"quality_attempt_ordinal":1,
                         "prompt_revision":"repair_v1","direct_only_authorized":True,"transition_reason":"quality_reject"})
    payload={**{key:prior[key] for key in ("schema_version","protocol","campaign_id","source","source_sha256","source_rows_expected","workers","micu_intent_limit","collection_controls","quality_repair","future_stage_plan","training_eligible","training_authorized","sft_may_start")},
             "round":"Q1-continuation","continuation":True,"direct_only":True,"predecessor_manifest_sha256":_digest(prior_manifest),
             "work_items":work,"deferred_classifier_queue":deferred}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def write_direct_delivery_recovery_manifest(*, prior_manifest: Path, outcomes: Path, next_round: str, output: Path) -> dict[str, Any]:
    """Freeze only R1/R2 replacement requests after unknown delivery."""
    if next_round not in {"R1","R2"}: raise ValueError("E3.21 recovery must be R1 or R2")
    if output.exists(): raise ValueError("E3.21 recovery destination is immutable")
    prior=_manifest(prior_manifest); result=json.loads(outcomes.read_text(encoding="utf-8"))
    if result.get("manifest_sha256") != _digest(prior_manifest): raise ValueError("E3.21 recovery lineage mismatch")
    expected={"R1":{"R0","Q1","Q1-continuation"},"R2":{"R1"}}[next_round]
    if prior.get("round") not in expected: raise ValueError("E3.21 recovery round order invalid")
    items={str(item["work_id"]):item for item in prior.get("work_items") or []}; work=[]
    for outcome in result.get("outcomes") or []:
        if outcome.get("delivery_status") != "unknown_delivery": continue
        item=items.get(str(outcome.get("work_id"))); request_id=outcome.get("request_id")
        if item is None or not isinstance(request_id,str) or not request_id: raise ValueError("E3.21 unknown delivery lacks predecessor")
        work.append({**{key:item[key] for key in ("sample_id","image_group_id","route_progression")},
                     "work_id":f"{next_round}:{item['sample_id']}:e321-direct-delivery","round":next_round,"resume_route":"direct",
                     "attempt_ordinal":int(item.get("attempt_ordinal",0))+1,"predecessor_request_id":request_id,
                     "quality_attempt_ordinal":int(item.get("quality_attempt_ordinal",0)),"prompt_revision":item.get("prompt_revision","base"),"direct_only_authorized":True})
    payload={**{key:prior[key] for key in ("schema_version","protocol","campaign_id","source","source_sha256","source_rows_expected","workers","micu_intent_limit","collection_controls","quality_repair","future_stage_plan","training_eligible","training_authorized","sft_may_start")},
             "round":next_round,"delivery_recovery":True,"direct_only":True,"predecessor_manifest_sha256":_digest(prior_manifest),"work_items":work}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--e320-source",type=Path,required=True)
    parser.add_argument("--e320-r1-outcomes",type=Path,required=True)
    parser.add_argument("--source-output",type=Path,required=True)
    parser.add_argument("--manifest-dir",type=Path,required=True)
    parser.add_argument("--campaign-id",required=True)
    args=parser.parse_args(argv)
    result=materialize_direct_source(candidates=args.candidates,e320_source=args.e320_source,
        e320_r1_outcomes=args.e320_r1_outcomes,output=args.source_output)
    rows={str(row["sample_id"]):row for row in _read_jsonl(args.source_output)}
    repair=[sample_id for sample_id,row in rows.items() if row.get("e321_origin") == "e320_q1_repair"]
    write_direct_manifest(source=args.source_output,sample_ids=repair,campaign_id=args.campaign_id,
        output=args.manifest_dir / "e321-direct-q1-repair.json",shard="q1-repair",repair=True)
    shards=shard_ids(args.source_output)
    for index, ids in enumerate(shards):
        write_direct_manifest(source=args.source_output,sample_ids=ids,campaign_id=args.campaign_id,
            output=args.manifest_dir / f"e321-direct-r0-shard-{index:02d}.json",shard=f"r0-{index:02d}")
    result["base_shards"]=[len(ids) for ids in shards]
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
