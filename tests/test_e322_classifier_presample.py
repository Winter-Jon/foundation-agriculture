import json
from collections import Counter

import pytest
import agrinet.rag.e322_campaign as campaign

from agrinet.rag.e35_classifier_cascade import public_classifier_card, public_teacher_input
from agrinet.rag.e322_campaign import (
    AuditContractFailure, _continuation_manifest, _generation, _one, _quality_prompt, _successor,
    _unresolved_started, final_report, run_campaign,
    validate_inputs, validate_trajectory,
)
from agrinet.rag.e322_artifact_audit import gate_decision
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.e35_private import private_trajectory_projection
from agrinet.rag.e322_presample import CELL_QUOTA, FOLD_QUOTA, RARE, prepare, select
from agrinet.rag.e322_presample import classifier_prompt


def _pool():
    rows=[]; serial=0
    for (arm,qtype,domain), count in CELL_QUOTA.items():
        for i in range(max(count*4,12)):
            fold=i%3; code=f"N{serial:05d}"; serial+=1
            sid=f"sample-{arm}-{qtype}-{domain}-{i}"
            rows.append({"sample_id":sid,"arm":arm,"question_type":qtype,"domain":domain,"canonical_class_code":code,"fold":fold if arm=="known" else None,"e3_fold":fold if arm!="known" else None,"image_sha256":f"image-{sid}","source_group_id":f"source-{sid}","near_duplicate_group_id":f"near-{sid}","classifier":{"checkpoint_sha256":f"cp-{arm}-{fold}","label_map_sha256":f"lm-{arm}","training_manifest_sha256":f"tm-{arm}-{fold}","registry_sha256":"registry","label_map_codes":[] if arm!="known" else None,"top5":[{"name":f"class-{j}","score":1/(j+1)} for j in range(5)]}})
    # Replace two generated simulated-Unknown Option-pest IDs with mandated IDs.
    rare_rows=[r for r in rows if (r["arm"],r["question_type"],r["domain"])==("simulated_unknown","option","pest")][:2]
    for row,sid in zip(rare_rows,sorted(RARE)):
        old=row["sample_id"]; row["sample_id"]=sid
        for key,prefix in (("image_sha256","image"),("source_group_id","source"),("near_duplicate_group_id","near")): row[key]=f"{prefix}-{sid}"
    return rows


def test_e322_deterministic_exact_sampling_and_rare_inclusion():
    pool=_pool(); queue={"count":525,"queue":[]}
    # Pad queue with unique source-bound rows that cannot enter a quota.
    while len(pool)<525:
        i=len(pool); pool.append({**pool[0],"sample_id":f"padding-{i}","canonical_class_code":pool[0]["canonical_class_code"],"image_sha256":f"pi-{i}","source_group_id":f"ps-{i}","near_duplicate_group_id":f"pn-{i}"})
    queue["queue"]=[{"sample_id":r["sample_id"]} for r in pool]
    first=select(queue,pool); second=select(queue,pool)
    assert [r["sample_id"] for r in first]==[r["sample_id"] for r in second]
    assert RARE <= {r["sample_id"] for r in first}
    assert Counter((r["arm"],r["question_type"],r["domain"]) for r in first)==Counter(CELL_QUOTA)
    assert Counter((r["arm"],int(r["fold"] if r["arm"]=="known" else r["e3_fold"])) for r in first)==Counter(FOLD_QUOTA)
    assert len({r["canonical_class_code"] for r in first})==32
    for key in ("sample_id","image_sha256","source_group_id","near_duplicate_group_id"): assert len({r[key] for r in first})==32


def test_e322_public_classifier_tools_do_not_leak_binding():
    row=_pool()[0]
    assert len(public_classifier_card(row)["candidates"])==3
    assert len(public_classifier_card(row,expanded=True)["candidates"])==5
    rendered=json.dumps(public_classifier_card(row,expanded=True))
    assert all(x not in rendered for x in ("checkpoint","fold","truth","label_map"))
    assert public_teacher_input({**row,"question":"q","teacher_system_prompts":{"classifier":"s"}},"classifier")["tools"]==["agrinet_classifier_predict","agrinet_classifier_expand"]


def test_e322_q1_and_delivery_lineage_are_bounded():
    prior={"sample_id":"s","attempt_ordinal":0,"quality_attempt_ordinal":0,"request_id":"req0","delivery_status":"delivered","disposition":"quality_reject"}
    q1=_successor(prior,"Q1",quality=True); assert q1["predecessor_request_id"]=="req0" and q1["quality_attempt_ordinal"]==1
    unknown={**prior,"delivery_status":"unknown_delivery","disposition":"delivery_unknown"}
    r1=_successor(unknown,"R1"); assert r1["attempt_ordinal"]==1 and r1["predecessor_request_id"]=="req0"
    r2=_successor({**r1,"delivery_status":"unknown_delivery","request_id":"req1"},"R2"); assert r2["attempt_ordinal"]==2 and r2["predecessor_request_id"]=="req1"
    with pytest.raises(ValueError): _successor(r2,"R3")


def test_e322_prepare_never_overwrites_existing_artifact(tmp_path):
    (tmp_path/"source.jsonl").write_text("owned\n")
    with pytest.raises(ValueError,match="immutable"):
        prepare(queue_path=tmp_path/"missing-queue",direct_source=tmp_path/"missing-direct",e320_source=tmp_path/"missing-e320",output_root=tmp_path,checkpoint_paths=[],label_maps=[],training_manifests=[],registry_path=tmp_path/"missing-registry")
    assert (tmp_path/"source.jsonl").read_text()=="owned\n"


def test_e322_private_projection_does_not_duplicate_image_bytes():
    trajectory={"route":"classifier","answer":"<think>x</think><answer>y</answer>","tool_trace":[],"messages":[{"role":"user","content":[{"type":"text","text":"q"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,secret"}}]}]}
    rendered=json.dumps(private_trajectory_projection(trajectory))
    assert "data:image" not in rendered
    assert "image_reference" in rendered


def test_e322_interrupted_intent_freezes_unknown_without_replay(tmp_path):
    item={"work_id":"R0:s:e322","sample_id":"s","round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0}
    path=tmp_path/"ledgers"/"R0_s_e322"/"events.jsonl"; path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(x) for x in [{"event":"binding"},{"event":"intent","key":"generation:1","kind":"generation","request_id":"req0"}])+"\n")
    frozen=_unresolved_started(item,tmp_path,{"classifier":{"checkpoint_sha256":"cp"}})
    assert frozen["delivery_status"]=="unknown_delivery" and frozen["request_id"]=="req0"
    assert _successor(frozen,"R1")["predecessor_request_id"]=="req0"


def _valid_answer(body: str) -> str:
    return ("<think>Visual observations:\n1. visible lesion; 2. visible margin; 3. visible color.\n"
            "Candidate hypotheses: alpha; beta; gamma.\nCandidate comparison: alpha best.\n"
            "Evidence: visible traits.\nRejected alternatives:\n- beta: wrong margin.\n"
            "- gamma: wrong color.\nUncertainty: medium confidence because image detail is limited."
            f"</think><answer>{body}</answer>")


def test_e322_open_prompt_and_validator_forbid_option_suffix():
    row={**_pool()[0], "question_type":"open"}
    prompt=classifier_prompt(row)
    assert "no option letter" in prompt.lower()
    assert "format-only example" not in prompt.lower()
    trajectory={"route":"classifier","answer":_valid_answer("apple black rot — A"),
                "tool_trace":[{"call":{"name":"agrinet_classifier_predict"}}],"messages":[]}
    with pytest.raises(ValueError,match="Open answer"):
        validate_trajectory(row,trajectory)


def test_e322_option_prompt_and_quality_repair_are_type_specific():
    option=next(r for r in _pool() if r["question_type"]=="option")
    option["public_options"]=[{"label":x,"name":f"class-{i}"} for i,x in enumerate("ABCD")]
    assert "<answer>example leaf blight — B</answer>" in classifier_prompt(option)
    assert "exactly the canonical public class name with no option letter" in _quality_prompt("base","open")
    assert "class name — LETTER" in _quality_prompt("base","option")


class _BudgetReport:
    def report(self): return {}


def test_e322_gate_scores_unknown_future_rag_as_stage_appropriate():
    source=[]; outcomes=[]
    for i in range(32):
        arm="known" if i<16 else "simulated_unknown"
        source.append({"sample_id":f"s{i}","arm":arm,"question_type":"open","domain":"disease",
                       "canonical_class_code":f"C{i}","fold":i%3 if arm=="known" else None,
                       "e3_fold":i%3 if arm!="known" else None,"image_sha256":f"i{i}",
                       "source_group_id":f"g{i}","near_duplicate_group_id":f"n{i}"})
        good=i<12 or 16<=i<28
        outcomes.append({"sample_id":f"s{i}","checkpoint_sha256":f"cp-{arm}-{i%3}",
                         "disposition":"semantic_correct" if i<12 else "future_rag" if good else "quality_reject",
                         "winner":i<12,"quality_attempt_ordinal":1 if not good else 0,
                         "delivery_status":"delivered","predict_calls":1,"expand_calls":0})
    report=final_report(source_rows=source,all_rounds=[{"outcomes":outcomes}],
                        queue={"queue":[{"sample_id":f"s{i}"} for i in range(32)]},budget=_BudgetReport())
    assert report["stage_appropriate_by_arm"]=={"known":12,"simulated_unknown":12}
    assert report["presample_gate_passed"] is False  # quality-exhausted rows still fail closed


class _Intents:
    def __init__(self): self.keys=[]
    def reserve(self,key): self.keys.append(key)


def test_e322_delivered_generation_parse_failure_settles_budget(tmp_path,monkeypatch):
    row={**_pool()[0],"question":"Identify this image.","image_path":"unused",
         "teacher_system_prompts":{"classifier":classifier_prompt(_pool()[0])}}
    item={"work_id":"R0:s:e322","sample_id":row["sample_id"],"round":"R0",
          "attempt_ordinal":0,"quality_attempt_ordinal":0,"prompt_revision":"base"}
    monkeypatch.setattr(campaign,"transport_image",lambda *a,**k:({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,x"}},None))
    budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=10_000)
    response={"choices":[{"message":{"content":"malformed final"}}],"usage":{"prompt_tokens":321,"cached_tokens":0}}
    with pytest.raises(ValueError,match="predict missing"):
        _generation(row,item,model="gpt-5.6-sol",teacher=lambda payload:response,budget=budget,intents=_Intents(),root=tmp_path)
    assert budget.report()["settled_uncached_input_tokens"]==321
    assert budget.report()["unknown_delivery_exposure_tokens"]==0


def test_e322_delivered_private_audit_contract_failure_is_not_unknown(tmp_path,monkeypatch):
    row={**_pool()[0],"question":"q"}
    item={"work_id":"R0:s:e322","sample_id":row["sample_id"],"round":"R0",
          "attempt_ordinal":0,"quality_attempt_ordinal":0,"predecessor_request_id":None}
    trajectory={"route":"classifier","answer":_valid_answer("class-0"),"tool_trace":[{"call":{"name":"agrinet_classifier_predict"}}],"messages":[]}
    monkeypatch.setattr(campaign,"_generation",lambda *a,**k:("generation-request",trajectory))
    monkeypatch.setattr(campaign,"_audit",lambda *a,**k:(_ for _ in ()).throw(AuditContractFailure("bad audit",request_id="audit-request")))
    outcome=_one(row,item,model="gpt-5.6-sol",teacher=None,budget=None,intents=None,root=tmp_path,names={})
    assert outcome["delivery_status"]=="delivered"
    assert outcome["disposition"]=="audit_contract_error"
    assert outcome["request_id"]=="generation-request"
    assert outcome["private_audit_request_id"]=="audit-request"


def test_e322_continuation_manifest_is_immutable_and_binds_predecessor(tmp_path):
    prior={"sample_id":"s","attempt_ordinal":0,"quality_attempt_ordinal":0,
           "request_id":"req0","delivery_status":"delivered","disposition":"quality_reject",
           "_manifest_sha256":"manifest0"}
    item=_successor(prior,"Q1",quality=True)
    path=tmp_path/"q1.json"
    first=_continuation_manifest(path,"Q1","root",[item])
    assert first["work_items"][0]["predecessor_request_id"]=="req0"
    assert first["work_items"][0]["predecessor_manifest_sha256"]=="manifest0"
    assert _continuation_manifest(path,"Q1","root",[item])==first
    with pytest.raises(ValueError,match="changed"):
        _continuation_manifest(path,"Q1","root",[])


def test_e322_v2_live_collection_requires_explicit_authorization(tmp_path):
    source=tmp_path/"source.jsonl"; source.write_text("")
    manifest=tmp_path/"manifest.json"; manifest.write_text("{}")
    monkey={"legacy_read_only":False}
    original=campaign.validate_inputs
    campaign.validate_inputs=lambda *a,**k:monkey
    try:
        with pytest.raises(ValueError,match="explicit authorization"):
            run_campaign(manifest=manifest,source=source,queue_path=tmp_path/"queue",output_root=tmp_path/"campaign",
                         private_registry=tmp_path/"private",authorize_live_collection=False)
    finally:
        campaign.validate_inputs=original


def test_e322_private_audit_recovery_reuses_frozen_generation(tmp_path,monkeypatch):
    row={**_pool()[0],"question":"q"}
    trajectory={"route":"classifier","answer":_valid_answer("class-0"),
                "tool_trace":[{"call":{"name":"agrinet_classifier_predict"}}],"messages":[],
                "image_sha256":row["image_sha256"]}
    parent=tmp_path/"parent.json"; parent.write_text(json.dumps(trajectory))
    item={"work_id":"R1:s:e322","sample_id":row["sample_id"],"round":"R1",
          "resume_operation":"private_audit","attempt_ordinal":1,"quality_attempt_ordinal":0,
          "predecessor_request_id":"audit-old","parent_path":str(parent),
          "parent_trajectory_sha256":campaign.digest(parent),"generation_request_id":"generation-old"}
    monkeypatch.setattr(campaign,"_generation",lambda *a,**k:pytest.fail("generation must not replay"))
    monkeypatch.setattr(campaign,"_audit",lambda *a,**k:{"semantic":"incorrect","quality":"pass","request_id":"audit-new"})
    outcome=_one(row,item,model="gpt-5.6-sol",teacher=None,budget=None,intents=None,root=tmp_path,names={})
    assert outcome["request_id"]=="audit-new"
    assert outcome["generation_request_id"]=="generation-old"
    assert outcome["predict_calls"]==0


def test_e322_generation_counts_each_provider_turn_as_an_intent(tmp_path,monkeypatch):
    row={**_pool()[0],"question":"Identify this image.","image_path":"unused",
         "teacher_system_prompts":{"classifier":classifier_prompt(_pool()[0])}}
    item={"work_id":"R0:s:e322","sample_id":row["sample_id"],"round":"R0",
          "attempt_ordinal":0,"quality_attempt_ordinal":0,"prompt_revision":"base"}
    monkeypatch.setattr(campaign,"transport_image",lambda *a,**k:({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,x"}},None))
    responses=iter([
        {"choices":[{"message":{"content":"<think>plan</think>"}}],"usage":{"prompt_tokens":10,"cached_tokens":0}},
        {"choices":[{"message":{"tool_calls":[{"id":"call-1","function":{"name":"agrinet_classifier_predict","arguments":"{}"}}]}}],"usage":{"prompt_tokens":20,"cached_tokens":0}},
        {"choices":[{"message":{"content":_valid_answer("class-0")}}],"usage":{"prompt_tokens":30,"cached_tokens":0}},
    ])
    intents=_Intents(); budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=10_000)
    _generation(row,item,model="gpt-5.6-sol",teacher=lambda payload:next(responses),budget=budget,intents=intents,root=tmp_path)
    assert intents.keys==[f"{item['work_id']}:generation:{i}" for i in (1,2,3)]


def test_e322_final_gate_binds_report_and_artifact_audit(tmp_path):
    report=tmp_path/"report.json"; audit=tmp_path/"audit.json"; output=tmp_path/"gate.json"
    report.write_text(json.dumps({"protocol":campaign.PROTOCOL,"campaign_gate_candidate":True}))
    audit.write_text(json.dumps({"protocol":campaign.PROTOCOL,"artifact_gate_passed":True}))
    decision=gate_decision(report=report,audit_report=audit,output=output)
    assert decision["presample_gate_passed"] is True
    assert decision["final_report_sha256"]==campaign.digest(report)
    assert decision["artifact_audit_sha256"]==campaign.digest(audit)
