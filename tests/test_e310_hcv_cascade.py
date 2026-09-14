import json
from pathlib import Path

from agrinet.rag.e310_hcv_cascade import (E310_PROTOCOL, e310_layered_audit_report,
    select_e310_audit, validate_e310_trajectory, write_e310_manifest,
    write_e310_replenishment_manifest)
from agrinet.rag.e35_cascade_collect import _e310_next_step_message, _request


def _row(i, arm="known", kind="open", domain="disease"):
    return {"sample_id": f"s-{i}", "arm":arm,"canonical_class_code":f"C{i:03d}","image_sha256":f"h-{i}","source_group_id":f"g-{i}","near_duplicate_group_id":f"n-{i}","question_type":kind,"task_domain":domain,"question":"Identify this image.","e39_protocol":E310_PROTOCOL,"teacher_system_prompts":{"direct":"d","classifier":"c","rag":"r"},"classifier":{"kind":"oof","held_out_fold":0,"label_map_codes":[f"C{i:03d}"],"checkpoint_sha256":"a"*64,"training_manifest_sha256":"b"*64,"label_map_sha256":"c"*64,"registry_sha256":"d"*64,"top5":[{"name":f"candidate-{j}","score":.9-j/10} for j in range(5)]}}


def _final():
    return "<think>Visual observations: brown lesion; irregular margin; leaf surface.\nCandidate hypotheses: candidate-0 and candidate-1.\nCandidate comparison: candidate-0 fits.\nEvidence: visible lesion.\nRejected alternatives: candidate-1: margin conflicts; candidate-2: texture conflicts.\nUncertainty: low — clear image.</think><answer>candidate-0</answer>"


def test_e310_accepts_paragraph_hcv_and_strict_rag_order():
    row=_row(1)
    trace=[{"call":{"name":"agrinet_classifier_predict"},"response":{}},{"call":{"name":"agrinet_rag_search"},"response":{}}]
    validate_e310_trajectory(row,{"route":"rag","answer":_final(),"tool_trace":trace})


def test_e310_forces_tool_only_after_preceding_plan():
    row=_row(1)
    initial=[{"role":"system","content":"r"},{"role":"user","content":"question"}]
    assert _request(row,"rag",initial,"model",64)["tool_choice"] == "auto"
    planned=initial+[{"role":"assistant","content":"<think>Plan classifier candidates.</think>"}]
    assert _request(row,"rag",planned,"model",64)["tool_choice"]["function"]["name"] == "agrinet_classifier_predict"
    native_predict={"role":"assistant","content":None,"tool_calls":[{"id":"p"}]}
    card=planned+[native_predict,{"role":"tool","tool_call_id":"p","content":"{}"}]
    assert _request(row,"rag",card,"model",64)["tool_choice"] == "none"
    rag_plan=card+[{"role":"assistant","content":"<think>Plan a discriminative RAG query.</think>"}]
    assert _request(row,"rag",rag_plan,"model",64)["tool_choice"]["function"]["name"] == "agrinet_rag_search"
    native_search={"role":"assistant","content":None,"tool_calls":[{"id":"r"}]}
    after_search=rag_plan+[native_search,{"role":"tool","tool_call_id":"r","content":"{}"}]
    assert _request(row,"rag",after_search,"model",64,rag_called=True)["tool_choice"] == "none"
    second_plan=after_search+[{"role":"assistant","content":"<think>Plan a second discriminative RAG query.</think>"}]
    assert _request(row,"rag",second_plan,"model",64,rag_called=True)["tool_choice"]["function"]["name"] == "agrinet_rag_search"
    three_searches=second_plan+[{"role":"assistant","content":None,"tool_calls":[{"function":{"name":"agrinet_rag_search"}}]} for _ in range(3)]
    capped=three_searches+[{"role":"assistant","content":"<think>Do not plan another search.</think>"}]
    assert _request(row,"rag",capped,"model",64,rag_called=True)["tool_choice"] == "none"
    instruction=_e310_next_step_message(route="rag",tool_name="agrinet_classifier_predict")
    assert instruction and "Do not answer" in instruction and "RAG query" in instruction


def test_e310_selects_fresh_audit_excluding_previous_ids(tmp_path:Path):
    rows=[]
    for arm in ("known","simulated_unknown"):
        for kind in ("open","option"):
            for domain in ("disease","pest"):
                for _ in range(8):rows.append(_row(len(rows),arm,kind,domain))
    prior={r["sample_id"] for r in rows[::5]}
    # This unit fixture deliberately has fewer than 107 classes; source-wide
    # isolation is covered by the E3.5/E3.9 builders, so patch only that gate.
    import agrinet.rag.e310_hcv_cascade as module
    original=module.validate_source_rows; module.validate_source_rows=lambda _: {"ready":True}
    try: chosen=select_e310_audit(rows,excluded_ids=prior)
    finally: module.validate_source_rows=original
    assert len(chosen)==32 and not {r["sample_id"] for r in chosen}&prior
    source=tmp_path/'source.jsonl'; source.write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    for r in chosen:r['e39_protocol']=E310_PROTOCOL
    source.write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    manifest=write_e310_manifest(source=source,campaign_id='e310',output=tmp_path/'manifest.json')
    assert manifest['collection_controls']['max_public_turns_per_route']==12
    assert manifest['schema_version'] == 'agrinet.e310-hcv-cascade-manifest/v1'


def test_e310_rejects_second_expand_and_unplanned_native_tool_call():
    row=_row(1)
    trace=[{"call":{"name":"agrinet_classifier_predict"},"response":{}},
           {"call":{"name":"agrinet_classifier_expand"},"response":{}},
           {"call":{"name":"agrinet_classifier_expand"},"response":{}}]
    try:
        validate_e310_trajectory(row,{"route":"classifier","answer":_final(),"tool_trace":trace})
    except ValueError as exc:
        assert "tool order" in str(exc)
    else:
        raise AssertionError("second classifier expansion must fail")
    messages=[{"role":"system","content":"s"},{"role":"user","content":"u"},
              {"role":"assistant","content":None,"tool_calls":[{"id":"p"}]},{"role":"tool","content":"{}"}]
    try:
        validate_e310_trajectory(row,{"route":"rag","answer":_final(),"tool_trace":[{"call":{"name":"agrinet_classifier_predict"},"response":{}},{"call":{"name":"agrinet_rag_search"},"response":{}}],"messages":messages})
    except ValueError as exc:
        assert "preceding Hermes planning" in str(exc)
    else:
        raise AssertionError("native call without standalone plan must fail")


def test_e310_recovery_replays_only_delivery_gaps_and_final_gate_stays_closed(tmp_path:Path):
    source=[]
    for i in range(32):
        arm="known" if i<16 else "simulated_unknown"
        row=_row(i,arm)
        row["private"]={"e310_route_coverage":("direct","classifier","rag")[i%3]}
        source.append(row)
    controls={"max_rag_searches":3,"max_public_turns_per_route":12}
    r0={"round":"R0","campaign_id":"audit","collection_controls":controls,"rows":[]}
    for i,row in enumerate(source):
        r0["rows"].append({"sample_id":row["sample_id"],"image_group_id":row["image_sha256"],"route_progression":["direct","classifier","rag"],"attempt_ordinal":0,"delivery_status":"unknown_delivery" if i==0 else "delivered","request_id":f"req-{i}","winner":False,"final_route":"direct","quality_status":"route_contract_reject"})
    r1_manifest=write_e310_replenishment_manifest(summary=r0,next_round="R1",output=tmp_path/"r1.json")
    assert [item["sample_id"] for item in r1_manifest["work_items"]]==[source[0]["sample_id"]]
    r1={"round":"R1","rows":[{**r0["rows"][0],"attempt_ordinal":1,"delivery_status":"unknown_delivery","request_id":"req-r1"}]}
    r2={"round":"R2","rows":[{**r1["rows"][0],"attempt_ordinal":2,"delivery_status":"unknown_delivery","request_id":"req-r2"}]}
    report=e310_layered_audit_report(source_rows=source,summaries=[r0,r1,r2])
    assert report["terminal_counts"]["delivery_shortfall"]==1
    assert report["full_campaign_remaining_candidates"]==1006
    assert report["protocol_gate_passed"] is False
