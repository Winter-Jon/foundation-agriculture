import json
from pathlib import Path

from agrinet.rag.e321_direct_first import (
    E321_PROTOCOL, materialize_direct_source, shard_ids, write_direct_manifest,
    write_direct_delivery_recovery_manifest, write_direct_quality_repair_manifest,
)
from agrinet.rag.e320_live import validate_e320_live_inputs
from agrinet.rag.e321_report import checkpoint_report
from agrinet.rag.e321_campaign import run_campaign


def _candidate(index: int, arm: str) -> dict:
    return {"sample_id":f"s-{index}","image_sha256":f"i-{index}","source_group_id":f"g-{index}",
            "near_duplicate_group_id":f"n-{index}","arm":arm,"question_type":"open","task_domain":"disease",
            "private":{"truth_code":f"N{index:05d}"},"public_options":[]}


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_e321_materializes_direct_only_scope_and_shards(tmp_path):
    pool=[_candidate(index, "known" if index < 535 else "simulated_unknown") for index in range(1070)]
    # Give the 32 audit rows their required simulated-Unknown arm.
    audit=pool[535:567]
    candidate_path=tmp_path / "candidates.jsonl"; audit_path=tmp_path / "audit.jsonl"
    _jsonl(candidate_path,pool); _jsonl(audit_path,audit)
    dispositions=["semantic_correct"]*25+["semantic_wrong"]*6+["quality_reject"]
    outcomes={"outcomes":[{"sample_id":row["sample_id"],"disposition":disposition,"request_id":f"r-{i}"} for i,(row,disposition) in enumerate(zip(audit,dispositions))]}
    out=tmp_path / "r1.json"; out.write_text(json.dumps(outcomes))
    source=tmp_path / "source.jsonl"
    report=materialize_direct_source(candidates=candidate_path,e320_source=audit_path,e320_r1_outcomes=out,output=source)
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert report["rows"] == 1039
    assert sum(row["arm"] == "known" for row in rows) == 535
    assert sum(row["arm"] == "simulated_unknown" for row in rows) == 504
    assert all(row["e39_protocol"] == E321_PROTOCOL for row in rows)
    repair=[row for row in rows if row.get("e321_origin")]
    assert len(repair) == 1 and repair[0]["e321_predecessor_request_id"] == "r-31"
    shards=shard_ids(source)
    assert len(shards) == 8 and sum(map(len,shards)) == 1038
    manifest=write_direct_manifest(source=source,sample_ids=[repair[0]["sample_id"]],campaign_id="test",output=tmp_path / "q1.json",shard="q1",repair=True)
    assert manifest["work_items"][0]["predecessor_request_id"] == "r-31"
    assert manifest["future_stage_plan"]["frozen_not_executed"] == ["classifier","rag","reject"]
    assert validate_e320_live_inputs(manifest=tmp_path / "q1.json",source=source)["ready"]


def test_e321_continuations_repair_only_quality_and_recover_only_delivery(tmp_path):
    rows=[_candidate(0, "known"), _candidate(1, "known"), _candidate(2, "known")]
    for row in rows: row["e39_protocol"] = E321_PROTOCOL
    source=tmp_path / "source.jsonl"; _jsonl(source, rows)
    prior_path=tmp_path / "prior.json"
    prior=write_direct_manifest(source=source,sample_ids=[row["sample_id"] for row in rows],campaign_id="test",output=prior_path,shard="r0")
    import hashlib
    outcomes={"manifest_sha256":hashlib.file_digest(prior_path.open("rb"),"sha256").hexdigest(),"outcomes":[
        {"work_id":prior["work_items"][0]["work_id"],"delivery_status":"delivered","disposition":"quality_reject","request_id":"quality"},
        {"work_id":prior["work_items"][1]["work_id"],"delivery_status":"delivered","disposition":"semantic_wrong","request_id":"semantic"},
        {"work_id":prior["work_items"][2]["work_id"],"delivery_status":"unknown_delivery","disposition":"delivery_unknown","request_id":"lost"},
    ]}
    result=tmp_path / "outcomes.json"; result.write_text(json.dumps(outcomes))
    repair=write_direct_quality_repair_manifest(prior_manifest=prior_path,outcomes=result,output=tmp_path / "repair.json")
    assert [item["sample_id"] for item in repair["work_items"]] == ["s-0"]
    assert repair["deferred_classifier_queue"] == [{"sample_id":"s-1","predecessor_request_id":"semantic","future_route":"classifier"}]
    recovery=write_direct_delivery_recovery_manifest(prior_manifest=prior_path,outcomes=result,next_round="R1",output=tmp_path / "r1.json")
    assert [item["sample_id"] for item in recovery["work_items"]] == ["s-2"]
    assert recovery["work_items"][0]["predecessor_request_id"] == "lost"


def test_e321_checkpoint_report_binds_complete_outcomes(tmp_path):
    row=_candidate(0, "known"); row["e39_protocol"] = E321_PROTOCOL
    source=tmp_path / "source.jsonl"; _jsonl(source,[row])
    manifest_path=tmp_path / "manifest.json"
    manifest=write_direct_manifest(source=source,sample_ids=["s-0"],campaign_id="test",output=manifest_path,shard="r0")
    public=tmp_path / "public.json"
    answer=("<think>Visual observations: dark border; irregular lesion; green tissue.\nCandidate hypotheses: leaf blight; leaf spot; rust.\nCandidate comparison: leaf blight fits; leaf spot conflicts; rust conflicts.\nEvidence: dark border is visible.\nRejected alternatives: leaf spot: rejected because visible trait conflicts with diffuse edge; rust: rejected because visible trait conflicts with absent orange pustules.\nUncertainty: medium confidence; resolution limits certainty.</think><answer>leaf blight</answer>")
    public.write_text(json.dumps({"route":"direct","answer":answer,"tool_trace":[]}))
    import hashlib
    outcome=tmp_path / "outcome.json"
    outcome.write_text(json.dumps({"manifest_sha256":hashlib.file_digest(manifest_path.open("rb"),"sha256").hexdigest(),"outcomes":[{"work_id":manifest["work_items"][0]["work_id"],"delivery_status":"delivered","disposition":"semantic_correct","parent_path":str(public)}]}))
    campaign=tmp_path / "campaign"; campaign.mkdir(); (campaign / "token_budget.jsonl").write_text("")
    report=checkpoint_report(manifest=manifest_path,source=source,outcomes=outcome,campaign_root=campaign,output=tmp_path / "report.json")
    assert report["public_validation"]["passed"]
    assert report["scope"]["arm"] == {"known":1}


def test_e321_campaign_skips_existing_direct_outcomes_and_freezes_empty_continuations(tmp_path, monkeypatch):
    rows=[]
    for index in range(8):
        row=_candidate(index, "known"); row["e39_protocol"] = E321_PROTOCOL; rows.append(row)
    source=tmp_path / "source.jsonl"; _jsonl(source,rows)
    manifests=tmp_path / "manifests"; outcomes=tmp_path / "outcomes"; reports=tmp_path / "reports"; campaign=tmp_path / "campaign"
    manifests.mkdir(); outcomes.mkdir(); campaign.mkdir()
    import hashlib
    for index,row in enumerate(rows):
        manifest_path=manifests / f"e321-direct-r0-shard-{index:02d}.json"
        manifest=write_direct_manifest(source=source,sample_ids=[row["sample_id"]],campaign_id="test",output=manifest_path,shard=f"r0-{index:02d}")
        trajectory=tmp_path / f"public-{index}.json"
        trajectory.write_text(json.dumps({"route":"direct","answer":"<think>Visual observations: dark border; irregular lesion; green tissue.\nCandidate hypotheses: leaf blight; leaf spot; rust.\nCandidate comparison: leaf blight fits; leaf spot conflicts; rust conflicts.\nEvidence: dark border is visible.\nRejected alternatives: leaf spot: rejected because visible trait conflicts with diffuse edge; rust: rejected because visible trait conflicts with absent orange pustules.\nUncertainty: medium confidence; resolution limits certainty.</think><answer>leaf blight</answer>","tool_trace":[]}))
        (outcomes / f"{manifest_path.stem}.json").write_text(json.dumps({"manifest_sha256":hashlib.file_digest(manifest_path.open("rb"),"sha256").hexdigest(),"outcomes":[{"work_id":manifest["work_items"][0]["work_id"],"delivery_status":"delivered","disposition":"semantic_correct","parent_path":str(trajectory)}]}))
    monkeypatch.setattr("agrinet.rag.e321_campaign._run_collect",lambda **kwargs: (_ for _ in ()).throw(AssertionError("existing outcomes must not be re-dispatched")))
    result=run_campaign(source=source,manifest_dir=manifests,outcome_dir=outcomes,report_dir=reports,campaign_root=campaign,budget_path=campaign / "budget.jsonl",private_registry=tmp_path / "labels.json",teacher_model="test",timeout=1)
    assert len(result["base_shards"]) == 8
    assert all(not item["direct_descendants"] for item in result["base_shards"])
    assert {p.name for p in manifests.glob("*-q1.json")} == {f"e321-direct-r0-shard-{i:02d}-q1.json" for i in range(8)}
