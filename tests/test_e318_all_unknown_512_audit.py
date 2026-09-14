import json

from agrinet.rag.e318_all_unknown_512_audit import (
    E318_PROTOCOL, E318_TOKEN_CAP, freeze_e318_summary, materialize_e318_source,
    e318_final_report, write_e318_manifest, write_e318_replenishment_manifest,
)


def _row(index, fold):
    return {
        "sample_id": f"sample-{index}", "image_sha256": f"image-{index}",
        "source_group_id": f"source-{index}", "near_duplicate_group_id": f"near-{index}",
        "arm": "simulated_unknown", "question_type": "open" if index % 2 else "option",
        "task_domain": "disease" if index % 4 < 2 else "pest",
        "classifier": {"held_out_fold": fold}, "private": {},
    }


def test_e318_512_materialization_is_private_and_manifest_freezes_budget(tmp_path):
    # Four source cells, 8 rows/cell and folds 11/11/10.
    rows=[]
    for index in range(32):
        fold=0 if index < 11 else (1 if index < 22 else 2)
        row=_row(index, fold)
        row["question_type"]=("open", "open", "option", "option")[index // 8]
        row["task_domain"]=("disease", "pest", "disease", "pest")[index // 8]
        rows.append(row)
    source_rows=materialize_e318_source(rows)
    assert all(row["e39_protocol"] == E318_PROTOCOL for row in source_rows)
    assert all(row["private"]["e318_route_coverage"] == "rag" for row in source_rows)
    assert all("e318_route_coverage" not in json.dumps(row["teacher_system_prompts"]) for row in source_rows)
    source=tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest=write_e318_manifest(source=source, campaign_id="e318-test", output=tmp_path / "manifest.json")
    assert manifest["collection_controls"]["transport_image_max_side"] == 512
    assert manifest["collection_controls"]["uncached_input_token_cap"] == E318_TOKEN_CAP
    assert manifest["training_eligible"] is False and manifest["sft_may_start"] is False


def test_e318_r1_replays_only_unresolved_delivery_under_same_512_controls(tmp_path):
    rows=[]
    for index in range(32):
        fold=0 if index < 11 else (1 if index < 22 else 2)
        row=_row(index, fold); row["question_type"]=("open", "open", "option", "option")[index // 8]; row["task_domain"]=("disease", "pest", "disease", "pest")[index // 8]; rows.append(row)
    source=tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in materialize_e318_source(rows)))
    manifest_path=tmp_path / "r0.json"
    manifest=write_e318_manifest(source=source,campaign_id="e318-test",output=manifest_path)
    import hashlib
    outcomes=tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({"manifest_sha256":hashlib.file_digest(manifest_path.open("rb"),"sha256").hexdigest(),"outcomes":[{"work_id":item["work_id"],"delivery_status":"unknown_delivery" if item["sample_id"] in {"sample-1","sample-3"} else "delivered","request_id":f"id-{item['sample_id']}","winner":False,"final_route":"rag"} for item in manifest["work_items"]]}))
    summary=freeze_e318_summary(manifest=manifest_path,outcomes=outcomes,output=tmp_path / "summary.json")
    r1=write_e318_replenishment_manifest(summary=summary,next_round="R1",output=tmp_path / "r1.json")
    assert {item["sample_id"] for item in r1["work_items"]} == {"sample-1","sample-3"}
    assert all(item["attempt_ordinal"] == 1 and item["predecessor_request_id"].startswith("id-") for item in r1["work_items"])
    assert r1["collection_controls"]["transport_image_max_side"] == 512


def test_e318_final_report_keeps_delivery_and_budget_separate(tmp_path):
    rows=[]
    for index in range(32):
        fold=0 if index < 11 else (1 if index < 22 else 2)
        row=_row(index, fold); row["question_type"]=("open", "open", "option", "option")[index // 8]; row["task_domain"]=("disease", "pest", "disease", "pest")[index // 8]; rows.append(row)
    source=materialize_e318_source(rows)
    r0={"round":"R0", "rows":[{"sample_id":row["sample_id"], "delivery_status":"unknown_delivery" if row["sample_id"] == "sample-0" else "delivered", "final_route":"rag", "quality_status":"route_contract_reject", "winner":False} for row in source]}
    r1={"round":"R1", "rows":[{"sample_id":"sample-0", "delivery_status":"unknown_delivery", "final_route":"rag", "winner":False}]}
    r2={"round":"R2", "rows":[{"sample_id":"sample-0", "delivery_status":"unknown_delivery", "final_route":"rag", "winner":False}]}
    report=e318_final_report(source_rows=source, summaries=[r0,r1,r2], token_budget={"uncached_input_token_cap":E318_TOKEN_CAP, "committed_uncached_input_tokens":1}, provider_usage={"prompt_tokens":100, "cached_tokens":10})
    assert report["terminal_counts"] == {"route_contract_reject":31, "delivery_shortfall":1}
    assert report["provider_usage"]["cache_rate_of_prompt_tokens"] == 0.1
