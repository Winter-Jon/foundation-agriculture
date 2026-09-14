import json
from collections import Counter

from agrinet.rag.e317_all_unknown_rag_audit import (
    E317_PROTOCOL, FOLD_TARGETS, e317_final_report, materialize_e317_source, select_e317_all_unknown,
    write_e317_manifest,
)
from agrinet.rag.e35_cascade_collect import _contract_error_code


def _row(index, *, fold, kind, domain):
    return {
        "sample_id": f"sample-{index}", "image_sha256": f"image-{index}",
        "source_group_id": f"source-{index}", "near_duplicate_group_id": f"near-{index}",
        "arm": "simulated_unknown", "question_type": kind, "task_domain": domain,
        "classifier": {"held_out_fold": fold}, "private": {},
    }


def test_e317_selects_32_new_unknown_rows_balanced_over_all_three_classifiers(tmp_path):
    rows=[]
    index=0
    for kind in ("open", "option"):
        for domain in ("disease", "pest"):
            for fold in range(3):
                for _ in range(12):
                    rows.append(_row(index, fold=fold, kind=kind, domain=domain)); index += 1
    previous=[_row(999, fold=0, kind="open", domain="disease")]
    selected=select_e317_all_unknown(rows, prior_rows=previous)
    assert len(selected) == 32
    assert Counter((row["question_type"], row["task_domain"]) for row in selected) == Counter({
        ("open", "disease"): 8, ("open", "pest"): 8,
        ("option", "disease"): 8, ("option", "pest"): 8})
    assert Counter(row["classifier"]["held_out_fold"] for row in selected) == Counter(FOLD_TARGETS)
    for kind, domain in (("open", "disease"), ("open", "pest"), ("option", "disease"), ("option", "pest")):
        assert {row["classifier"]["held_out_fold"] for row in selected if (row["question_type"], row["task_domain"]) == (kind, domain)} == {0, 1, 2}
    source_rows=materialize_e317_source(selected)
    assert all(row["e39_protocol"] == E317_PROTOCOL for row in source_rows)
    assert all(row["private"]["e317_route_coverage"] == "rag" for row in source_rows)
    assert all("e317_route_coverage" not in json.dumps(row["teacher_system_prompts"]) for row in source_rows)
    source=tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest=write_e317_manifest(source=source, campaign_id="e317-test", output=tmp_path / "manifest.json")
    assert manifest["all_simulated_unknown"] is True
    assert manifest["classifier_fold_targets"] == FOLD_TARGETS
    assert manifest["collection_controls"]["uncached_input_token_cap"] == 1200000
    assert manifest["training_eligible"] is False and manifest["sft_may_start"] is False


def test_e317_delivered_validator_error_is_not_replayable_delivery_ambiguity():
    assert _contract_error_code(ValueError("E3.17 RAG evidence linkage is incomplete")) == "e317_rag_contract"
    # In the collector every ValueError occurs after a durable provider
    # response; unrecognized inherited HCV text must be a contract terminal.
    assert _contract_error_code(ValueError("unexpected controller validation detail")) is None


def test_e317_final_report_counts_budget_as_delivery_shortfall():
    source=[_row(index, fold=index % 3, kind="open", domain="disease") for index in range(32)]
    # Use complete synthetic R0/R1/R2 scope: R0 closes every source, later
    # rounds are empty because no delivery ambiguity remains.
    r0={"round": "R0", "rows": [{"sample_id": row["sample_id"], "delivery_status": "budget_shortfall", "final_route": "rag", "winner": False} for row in source]}
    report=e317_final_report(source_rows=source, summaries=[r0, {"round": "R1", "rows": []}, {"round": "R2", "rows": []}])
    assert report["terminal_counts"] == {"delivery_shortfall": 32}
    assert report["latest_delivery_statuses"] == {"budget_shortfall": 32}
