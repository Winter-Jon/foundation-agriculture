import json
from pathlib import Path

from agrinet.rag import micu_classifier_hcv_e2_dynamic_recovery_rewrite as recovery_rewrite
from agrinet.rag.micu_classifier_hcv_e2_dynamic_recovery_rewrite import _patch_bound_schema, final_report, plan_replenishment, plan_r0


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_rewrite_r0_contains_only_selected_immutable_winners(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    rows = [{"sample_id": f"s{index}", "image_group_id": f"g{index}"} for index in range(32)]
    write(source, {"rows": rows})
    parent = tmp_path / "parent.json"; write(parent, {"status": "closed"})
    final = tmp_path / "final.json"
    write(final, {"campaign_id": "campaign", "groups": [
        {"image_group_id": "g0", "sample_id": "s0", "state": "selected", "winner_work_id": "winner", "winner_request_id": "old"},
        {"image_group_id": "g1", "sample_id": "s1", "state": "quality_rejected"},
    ]})
    summary = tmp_path / "summary.json"
    write(summary, {"campaign_id": "campaign", "rows": [{"work_id": "winner", "route": "direct", "parent_path": str(parent)}]})
    result = plan_r0(campaign_id="campaign", collection_final=final, collection_summary=summary, source=source, output=tmp_path / "manifest.json")
    assert len(result["work_items"]) == 1
    assert result["work_items"][0]["sample_id"] == "s0"
    assert result["work_items"][0]["predecessor_request_id"] is None


def test_rewrite_r0_finds_winner_closed_in_collection_recovery_round(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write(source, {"rows": [{"sample_id": "s" if index == 0 else f"other-{index}", "image_group_id": "g" if index == 0 else f"other-group-{index}"} for index in range(32)]})
    parent = tmp_path / "parent.json"; write(parent, {"status": "closed"})
    final = tmp_path / "final.json"
    write(final, {"campaign_id": "campaign", "groups": [{"image_group_id": "g", "sample_id": "s", "state": "selected", "winner_work_id": "R1:g:direct:1:parent_audit", "winner_request_id": "r1-request"}]})
    summaries = tmp_path / "summaries"
    from agrinet.rag.micu_classifier_hcv_v2 import sha256
    write(summaries / "r0.json", {"campaign_id": "campaign", "source_sha256": sha256(source), "rows": []})
    write(summaries / "r1.json", {"campaign_id": "campaign", "source_sha256": sha256(source), "rows": [{"work_id": "R1:g:direct:1:parent_audit", "route": "direct", "parent_path": str(parent)}]})
    write(summaries / "r2.json", {"campaign_id": "campaign", "source_sha256": sha256(source), "rows": []})
    result = plan_r0(campaign_id="campaign", collection_final=final, collection_summary=summaries / "r2.json", source=source, output=tmp_path / "manifest.json")
    assert result["work_items"][0]["parent_path"] == str(parent)
    assert result["work_items"][0]["collection_winner_request_id"] == "r1-request"


def test_rewrite_v2_r0_manifest_binds_approved_patch(tmp_path: Path) -> None:
    source = tmp_path / "source.json"; write(source, {"rows": [{"sample_id": "s" if index == 0 else f"other-{index}", "image_group_id": "g" if index == 0 else f"other-group-{index}"} for index in range(32)]})
    parent = tmp_path / "parent.json"; write(parent, {"status": "closed"})
    final = tmp_path / "final.json"; write(final, {"campaign_id": "campaign", "groups": [{"image_group_id": "g", "sample_id": "s", "state": "selected", "winner_work_id": "winner"}]})
    summary = tmp_path / "summary.json"; write(summary, {"campaign_id": "campaign", "rows": [{"work_id": "winner", "route": "direct", "parent_path": str(parent)}]})
    patch = tmp_path / "patch.json"; write(patch, {"patch_id": "0001-rewrite-structure-v2"})
    result = plan_r0(campaign_id="campaign", collection_final=final, collection_summary=summary, source=source, output=tmp_path / "manifest.json", rewrite_schema="structured-observation-comparison-v2", contract_patch=patch)
    assert result["rewrite_schema"] == "structured-observation-comparison-v2"
    assert result["contract_patch_sha256"]


def test_rewrite_replenishment_and_final_report_keep_delivery_separate_from_quality(tmp_path: Path) -> None:
    r0 = tmp_path / "r0.json"
    unresolved = {"work_id": "R0:g:rewrite", "planned_work": {"work_id": "R0:g:rewrite", "image_group_id": "g", "sample_id": "s", "scope": "rewrite", "attempt_ordinal": 0}, "delivery_status": "unknown_delivery", "new_request_id": "old-request"}
    write(r0, {"campaign_id": "campaign", "round": "R0", "source_sha256": "source", "rows": [unresolved]})
    r1_manifest = plan_replenishment(summary=r0, next_round="R1", output=tmp_path / "r1-manifest.json")
    assert r1_manifest["work_items"] == [{**unresolved["planned_work"], "work_id": "R1:g:rewrite", "round": "R1", "attempt_ordinal": 1, "predecessor_request_id": "old-request", "recovery_reason": "unknown_delivery"}]
    r1 = tmp_path / "r1.json"
    delivered_reject = {"work_id": "R1:g:rewrite", "planned_work": r1_manifest["work_items"][0], "delivery_status": "delivered", "new_request_id": "new-request", "quality_status": "quality_rejected"}
    write(r1, {"campaign_id": "campaign", "round": "R1", "source_sha256": "source", "rows": [delivered_reject]})
    r2 = tmp_path / "r2.json"; write(r2, {"campaign_id": "campaign", "round": "R2", "source_sha256": "source", "rows": []})
    collection = tmp_path / "collection.json"; write(collection, {"campaign_id": "campaign", "groups": [{"image_group_id": "g", "sample_id": "s", "state": "selected"}]})
    report = final_report(collection_final=collection, summaries=[r0, r1, r2], output=tmp_path / "final.json")
    assert report["counts"] == {"quality_rejected": 1}
    assert report["groups"][0]["rewrite_request_id"] == "new-request"


def test_audit_only_recovery_reuses_closed_rewrite(monkeypatch, tmp_path: Path) -> None:
    parent, rewrite = tmp_path / "parent.json", tmp_path / "rewrite.json"
    write(parent, {"status": "closed", "trace": []})
    write(rewrite, {"status": "closed", "rewrite": {"reasoning": "fixed"}})
    calls = []
    def audit(**kwargs):
        calls.append(kwargs)
        return {"status": "accept"}
    monkeypatch.setattr(recovery_rewrite, "run_private_audit", audit)
    item = {"work_id": "R1:g:rewrite_audit", "image_group_id": "g", "sample_id": "s",
            "scope": "rewrite_audit", "parent_path": str(parent), "rewrite_path": str(rewrite)}
    rows = {"s": {"sample_id": "s", "image_group_id": "g", "image_sha256": "image"}}
    result = recovery_rewrite._audit_one(item, source_rows=rows, output=tmp_path / "out", campaign_root=tmp_path, teacher_model="model", timeout=1)
    assert result[0]["quality_status"] == "accept"
    assert len(calls) == 1
    assert calls[0]["parent"]["reasoning_rewrite"] == {"reasoning": "fixed"}


def test_future_v2_rewrite_manifest_requires_immutable_patch_sha(tmp_path: Path) -> None:
    patch = tmp_path / "0001.json"
    write(patch, {"patch_id": "0001-rewrite-structure-v2"})
    from agrinet.rag.micu_classifier_hcv_v2 import sha256
    plan = {"rewrite_schema": "structured-observation-comparison-v2", "contract_patch": str(patch), "contract_patch_sha256": sha256(patch)}
    assert _patch_bound_schema(plan)[0] == "structured-observation-comparison-v2"
    plan["contract_patch_sha256"] = "0" * 64
    import pytest
    with pytest.raises(ValueError, match="patch binding"):
        _patch_bound_schema(plan)


def test_rewrite_v2_collector_renders_public_text_before_private_audit(monkeypatch, tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"; write(parent, {"final": "Apple scab", "trace": []})
    item = {"work_id": "R0:g:rewrite", "image_group_id": "g", "sample_id": "s", "route": "direct", "parent_path": str(parent)}
    row = {"sample_id": "s", "image_group_id": "g", "image_sha256": "image", "question": "What is visible?", "question_type": "open", "language": "en"}
    structured = {"final": "Apple scab", "visual_observations": ["brown spots", "rough lesions", "curled leaves"],
                  "candidate_comparisons": [{"candidate": "Apple scab", "support": "rough lesions", "counterevidence": "no orange pustules"}, {"candidate": "Apple rust", "support": "leaf lesions", "counterevidence": "no orange pustules"}, {"candidate": "Powdery mildew", "support": "leaf damage", "counterevidence": "no white coating"}],
                  "evidence": "visible lesion pattern", "uncertainty": "leaf underside unseen"}
    monkeypatch.setattr(recovery_rewrite, "_invoke_micu", lambda request, timeout: {"choices": [{"message": {"content": json.dumps(structured)}}]})
    audits = []
    monkeypatch.setattr(recovery_rewrite, "run_private_audit", lambda **kwargs: audits.append(kwargs) or {"status": "accept"})
    result = recovery_rewrite._rewrite_one(item, source_rows={"s": row}, registry_names=[], output=tmp_path / "out", campaign_root=tmp_path, teacher_model="gpt-5.6-sol", timeout=1, rewrite_schema="structured-observation-comparison-v2")
    saved = json.loads((tmp_path / "out/public/g/rewrite.json").read_text())
    assert result[0]["quality_status"] == "closed"
    assert saved["rewrite"]["reasoning"].startswith("Visual observations:")
    assert saved["structured_rewrite"] == structured
    assert audits[0]["parent"]["reasoning_rewrite"] == saved["rewrite"]
