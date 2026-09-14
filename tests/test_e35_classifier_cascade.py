from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from agrinet.rag.e35_classifier_cascade import (
    build_candidate_source, candidate_report, next_route, public_teacher_input, recovery_attempt,
    freeze_round_summary, initial_manifest, replenishment_manifest, select_audit, validate_contract,
    cascade_outcome, hermes_candidate, public_classifier_card, validate_rag_terminal, validate_source_rows,
)
from agrinet.rag.e35_classifier_cascade import audit_final_report, campaign_report, delivery_reauthorization_final_report, delivery_reauthorization_manifest, full_campaign_manifest_after_audit, prepare_scoring_scopes
from agrinet.rag.e35_cascade_collect import _append_native_tool_exchange, _initial_messages, _ledger_payload, _request, run_work_item
from agrinet.rag.e35_artifacts import export_candidates
from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.e35_budget import BudgetExhausted, BudgetObservedOverrun, E35TokenBudget
from agrinet.rag.e35_transport import TRANSPORT_IMAGE_VERSION, transport_image
from agrinet.rag.e35_private import private_parent_request, private_trajectory_projection, run_private_parent
from agrinet.rag.e35_rewrite import rewrite_winner
from agrinet.rag.e35_classifier_cascade import V9_COLLECTION_CONTROLS, v9_continuation_manifest


def _bound_row(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return _rows()[0] | {"system_prompt": "one English prompt", "question": "Identify it", "image_path": str(image)}


@pytest.fixture
def contract():
    return yaml.safe_load(Path("configs/sampling/e35-dual-arm-rag-distill-contract-v1.yaml").read_text())


def _rows():
    rows = []
    for arm in ("known", "simulated_unknown"):
        for class_index in range(107):
            code = f"C{class_index:03d}"
            for number, question_type in enumerate(("open", "open", "open", "option", "option")):
                token = f"{arm}-{code}-{number}"
                rows.append({
                    "sample_id": token, "arm": arm, "canonical_class_code": code,
                    "image_sha256": f"sha-{token}", "source_group_id": f"source-{token}",
                    "near_duplicate_group_id": f"near-{token}", "question_type": question_type,
                    "task_domain": "disease" if class_index % 2 == 0 else "pest",
                    "classifier": ({"kind": "oof", "held_out_fold": class_index % 3}
                                   if arm == "known" else {"kind": "classfold", "label_map_codes": ["other"]})
                                  | {"checkpoint_sha256": "a" * 64, "training_manifest_sha256": "b" * 64,
                                     "label_map_sha256": "c" * 64, "registry_sha256": "d" * 64,
                                     "top5": [{"name": f"candidate-{rank}", "score": 0.5 - rank / 100}
                                              for rank in range(5)]},
                })
    return rows


def test_contract_and_dual_arm_1070_candidate_pool(contract):
    validate_contract(contract)
    report = candidate_report(_rows())
    assert report["ready"] and report["rows"] == 1070
    assert report["per_arm"] == {"known": 535, "simulated_unknown": 535}
    assert report["training_eligible"] is False and report["sft_may_start"] is False


def test_route_specific_teacher_prompts_and_full_tool_schemas(tmp_path: Path):
    row = _bound_row(tmp_path) | {"teacher_system_prompts": {
        "direct": "<think>Direct Hermes route</think>",
        "classifier": "<think>Classifier Hermes route</think>",
        "rag": "<think>RAG Hermes route; call RAG first</think>",
    }}
    direct = _request(row, "direct", _initial_messages(row, "direct"), "test", 64)
    classifier = _request(row, "classifier", _initial_messages(row, "classifier"), "test", 64)
    rag_messages = _initial_messages(row, "rag")
    rag = _request(row, "rag", rag_messages, "test", 64)
    assert direct["messages"][0]["content"] == row["teacher_system_prompts"]["direct"] and "tools" not in direct
    assert classifier["messages"][0]["content"] == row["teacher_system_prompts"]["classifier"]
    assert [item["function"]["name"] for item in classifier["tools"]] == ["agrinet_classifier_predict", "agrinet_classifier_expand"]
    assert classifier["tools"][1]["function"]["parameters"]["required"] == ["reason"]
    assert rag["messages"][0]["content"] == row["teacher_system_prompts"]["rag"]
    assert [item["function"]["name"] for item in rag["tools"]] == ["agrinet_classifier_predict", "agrinet_classifier_expand", "agrinet_rag_search"]
    assert rag["tools"][2]["function"]["parameters"]["required"] == ["query", "retrieval_type", "rationale"]
    assert rag["tool_choice"] == {"type": "function", "function": {"name": "agrinet_rag_search"}}
    # The controller, not response text, owns the transition to auto choice.
    assert _request(row, "rag", rag_messages, "test", 64, rag_called=True)["tool_choice"] == "auto"


def test_v9_budget_is_atomic_and_unknown_reservations_remain_charged(tmp_path: Path):
    budget = E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=100)
    budget.reserve("unknown", uncached_input_tokens=60, metadata={})
    with pytest.raises(BudgetExhausted):
        budget.reserve("blocked", uncached_input_tokens=41, metadata={})
    budget.reserve("delivered", uncached_input_tokens=40, metadata={})
    budget.settle("delivered", uncached_input_tokens=12)
    assert budget.report() == {"uncached_input_token_cap": 100, "reserved_uncached_input_tokens": 100,
                               "settled_uncached_input_tokens": 12, "unknown_delivery_exposure_tokens": 60,
                               "committed_uncached_input_tokens": 72, "remaining_uncached_input_tokens": 28}


def test_v9_budget_allows_per_call_estimate_overrun_within_campaign_cap(tmp_path: Path):
    budget = E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=100)
    budget.reserve("audit", uncached_input_tokens=30, metadata={})
    budget.settle("audit", uncached_input_tokens=31)
    assert budget.report()["settled_uncached_input_tokens"] == 31


def test_v9_budget_stops_when_observed_usage_exceeds_campaign_cap(tmp_path: Path):
    budget = E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=100)
    budget.reserve("audit", uncached_input_tokens=30, metadata={})
    with pytest.raises(BudgetObservedOverrun):
        budget.settle("audit", uncached_input_tokens=101)


def test_private_projection_replaces_data_url_with_single_image_reference(tmp_path: Path):
    from PIL import Image
    image = tmp_path / "real.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image)
    row = _rows()[0] | {"system_prompt": "one English prompt", "question": "Identify it", "image_path": str(image)}
    trajectory = {"route": "classifier", "answer": "candidate 0", "tool_calls": [], "tool_trace": [],
                  "messages": [{"role": "user", "content": [{"type": "text", "text": "question"},
                  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,very-large-but-public"}}]}]}
    projected = private_trajectory_projection(trajectory)
    rendered = json.dumps(projected)
    assert "data:image" not in rendered
    assert projected["messages"][0]["content"][1] == {"type": "image_reference", "image_sha256": "bound"}
    row["private"] = {"truth_code": "C000", "truth_name": "Private canonical name"}
    request = private_parent_request(row, trajectory, model="test", transport_max_side=1024)
    assert request["messages"][1]["content"][0]["text"].count("data:image") == 0


def test_v9_transport_view_is_deterministic_and_keeps_original_identity(tmp_path: Path):
    from PIL import Image
    image = tmp_path / "large.png"
    Image.new("RGB", (4000, 2000), (5, 25, 45)).save(image)
    _, first = transport_image(image, max_side=1024)
    _, second = transport_image(image, max_side=1024)
    assert first == second
    assert first["version"] == TRANSPORT_IMAGE_VERSION and first["size"] == [1024, 512]


def test_v9_manifest_freezes_only_4_classifier_and_6_rag_continuations(tmp_path: Path):
    source = _rows()
    for row in source: row["image_group_id"] = row["image_sha256"]
    selected = [
        {"sample_id": source[index]["sample_id"], "resume_route": "classifier", "prior_state": "direct_confirmed_private_reject", "predecessor_request_id": f"d-{index}"}
        for index in range(4)
    ] + [
        {"sample_id": source[index]["sample_id"], "resume_route": "rag", "prior_state": "rag_unknown_delivery", "predecessor_request_id": f"r-{index}"}
        for index in range(4, 10)
    ]
    source_path, continuation_path, output = tmp_path / "source.jsonl", tmp_path / "continuation.json", tmp_path / "manifest.json"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source))
    continuation_path.write_text(json.dumps(selected))
    manifest = v9_continuation_manifest(source_rows=source, continuation=selected, source_path=source_path, continuation_path=continuation_path, campaign_id="v9", output=output)
    assert manifest["collection_controls"] == V9_COLLECTION_CONTROLS
    assert [item["resume_route"] for item in manifest["work_items"]].count("classifier") == 4
    assert [item["resume_route"] for item in manifest["work_items"]].count("rag") == 6


def test_v9_manifest_allows_explicit_local_rag_tool_repair(tmp_path: Path):
    source = _rows()[:6]
    for row in source:
        row["image_group_id"] = row["image_sha256"]
    continuation = [{"sample_id": row["sample_id"], "resume_route": "rag",
                     "prior_state": "rag_local_tool_shortfall", "predecessor_request_id": f"delivered-{i}"}
                    for i, row in enumerate(source)]
    source_path, continuation_path, output = tmp_path / "source.jsonl", tmp_path / "continuation.json", tmp_path / "manifest.json"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source))
    continuation_path.write_text(json.dumps(continuation))
    manifest = v9_continuation_manifest(source_rows=source, continuation=continuation, source_path=source_path,
                                        continuation_path=continuation_path, campaign_id="local-rag-repair", output=output)
    assert manifest["source_rows_expected"] == 6
    assert {row["prior_state"] for row in manifest["work_items"]} == {"rag_local_tool_shortfall"}


def test_v9_manifest_allows_four_rag_budget_repair_samples(tmp_path: Path):
    source = _rows()[:4]
    for row in source:
        row["image_group_id"] = row["image_sha256"]
    continuation = [{"sample_id": row["sample_id"], "resume_route": "rag",
                     "prior_state": "rag_budget_reservation_overrun", "predecessor_request_id": f"delivered-{i}"}
                    for i, row in enumerate(source)]
    source_path, continuation_path, output = tmp_path / "source.jsonl", tmp_path / "continuation.json", tmp_path / "manifest.json"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source))
    continuation_path.write_text(json.dumps(continuation))
    manifest = v9_continuation_manifest(source_rows=source, continuation=continuation, source_path=source_path,
                                        continuation_path=continuation_path, campaign_id="budget-repair", output=output)
    assert manifest["source_rows_expected"] == 4


def test_live_adapter_allows_frozen_four_row_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from agrinet.rag import e35_live
    manifest = {"schema_version": "agrinet.e35-cascade-manifest/v2", "round": "R0", "audit_only": True,
                "continuation": True, "source_rows_expected": 4, "work_items": [],
                "collection_controls": {"uncached_input_token_cap": 200000, "transport_image_max_side": 1024,
                                        "max_public_turns_per_route": 2,
                                        "reservation_uncached_tokens": {"generation": {"classifier": 2742, "rag": 2775}, "private_audit": 20000}}}
    manifest_path, registry, source = tmp_path / "manifest.json", tmp_path / "registry.jsonl", tmp_path / "source.jsonl"
    manifest_path.write_text(json.dumps(manifest))
    registry.write_text(json.dumps({"canonical_code": "C000", "canonical_english_name": "name"}) + "\n")
    source.write_text("")
    monkeypatch.setattr(e35_live, "local_rag_health", lambda _endpoint: {"status": "ok"})
    assert e35_live.main(["--manifest", str(manifest_path), "--source", str(source),
                          "--output", str(tmp_path / "out.json"), "--output-root", str(tmp_path / "root"),
                          "--rag-endpoint", "http://127.0.0.1:8077", "--private-registry", str(registry), "--dry-run"]) == 0


def test_pool_rejects_identity_reuse_or_unknown_label_map():
    rows = _rows()
    rows[-1]["image_sha256"] = rows[0]["image_sha256"]
    rows[-2]["classifier"]["label_map_codes"] = [rows[-2]["canonical_class_code"]]
    report = validate_source_rows(rows)
    assert not report["ready"]
    assert "identity overlap: image_sha256" in report["errors"]
    assert "simulated Unknown truth is visible to classifier" in report["errors"]


def test_audit_selection_is_deterministic_and_stays_in_final_pool():
    rows = _rows()
    chosen = select_audit(rows)
    assert len(chosen) == 32
    assert {row["arm"] for row in chosen[:16]} == {"known"}
    assert {row["arm"] for row in chosen[16:]} == {"simulated_unknown"}
    assert [row["sample_id"] for row in chosen] == [row["sample_id"] for row in select_audit(list(reversed(rows)))]
    assert set(__import__("collections").Counter((row["arm"], row["question_type"], row["task_domain"]) for row in chosen).values()) == {4}
    fresh = select_audit(rows, seed="e35-audit-v3", exclude_ids={row["sample_id"] for row in chosen})
    assert len(fresh) == 32 and {row["sample_id"] for row in fresh}.isdisjoint({row["sample_id"] for row in chosen})


def test_private_rag_witness_is_deterministic_and_never_enters_public_projection():
    rows = _rows()
    for row in rows:
        row["private"] = {"truth_code": row["canonical_class_code"]}
    chosen = select_audit(rows, seed="e35-audit-v5", rag_witnesses_per_arm=1)
    witnesses = [row for row in chosen if row.get("private", {}).get("audit_protocol", {}).get("rag_witness")]
    assert len(witnesses) == 2 and {row["arm"] for row in witnesses} == {"known", "simulated_unknown"}
    for row in witnesses:
        row.update({"system_prompt": "one English prompt", "question": "Identify it"})
        assert "rag_witness" not in str(public_teacher_input(row, "rag"))
    repeated_rows = _rows()
    for row in repeated_rows:
        row["private"] = {"truth_code": row["canonical_class_code"]}
    repeated = select_audit(repeated_rows, seed="e35-audit-v5", rag_witnesses_per_arm=1)
    assert [row["sample_id"] for row in witnesses] == [row["sample_id"] for row in repeated if row.get("private", {}).get("audit_protocol", {}).get("rag_witness")]


def test_source_builder_assigns_one_question_form_per_selected_image():
    pools = {"known": [], "simulated_unknown": []}
    for row in _rows():
        item = deepcopy(row); item.pop("question_type")
        # Five rows per class are enough for this deterministic fixture.
        pools[item["arm"]].append(item)
    registry = {f"C{index:03d}": {"name": f"public class {index}", "domain": "disease"}
                for index in range(107)}
    frozen = build_candidate_source(pools, registry=registry)
    assert candidate_report(frozen)["ready"]
    assert [r["question_type"] for r in frozen[:5]] == ["open", "open", "open", "option", "option"]
    option = frozen[3]
    assert len(option["public_options"]) == 4
    assert "correct_option" not in str({key: option[key] for key in option if key != "private"})


def test_public_projection_and_cascade_require_private_rejection():
    row = {"system_prompt": "one English prompt", "question": "What is this?",
           "image_sha256": "image", "canonical_class_code": "secret",
           "private": {"fold": 2, "audit": "secret"}, "image_path": "labelled/path"}
    public = public_teacher_input(row, "rag")
    assert public["tools"][-1] == "agrinet_rag_search"
    assert "secret" not in str(public) and "path" not in str(public)
    assert next_route(current="direct", parent_delivery="delivered", private_audit="accept", route_contract_ok=False) is None
    assert next_route(current="direct", parent_delivery="delivered", private_audit="reject", route_contract_ok=True) == "classifier"
    assert next_route(current="classifier", parent_delivery="unknown_delivery", private_audit="reject", route_contract_ok=True) is None
    row["classifier"] = {"top5": [{"name": f"candidate-{i}", "score": 0.2} for i in range(5)],
                         "held_out_fold": 1, "checkpoint_sha256": "secret"}
    card = public_classifier_card(row)
    assert len(card["candidates"]) == 3 and "fold" not in str(card).casefold() and "secret" not in str(card)


def test_public_classifier_card_rounds_scores_half_up_to_three_decimal_places():
    row = {"classifier": {"top5": [
        {"name": "candidate-1", "score": 0.9995},
        {"name": "candidate-2", "score": 0.2345},
        {"name": "candidate-3", "score": 0.12349},
        {"name": "candidate-4", "score": 0.0105},
        {"name": "candidate-5", "score": 0.0004},
    ]}}

    assert [item["score"] for item in public_classifier_card(row)["candidates"]] == [1.0, 0.235, 0.123]
    assert [item["score"] for item in public_classifier_card(row, expanded=True)["candidates"]] == [
        1.0, 0.235, 0.123, 0.011, 0.0,
    ]


def test_private_truth_name_stays_out_of_public_teacher_projection(tmp_path: Path):
    row = _bound_row(tmp_path)
    row["private"] = {"truth_code": "C000", "truth_name": "Private canonical name"}
    request = private_parent_request(row, {"route": "direct", "answer": "candidate 0"}, model="test")
    assert "Private canonical name" in str(request)
    assert "Private canonical name" not in str(public_teacher_input(row, "direct"))
    row["private"].pop("truth_name")
    with pytest.raises(ValueError, match="truth/image binding"):
        private_parent_request(row, {"route": "direct", "answer": "candidate 0"}, model="test")


def test_private_rag_witness_rejects_pre_rag_and_requires_actual_rag_response(tmp_path: Path):
    row = _bound_row(tmp_path)
    row["private"] = {"truth_code": "C000", "truth_name": "Private canonical name",
                      "audit_protocol": {"rag_witness": True}}
    accepted = {"choices": [{"message": {"content": json.dumps({"decision": "accept", "reason": "correct", "rag_evidence_insufficient": False})}}]}
    with pytest.raises(ValueError, match="must reject pre-RAG"):
        run_private_parent(row, {"route": "direct", "answer": "candidate 0", "tool_trace": []}, call=lambda _: accepted, model="test")
    with pytest.raises(ValueError, match="actual RAG call and response"):
        run_private_parent(row, {"route": "rag", "answer": "candidate 0", "tool_trace": []}, call=lambda _: accepted, model="test")
    result = run_private_parent(row, {"route": "rag", "answer": "candidate 0", "tool_trace": [{"call": {"name": "agrinet_rag_search"}, "response": {"tool": "agrinet_rag_search", "evidence": ["public"]}}]}, call=lambda _: accepted, model="test")
    assert result["decision"] == "accept"


def test_live_controller_escalates_only_after_private_rejection(tmp_path: Path):
    row = _bound_row(tmp_path)
    calls = iter([
        {"id": "direct", "choices": [{"message": {"content": "candidate 0"}}]},
        {"id": "predict", "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]},
        {"id": "classifier", "choices": [{"message": {"content": "candidate 0"}}]},
    ])
    audits = iter([{"decision": "reject"}, {"decision": "accept"}])
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"}, row=row, output_root=tmp_path, teacher=lambda _: next(calls), private_audit=lambda *_: next(audits), rag_search=lambda *_: {}, intent_limit=20)
    assert outcome["winner"] and outcome["final_route"] == "classifier"
    events = (tmp_path / "global_micu_intents.jsonl").read_text().splitlines()
    assert any("private-audit" in event for event in events)


def test_live_controller_preserves_native_tool_exchange_on_continuation(tmp_path: Path):
    row = _bound_row(tmp_path)
    requests = []
    responses = iter([
        {"id": "direct", "choices": [{"message": {"content": "candidate 0"}}]},
        {"id": "predict", "choices": [{"message": {"content": None, "tool_calls": [{"id": "call-native-1", "type": "function", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]},
        {"id": "classifier", "choices": [{"message": {"content": "candidate 0"}}]},
    ])
    def teacher(request):
        requests.append(deepcopy(request))
        return next(responses)
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"},
                            row=row, output_root=tmp_path, teacher=teacher,
                            private_audit=lambda *_: {"decision": "reject" if len(requests) == 1 else "accept"},
                            rag_search=lambda *_: {}, intent_limit=20)
    assert outcome["winner"] and outcome["final_route"] == "classifier"
    continuation = requests[2]["messages"]
    assert continuation[-2] == {"role": "assistant", "content": None, "tool_calls": [{"id": "call-native-1", "type": "function", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}
    assert continuation[-1]["role"] == "tool"
    assert continuation[-1]["tool_call_id"] == "call-native-1"


def test_native_tool_identity_mismatch_is_route_contract_reject_and_ledger_keeps_ids(tmp_path: Path):
    messages = []
    raw = {"choices": [{"message": {"tool_calls": [{"id": "call-real", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]}
    with pytest.raises(ValueError, match="identity changed"):
        _append_native_tool_exchange(messages, raw, {"tool_call_id": "call-other"}, {})
    payload = _ledger_payload({"messages": [{"role": "assistant", "content": None, "tool_calls": raw["choices"][0]["message"]["tool_calls"]}, {"role": "tool", "tool_call_id": "call-real", "content": "{}"}]}, sample_id="s", route="classifier", turn=2)
    assert payload["messages"][-2]["tool_calls"][0]["id"] == "call-real"
    assert payload["messages"][-1]["tool_call_id"] == "call-real"


def test_live_controller_does_not_escalate_unknown_delivery(tmp_path: Path):
    row = _bound_row(tmp_path)
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"}, row=row, output_root=tmp_path, teacher=lambda _: (_ for _ in ()).throw(RuntimeError("network")), private_audit=lambda *_: {"decision": "accept"}, rag_search=lambda *_: {}, intent_limit=20)
    assert not outcome["winner"] and outcome["delivery_status"] == "unknown_delivery"


def test_live_controller_records_local_rag_failure_as_tool_shortfall(tmp_path: Path):
    row = _bound_row(tmp_path)
    rag_call = {"id": "rag-first", "choices": [{"message": {"tool_calls": [{
        "id": "r1", "function": {"name": "agrinet_rag_search", "arguments": json.dumps({
            "query": "visible leaf lesion", "rationale": "retrieve evidence", "retrieval_type": "visual"})}}]}}]}
    outcome = run_work_item(
        work_item={"work_id": "R0:s:rag", "sample_id": row["sample_id"], "round": "R0", "resume_route": "rag"},
        row=row, output_root=tmp_path, teacher=lambda _: rag_call, private_audit=lambda *_: {"decision": "accept"},
        rag_search=lambda *_: (_ for _ in ()).throw(RuntimeError("HTTP 500")), intent_limit=20)
    assert not outcome["winner"]
    assert outcome["delivery_status"] == "delivered"
    assert outcome["quality_status"] == "local_rag_failure"
    assert outcome["final_route"] == "rag"
    events = [json.loads(line) for line in (tmp_path / "ledgers" / "R0_s_rag" / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("result") == 1
    assert [event["key"] for event in events if event["event"] == "intent"] == ["rag:generation:1"]


def test_recovery_attempts_keep_public_parent_and_private_audit_context_separate(tmp_path: Path):
    row = _bound_row(tmp_path)
    contexts = []
    def collect(work_id: str, attempt: int):
        response = {"id": work_id, "choices": [{"message": {"content": "candidate 0"}}]}
        return run_work_item(work_item={"work_id": work_id, "sample_id": row["sample_id"],
                                       "round": f"R{attempt}", "attempt_ordinal": attempt}, row=row,
                             output_root=tmp_path, teacher=lambda _: response,
                             private_audit=lambda _, __, context: contexts.append(context) or {"decision": "accept"},
                             rag_search=lambda *_: {}, intent_limit=20)
    r0 = collect("R0:s:cascade", 0)
    r1 = collect("R1:s:cascade", 1)
    assert r0["winner"] and r1["winner"]
    assert r0["parent_path"] != r1["parent_path"]
    assert {item["attempt_ordinal"] for item in contexts} == {0, 1}
    assert {item["work_id"] for item in contexts} == {"R0:s:cascade", "R1:s:cascade"}


def test_e35_ledger_persists_request_id_for_unresolved_delivery(tmp_path: Path):
    ledger = E35Ledger(tmp_path / "ledger", work_id="R0:s:cascade", attempt_ordinal=0, intent_limit=8)
    with pytest.raises(DeliveryUnresolved) as caught:
        ledger.call(kind="generation", key="direct:generation:1", payload={"public": True},
                    invoke=lambda: (_ for _ in ()).throw(RuntimeError("transport")))
    assert caught.value.request_id
    with pytest.raises(DeliveryUnresolved) as resumed:
        E35Ledger(tmp_path / "ledger", work_id="R0:s:cascade", attempt_ordinal=0, intent_limit=8)
    assert resumed.value.request_id == caught.value.request_id


def test_live_controller_records_tool_contract_reject_without_recovery(tmp_path: Path):
    row = _bound_row(tmp_path)
    invalid_tool = {"id": "direct", "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]}
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"}, row=row, output_root=tmp_path, teacher=lambda _: invalid_tool, private_audit=lambda *_: {"decision": "accept"}, rag_search=lambda *_: {}, intent_limit=20)
    assert not outcome["winner"] and outcome["quality_status"] == "route_contract_reject"


def test_live_controller_classifies_missing_witness_rag_evidence_as_quality_not_delivery(tmp_path: Path):
    row = _bound_row(tmp_path)
    row["private"] = {"truth_code": "C000", "audit_protocol": {"rag_witness": True}}
    responses = iter([
        {"id": "direct", "choices": [{"message": {"content": "candidate 0"}}]},
        {"id": "predict", "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]},
        {"id": "classifier", "choices": [{"message": {"content": "candidate 0"}}]},
        {"id": "rag", "choices": [{"message": {"content": "candidate 0"}}]},
    ])
    audits = iter([{"decision": "reject"}, {"decision": "reject"}, {"decision": "accept"}])
    def audit(_, trajectory, __):
        return next(audits)
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"}, row=row, output_root=tmp_path, teacher=lambda _: next(responses), private_audit=audit, rag_search=lambda *_: {}, intent_limit=20)
    assert outcome["delivery_status"] == "delivered" and outcome["quality_status"] == "route_contract_reject"
    assert outcome["contract_error"] == "missing_rag_call_response"


def test_live_controller_rejects_bad_tool_arguments_and_expand_order(tmp_path: Path):
    row = _bound_row(tmp_path)
    invalid_expand = {"id": "classifier", "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "agrinet_classifier_expand", "arguments": '{"reason": "more"}'}}]}}]}
    outcome = run_work_item(work_item={"work_id": "R0:s:cascade", "sample_id": row["sample_id"], "round": "R0"}, row=row, output_root=tmp_path / "expand", teacher=lambda _: invalid_expand, private_audit=lambda *_: {"decision": "accept"}, rag_search=lambda *_: {}, intent_limit=20)
    assert outcome["quality_status"] == "route_contract_reject"


def test_recovery_is_new_lineage_and_r2_shortfalls():
    assert recovery_attempt(prior_attempt_ordinal=0, status="truncated", predecessor_request_id="r0") == {
        "state": "retry", "new_attempt": True, "attempt_ordinal": 1, "predecessor_request_id": "r0"}
    assert recovery_attempt(prior_attempt_ordinal=2, status="truncated", predecessor_request_id="r2") == {
        "state": "delivery_shortfall", "new_attempt": False}
    with pytest.raises(ValueError):
        recovery_attempt(prior_attempt_ordinal=0, status="quality_reject", predecessor_request_id="r0")
    summary = {"round": "R0", "campaign_id": "c", "rows": [
        {"sample_id": "closed", "image_group_id": "g0", "route_progression": ["direct"], "attempt_ordinal": 0, "delivery_status": "delivered", "request_id": "done"},
        {"sample_id": "retry", "image_group_id": "g1", "route_progression": ["direct"], "attempt_ordinal": 0, "delivery_status": "unknown_delivery", "request_id": "unresolved-id"},
    ]}
    assert [item["sample_id"] for item in replenishment_manifest(summary=summary, next_round="R1")["work_items"]] == ["retry"]


def test_rag_refusal_requires_real_rag_and_private_evidence():
    trajectory = {"route": "rag", "answer": "INSUFFICIENT_EVIDENCE", "tool_calls": []}
    with pytest.raises(ValueError, match="RAG refusal"):
        validate_rag_terminal(trajectory)
    trajectory["tool_calls"] = [{"name": "agrinet_rag_search", "arguments": {}}]
    trajectory["private_rag_evidence_insufficient"] = True
    with pytest.raises(ValueError, match="retained actual RAG response"):
        validate_rag_terminal(trajectory)
    trajectory["tool_trace"] = [{"call": {"name": "agrinet_rag_search", "arguments": {}},
                                "response": {"tool": "agrinet_rag_search", "raw_response": {"evidence": []}}}]
    validate_rag_terminal(trajectory)


def test_round_manifest_and_recovery_preserve_new_lineage(tmp_path):
    rows = _rows()
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    manifest = initial_manifest(campaign_id="audit", source_path=source, rows=rows[:32],
                                output=tmp_path / "r0.json", audit_only=True)
    assert manifest["source_rows_expected"] == 32
    outcomes = [{"work_id": item["work_id"], "delivery_status": "unknown_delivery",
                 "request_id": f"request-{index}", "winner": False}
                for index, item in enumerate(manifest["work_items"])]
    summary = freeze_round_summary(manifest=manifest, outcomes=outcomes)
    r1 = replenishment_manifest(summary=summary, next_round="R1")
    assert len(r1["work_items"]) == 32
    assert all(item["attempt_ordinal"] == 1 for item in r1["work_items"])
    assert all(item["predecessor_request_id"].startswith("request-") for item in r1["work_items"])


def test_cascade_outcome_never_bypasses_private_rejection():
    work = {"work_id": "w"}
    winner = cascade_outcome(work_item=work, stages=[
        {"route": "direct", "delivery_status": "delivered", "private_audit": "reject", "request_id": "d"},
        {"route": "classifier", "delivery_status": "delivered", "private_audit": "accept", "request_id": "c"},
    ])
    assert winner["winner"] and winner["final_route"] == "classifier"
    pending = cascade_outcome(work_item=work, stages=[{"route": "direct", "delivery_status": "truncated", "request_id": "d"}])
    assert not pending["winner"] and pending["delivery_status"] == "truncated"


def test_hermes_candidate_keeps_rag_response_and_rejects_private_leakage():
    row = _rows()[0]
    row["system_prompt"] = "one fixed English SFT prompt"
    messages = [
        {"role": "system", "content": "one English prompt"},
        {"role": "user", "content": "diagnose image"},
        {"role": "assistant", "content": "<think>Need evidence.</think>"},
        {"role": "tool_call", "content": {"name": "agrinet_rag_search", "arguments": {}}},
        {"role": "tool", "content": {"tool": "agrinet_rag_search", "evidence": []}},
        {"role": "assistant", "content": "<think>No support.</think><answer>INSUFFICIENT_EVIDENCE</answer>"},
    ]
    trajectory = {"route": "rag", "tool_calls": [{"name": "agrinet_rag_search"}], "tool_trace": [{"call": {"name": "agrinet_rag_search"}, "response": {"tool": "agrinet_rag_search", "raw_response": {"evidence": []}}}], "private_rag_evidence_insufficient": True}
    candidate = hermes_candidate(row=row, trajectory=trajectory, rewrite={"reasoning": "No support.", "final": "INSUFFICIENT_EVIDENCE", "messages": messages}, rewrite_audit="accept")
    assert candidate["route"] == "rag"
    assert candidate["messages"][0] == {"role": "system", "content": "one fixed English SFT prompt"}
    messages[1]["content"] = "truth_code leak"
    with pytest.raises(ValueError, match="leaks"):
        hermes_candidate(row=row, trajectory=trajectory, rewrite={"reasoning": "No support.", "final": "INSUFFICIENT_EVIDENCE", "messages": messages}, rewrite_audit="accept")


def test_export_requires_unique_private_rewrite_audited_winners(tmp_path: Path):
    row = _rows()[0]
    messages = [{"role": "system", "content": "one English prompt"}, {"role": "user", "content": "Identify"}, {"role": "assistant", "content": "<think>Visible evidence supports it.</think><answer>candidate 0</answer>"}]
    accepted = [{"sample_id": row["sample_id"], "winner": True, "request_id": "parent-1", "rewrite_request_id": "rewrite-1", "trajectory": {"route": "direct", "tool_calls": [], "answer": "candidate 0"}, "rewrite": {"reasoning": "Visible evidence supports it.", "final": "candidate 0", "messages": messages}, "rewrite_audit": "accept"}]
    result = export_candidates(source_rows=[row], accepted=accepted, output_dir=tmp_path / "export")
    assert result["rows"] == 1
    assert json.loads((tmp_path / "export" / "private-leakage-audit.json").read_text())["rows"][0]["status"] == "accept"
    with pytest.raises(ValueError, match="immutable"):
        export_candidates(source_rows=[row], accepted=accepted, output_dir=tmp_path / "export")


def test_rewrite_winner_uses_durable_private_rewrite_and_audit(tmp_path: Path):
    row = _rows()[0]
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"route": "direct", "answer": "candidate 0", "tool_calls": []}))
    outcome = {"work_id": "R0:s:cascade", "winner": True, "delivery_status": "delivered", "request_id": "parent-id", "parent_path": str(parent)}
    messages = [{"role": "system", "content": "one English prompt"}, {"role": "user", "content": "Identify"}, {"role": "assistant", "content": "<think>Visible evidence supports it.</think><answer>candidate 0</answer>"}]
    responses = iter([{"choices": [{"message": {"content": json.dumps({"reasoning": "Visible evidence supports it.", "final": "candidate 0", "messages": messages})}}]}, {"choices": [{"message": {"content": json.dumps({"decision": "accept"})}}]}])
    result = rewrite_winner(row=row, outcome=outcome, campaign_root=tmp_path / "campaign", call=lambda _: next(responses), model="test", intent_limit=20)
    assert result["winner"] and result["rewrite_audit"] == "accept"
    assert (tmp_path / "campaign" / "rewrite" / row["sample_id"] / "rewrite.json").is_file()


def test_campaign_report_separates_arm_route_rag_and_shortfall_dimensions():
    report = campaign_report([
        {"arm": "known", "canonical_class_code": "C001", "winner": True, "delivery_status": "delivered", "final_route": "rag", "quality_status": "accepted", "rag_result": "correct_answer"},
        {"arm": "known", "canonical_class_code": "C001", "winner": False, "delivery_status": "truncated", "final_route": "classifier"},
        {"arm": "simulated_unknown", "canonical_class_code": "C002", "winner": False, "delivery_status": "delivered", "final_route": "rag", "quality_status": "accepted", "rag_result": "compliant_refusal"},
        {"arm": "simulated_unknown", "canonical_class_code": "C002", "winner": False, "delivery_status": "delivered", "final_route": "direct", "quality_status": "route_contract_reject"},
    ])
    assert report["arms"]["known"]["rag_terminals"] == {"correct_answer": 1}
    assert report["arms"]["known"]["shortfalls"] == {"delivery_shortfall": 1}
    assert report["arms"]["simulated_unknown"]["rag_terminals"] == {"compliant_refusal": 1}
    assert report["arms"]["simulated_unknown"]["shortfalls"] == {"route_contract_reject": 1}
    source = _rows()
    coverage = campaign_report([], source_rows=source)
    assert len(coverage["arms"]["known"]["per_class"]) == 107
    assert coverage["arms"]["known"]["per_class"]["C000"] == {"target": 5, "closed": 0, "shortfall": 5}


def test_audit_final_report_does_not_expand_after_r2_delivery_shortfall():
    rows = select_audit(_rows())
    summaries = [{"round": round_name, "rows": [
        {"sample_id": row["sample_id"], "delivery_status": "unknown_delivery",
         "winner": False, "final_route": "direct"} for row in rows]}
        for round_name in ("R0", "R1", "R2")]
    report = audit_final_report(source_rows=rows, summaries=summaries)
    assert report["protocol_gate_passed"] is False
    assert report["full_campaign_expansion"] == "not_authorized"
    assert report["per_arm"]["known"]["delivery_shortfall"] == 16


def test_audit_final_report_overlays_sparse_recovery_summaries():
    rows = select_audit(_rows())
    r0_rows = [{"sample_id": row["sample_id"], "delivery_status": "delivered", "winner": True, "final_route": "direct"} for row in rows]
    retry = rows[0]["sample_id"]
    r0_rows[0] = {"sample_id": retry, "delivery_status": "unknown_delivery", "winner": False, "final_route": "direct"}
    report = audit_final_report(source_rows=rows, summaries=[
        {"round": "R0", "rows": r0_rows},
        {"round": "R1", "rows": [{"sample_id": retry, "delivery_status": "delivered", "winner": True, "final_route": "rag"}]},
        {"round": "R2", "rows": []},
    ])
    assert report["per_arm"][rows[0]["arm"]]["closed"] == 16
    assert report["rag_evidence_closed"] is False


def test_full_campaign_manifest_requires_evidence_bearing_audit_gate(tmp_path: Path):
    rows = _rows()
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="not authorized"):
        full_campaign_manifest_after_audit(audit_report={"schema_version": "agrinet.e35-audit-final-report/v1", "protocol_gate_passed": False, "rag_evidence_closed": False}, campaign_id="full", source_path=source, rows=rows, output=tmp_path / "full.json")
    manifest = full_campaign_manifest_after_audit(audit_report={"schema_version": "agrinet.e35-audit-final-report/v1", "protocol_gate_passed": True, "rag_evidence_closed": True}, campaign_id="full", source_path=source, rows=rows, output=tmp_path / "full.json")
    assert manifest["source_rows_expected"] == 1070 and manifest["audit_only"] is False


def test_delivery_reauthorization_creates_new_r0_for_only_exhausted_delivery_gaps(tmp_path: Path):
    rows = _rows()
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    exhausted = [{"sample_id": row["sample_id"], "delivery_status": "unknown_delivery",
                  "request_id": f"r2-{index}"} for index, row in enumerate(rows[:25])]
    prior = {"campaign_id": "v5", "round": "R2", "rows": exhausted}
    manifest = delivery_reauthorization_manifest(source_rows=rows, prior_summary=prior, campaign_id="v6",
        source_output=tmp_path / "v6-source.jsonl", manifest_output=tmp_path / "v6-r0.json")
    assert manifest["delivery_reauthorization"] is True and manifest["source_rows_expected"] == 25
    assert all(item["round"] == "R0" and item["attempt_ordinal"] == 0 for item in manifest["work_items"])
    assert {item["predecessor_request_id"] for item in manifest["work_items"]} == {f"r2-{index}" for index in range(25)}
    assert manifest["source_sha256"] == __import__("hashlib").file_digest((tmp_path / "v6-source.jsonl").open("rb"), "sha256").hexdigest()


def test_delivery_reauthorization_final_replaces_only_base_delivery_gaps():
    rows = select_audit(_rows())
    gap_ids = {row["sample_id"] for row in rows[:3]}
    base_rows = [{"sample_id": row["sample_id"], "delivery_status": "unknown_delivery" if row["sample_id"] in gap_ids else "delivered",
                  "winner": row["sample_id"] not in gap_ids, "final_route": "direct", "parent_path": None} for row in rows]
    report = delivery_reauthorization_final_report(source_rows=rows, base_r0={"campaign_id": "v5", "round": "R0", "rows": base_rows},
        reauthorization_summaries=[{"campaign_id": "v6", "round": "R0", "rows": [{"sample_id": item, "delivery_status": "unknown_delivery", "winner": False, "final_route": "direct", "parent_path": None} for item in gap_ids]},
                                      {"campaign_id": "v6", "round": "R1", "rows": [{"sample_id": next(iter(gap_ids)), "delivery_status": "delivered", "winner": True, "final_route": "direct", "parent_path": None}]},
                                      {"campaign_id": "v6", "round": "R2", "rows": []}])
    assert report["delivery_reauthorization"] is True
    assert sum(values.get("closed", 0) for values in report["per_arm"].values()) == 30
    assert sum(values.get("delivery_shortfall", 0) for values in report["per_arm"].values()) == 2


def test_delivery_reauthorization_final_layers_fresh_lineages_without_replaying_closures():
    rows = select_audit(_rows())
    gap_ids = [row["sample_id"] for row in rows[:3]]
    base_rows = [{"sample_id": row["sample_id"], "delivery_status": "unknown_delivery" if row["sample_id"] in gap_ids else "delivered",
                  "winner": row["sample_id"] not in gap_ids, "final_route": "direct", "parent_path": None} for row in rows]
    def summary(campaign, round_name, selected):
        return {"campaign_id": campaign, "round": round_name, "rows": selected}
    pending = lambda ids: [{"sample_id": item, "delivery_status": "unknown_delivery", "winner": False, "final_route": "direct", "parent_path": None} for item in ids]
    closed = {"sample_id": gap_ids[0], "delivery_status": "delivered", "winner": True, "final_route": "direct", "parent_path": None}
    v6 = [summary("v6", "R0", pending(gap_ids)), summary("v6", "R1", [closed]), summary("v6", "R2", [])]
    v7 = [summary("v7", "R0", pending(gap_ids[1:])), summary("v7", "R1", pending(gap_ids[1:])), summary("v7", "R2", [])]
    report = delivery_reauthorization_final_report(source_rows=rows, base_r0={"campaign_id": "v5", "round": "R0", "rows": base_rows},
        reauthorization_summaries=v7, prior_reauthorization_groups=[v6])
    assert report["prior_reauthorization_campaign_ids"] == ["v6"]
    assert sum(values.get("closed", 0) for values in report["per_arm"].values()) == 30
    assert sum(values.get("delivery_shortfall", 0) for values in report["per_arm"].values()) == 2


def test_scoring_scope_selection_is_globally_three_key_isolated():
    base = _rows()
    known = []
    holds = {0: [], 1: [], 2: []}
    for row in base:
        if row["arm"] == "known":
            known.append({key: row[key] for key in ("canonical_class_code", "image_sha256", "source_group_id", "near_duplicate_group_id")})
            # Make the E3 pool independently identifiable while preserving class coverage.
            holds[len(holds[0]) % 3].append({"canonical_class_code": row["canonical_class_code"],
                                               "image_sha256": "e3-" + row["image_sha256"],
                                               "image_path": "/e3/" + row["sample_id"],
                                               "phash": "e3-" + row["near_duplicate_group_id"]})
    metadata = {row["image_sha256"]: row for fold_rows in holds.values() for row in fold_rows}
    scopes = prepare_scoring_scopes(oof_rows=known, e3_holdouts=holds, image_metadata=metadata)
    assert {arm: len(rows) for arm, rows in scopes.items()} == {"known": 535, "simulated_unknown": 535}
    assert {row["image_sha256"] for row in scopes["known"]}.isdisjoint({row["image_sha256"] for row in scopes["simulated_unknown"]})
