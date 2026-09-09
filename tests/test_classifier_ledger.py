import json
from pathlib import Path

import pytest
import yaml

from agrinet.rag.classifier_ledger import BudgetExhausted, DeliveryUnresolved, RequestLedger


@pytest.fixture
def contract():
    return yaml.safe_load(Path("configs/sampling/hcv-classifier-distill-contract-v1.yaml").read_text())


def ledger(tmp_path, contract, **kwargs):
    return RequestLedger(tmp_path, contract=contract, image_group_id="image-1",
                         view="without_candidates", **kwargs)


def test_intent_precedes_transport_and_completed_request_is_cached(tmp_path, contract):
    store = ledger(tmp_path, contract)
    calls = []

    def invoke(payload):
        events = [json.loads(line) for line in store.path.read_text().splitlines()]
        assert events[-1]["event"] == "intent"
        calls.append(payload)
        return {"id": "provider-id", "usage": {"total_tokens": 17},
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}

    first = store.call("generation", "turn-1", {"messages": []}, invoke)
    resumed = ledger(tmp_path, contract)
    assert resumed.call("generation", "turn-1", {"messages": []}, invoke) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="different payload"):
        resumed.call("generation", "turn-1", {"messages": ["changed"]}, invoke)


def test_timeout_is_retained_without_secret_and_never_replayed(tmp_path, contract):
    store = ledger(tmp_path, contract)
    calls = []

    def invoke(payload):
        calls.append(payload)
        raise TimeoutError("secret-token-must-not-leak")

    with pytest.raises(DeliveryUnresolved):
        store.call("generation", "turn-1", {}, invoke)
    assert "secret-token" not in store.path.read_text()
    assert "unknown_delivery" in store.path.read_text()
    resumed = ledger(tmp_path, contract)
    for key in ("turn-1", "turn-2"):
        with pytest.raises(DeliveryUnresolved):
            resumed.call("generation", key, {}, invoke)
    assert len(calls) == 1


def test_process_interruption_leaves_unresolved_intent(tmp_path, contract):
    store = ledger(tmp_path, contract)

    def interrupted(payload):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.call("rag", "tool-1", {}, interrupted)
    with pytest.raises(DeliveryUnresolved):
        ledger(tmp_path, contract).call("rag", "tool-1", {}, lambda _: pytest.fail("replayed"))


def test_truncated_response_is_preserved_and_blocks_continuation(tmp_path, contract):
    store = ledger(tmp_path, contract)
    raw = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
    with pytest.raises(DeliveryUnresolved):
        store.call("generation", "turn-1", {}, lambda _: raw)
    assert "partial" in store.path.read_text()
    with pytest.raises(DeliveryUnresolved):
        store.call("generation", "turn-2", {}, lambda _: pytest.fail("continued"))


def test_budget_counts_attempts_and_private_audit_has_separate_channel(tmp_path, contract):
    store = ledger(tmp_path / "public", contract)
    for i in range(7):
        store.call("generation", str(i), {}, lambda _: {"usage": {"total_tokens": 1}})
    with pytest.raises(BudgetExhausted):
        store.call("generation", "8", {}, lambda _: pytest.fail("over budget"))
    with pytest.raises(ValueError):
        store.call("audit", "audit-1", {}, lambda _: {})
    private = ledger(tmp_path / "private", contract, channel="private")
    private.call("audit", "audit-1", {"truth": "private"}, lambda _: {"decision": "reject"})
    with pytest.raises(BudgetExhausted):
        private.call("audit", "audit-2", {}, lambda _: {})
    assert "truth" not in store.path.read_text()


def test_different_view_cannot_reuse_directory(tmp_path, contract):
    ledger(tmp_path, contract)
    with pytest.raises(DeliveryUnresolved):
        RequestLedger(tmp_path, contract=contract, image_group_id="image-1", view="with_candidates")


def test_partial_event_is_unknown_delivery(tmp_path, contract):
    store = ledger(tmp_path, contract)
    with store.path.open("a") as stream:
        stream.write('{"event":')
    with pytest.raises(DeliveryUnresolved):
        ledger(tmp_path, contract)


def test_concurrent_same_request_invokes_transport_once(tmp_path, contract):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    stores = [ledger(tmp_path, contract), ledger(tmp_path, contract)]
    barrier = Barrier(2)
    calls = []

    def run(store):
        barrier.wait(timeout=5)

        def invoke(payload):
            calls.append(payload)
            return {"result": "delivered"}

        return store.call("generation", "same-turn", {}, invoke)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, stores))
    assert results == [{"result": "delivered"}] * 2
    assert len(calls) == 1


@pytest.mark.parametrize("response", [{"error": {"message": "provider failure"}}, {"choices": None}])
def test_invalid_raw_response_is_retained(tmp_path, contract, response):
    store = ledger(tmp_path, contract)
    with pytest.raises(DeliveryUnresolved):
        store.call("generation", "turn-1", {}, lambda _: response)
    result = json.loads(store.path.read_text().splitlines()[-1])
    assert result["response"] == response
    assert result["status"] == "invalid_response"


def test_duplicate_intent_is_not_silently_deduplicated(tmp_path, contract):
    store = ledger(tmp_path, contract)
    store.call("generation", "turn-1", {}, lambda _: {})
    lines = store.path.read_text().splitlines()
    with store.path.open("a") as stream:
        stream.write(lines[1] + "\n")
    with pytest.raises(DeliveryUnresolved):
        ledger(tmp_path, contract)
