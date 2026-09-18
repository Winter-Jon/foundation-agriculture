import json

import numpy as np
import pytest

from agrinet.vlm.full_tool import CLASSIFIER_PREDICT, RAG_SEARCH, rag_request, validate_arguments, validate_messages
from agrinet.rag.e344_prepare import cluster, select, pilot


def trajectory(modes=("visual",), answer="apple black rot"):
    messages = []
    for i, mode in enumerate(("classifier",) + modes):
        name = CLASSIFIER_PREDICT if i == 0 else RAG_SEARCH
        args = {} if i == 0 else {"retrieval_type": mode, "query": "lesions", "top_k": 3, "rationale": "resolve visible lesion morphology"}
        if mode == "visual":
            args["image"] = "query_image"
        messages.extend([{"role": "assistant", "tool_calls": [{"id": str(i), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                         {"role": "tool", "tool_call_id": str(i), "content": "{}"}])
    messages.append({"role": "assistant", "content": f"<think>Visible lesions; nearest alternative is uncertain.</think><answer>{answer}</answer>"})
    return messages


@pytest.mark.parametrize("mode", ["visual", "name", "semantic"])
def test_modal_binding(mode):
    args = {"retrieval_type": mode, "query": "lesions", "top_k": 3, "rationale": "resolve ambiguity"}
    if mode == "visual":
        args["image"] = "query_image"
    body = rag_request(args, "/bound/image.jpg")
    assert ("image_path" in body) == (mode == "visual")
    args["image"] = "/arbitrary/image.jpg"
    assert validate_arguments(RAG_SEARCH, args)


def test_name_normalization_preserves_unicode_canonical_names():
    from agrinet.rag.milvus import MilvusSiglipBackend
    assert MilvusSiglipBackend._normalize_name('Potato_Bacterial Ring Rot') == 'potato bacterial ring rot'
    assert MilvusSiglipBackend._normalize_name('马铃薯环腐病') == '马铃薯环腐病'


def test_order_limits_response_integrity_and_insufficiency():
    assert validate_messages(trajectory(("visual", "name", "semantic")))["rag_calls"] == 3
    assert validate_messages(trajectory(("visual", "visual")))["repeated_searches"] == 1
    assert validate_messages(trajectory(answer="INSUFFICIENT_EVIDENCE"))["insufficient_evidence"]
    for messages in (trajectory(()), trajectory(("visual",) * 4), trajectory()[2:], trajectory()[:1] + trajectory()[2:]):
        with pytest.raises(ValueError):
            validate_messages(messages)


def test_cluster_determinism_and_group_bound():
    x = np.random.default_rng(7).normal(size=(30, 8))
    a = cluster(x, [str(i % 3) for i in range(30)])
    assert a == cluster(x, [str(i % 3) for i in range(30)])
    assert len(set(a[0])) == 3


def test_priority_and_no_image_reuse():
    pool = [{"canonical_class_code": "x", "image_sha256": str(i), "source_group_id": str(i),
             "near_duplicate_group_id": str(i), "cluster": i % 3, "center_distance": i / 10,
             "bindings": {"known": {}, "simulated_unknown": {}}} for i in range(14)]
    chosen, deficits = select(pool)
    assert [(r["arm"], r["question_type"]) for r in chosen] == [("simulated_unknown", "open")] * 6 + [("known", "open")] * 6 + [("simulated_unknown", "option")] * 2
    assert len({r["image_sha256"] for r in chosen}) == 14
    assert [r["selected"] for r in deficits] == [6, 6, 2, 0]
    assert all(r["qualified"] == 0 for r in deficits)


def test_unknown_delivery_is_not_replayed(tmp_path):
    from agrinet.rag.e35_ledger import E35Ledger, DeliveryUnresolved
    ledger = E35Ledger(tmp_path, work_id="pilot:one", attempt_ordinal=0, intent_limit=9)
    calls = []
    def fail():
        calls.append(1)
        raise TimeoutError("unknown")
    with pytest.raises(DeliveryUnresolved):
        ledger.call(kind="generation", key="turn:0", payload={"public": 1}, invoke=fail)
    with pytest.raises(DeliveryUnresolved):
        ledger.call(kind="generation", key="turn:0", payload={"public": 1}, invoke=fail)
    assert len(calls) == 1


def test_oof_and_holdout_eligibility():
    from agrinet.rag.e344_assets import eligible_binding
    binding = {"training_shas": {"train"}, "training_codes": {"apple"},
               "label_codes": ["apple"], "heldout_shas": {"held"}}
    assert eligible_binding({"image_sha256": "held", "canonical_class_code": "apple"}, binding, "known")
    assert not eligible_binding({"image_sha256": "other", "canonical_class_code": "apple"}, binding, "known")
    assert eligible_binding({"image_sha256": "other", "canonical_class_code": "pear"}, binding, "simulated_unknown")
    assert not eligible_binding({"image_sha256": "held", "canonical_class_code": "apple"}, binding, "simulated_unknown")


def test_native_sample_keeps_evidence_and_private_channel_separate(tmp_path, monkeypatch):
    from agrinet.rag import e344_collect
    monkeypatch.setattr(e344_collect, "transport_image", lambda *a, **k: ({"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}, {}))
    generated = iter([m for m in trajectory() if m["role"] == "assistant"])
    wire = []
    def teacher(payload):
        wire.append(json.dumps(payload))
        return {"choices": [{"message": next(generated)}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    row = {"sample_id": "one", "image_path": "/image", "image_sha256": "sha", "question_type": "open",
           "private": {"truth": "SECRET_PRIVATE_TRUTH"}}
    result = e344_collect.collect_sample(row, root=tmp_path, teacher=teacher,
        classifier=lambda r: {"top3": [{"name": "apple black rot"}]},
        retrieve=lambda r, a: {"evidence": [{"name": "apple black rot", "text": "Original evidence text"}]},
        private_audit=lambda r, t: {"semantic": "correct", "quality": "pass"})
    assert result["training_eligible"]
    assert len(result["requests"]) == 3
    assert all("SECRET_PRIVATE_TRUTH" not in text for text in wire)
    assert "Original evidence text" in result["messages"][-2]["content"]


def test_options_deduplicate_names_and_bind_private_letter():
    from agrinet.rag.e344_export import make_options
    registry = {code: {"canonical_english_name": name} for code, name in
                [("a", "apple"), ("b", "pear"), ("alias", "APPLE"), ("c", "peach"), ("d", "plum")]}
    options, correct = make_options("a", ["alias", "b", "c", "d"], registry, "sha")
    assert len({x["name"].casefold() for x in options}) == 4
    assert options[ord(correct)-65]["name"] == "apple"
    assert (options, correct) == make_options("a", ["alias", "b", "c", "d"], registry, "sha")


def test_export_rejects_tampered_evidence():
    from agrinet.rag.e344_export import export_one
    from agrinet.vlm.full_tool import contract_hashes
    messages = trajectory()
    messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "Identify"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]})
    calls = [m["tool_calls"][0] for m in messages if m.get("tool_calls")]
    value = {"messages": messages, "tool_trace": [{"call": call, "response": {}} for call in calls],
             "training_eligible": True, "private_audit": {"semantic": "correct", "quality": "pass"},
             "contract": contract_hashes()}
    source = {"image_path": "/image", "question_type": "option", "public_options": [{"name": "apple black rot"}], "private": {"truth_name": "apple black rot"}}
    exported, _ = export_one(source, value)
    assert exported["messages"][-1] == messages[-1]
    value["tool_trace"][0]["response"] = {"invented": "evidence"}
    with pytest.raises(ValueError, match="tool_evidence_changed"):
        export_one(source, value)


def test_protocol_repair_preserves_failed_action(tmp_path, monkeypatch):
    from agrinet.rag import e344_collect
    monkeypatch.setattr(e344_collect, "transport_image", lambda *a, **k: ({"type": "image_url", "image_url": {"url": "image"}}, {}))
    bad = {"role": "assistant", "content": "<answer>malformed</answer>"}
    generated = iter([bad] + [m for m in trajectory() if m["role"] == "assistant"])
    row = {"sample_id": "repair", "image_path": "/image", "image_sha256": "sha", "question_type": "open", "private": {"truth_name": "apple black rot"}}
    result = e344_collect.collect_sample(row, root=tmp_path,
        teacher=lambda payload: {"choices": [{"message": next(generated)}]},
        classifier=lambda r: {"top3": [{"name": "apple black rot"}]}, retrieve=lambda r,a: {"evidence": []},
        private_audit=lambda r,t: {"semantic": "correct", "quality": "pass"})
    assert result["training_eligible"]
    assert result["quality_repairs"] == 1
    assert bad in result["messages"]


def test_recovery_inspects_terminal_failure_and_reuses_success(tmp_path):
    from agrinet.rag.e344_recovery import SampleLedger
    ledger = SampleLedger(tmp_path, work_id='sample')
    invoked = []
    def invoke():
        invoked.append(1)
        if len(invoked) == 1:
            raise TimeoutError()
        return {'ok': True}
    first = ledger.call(kind='generation', key='turn:0', payload={'public': 1}, invoke=invoke)
    assert first[1] == {'ok': True}
    assert ledger.call(kind='generation', key='turn:0', payload={'public': 1}, invoke=invoke) == first
    assert len(invoked) == 2
    assert len(list(tmp_path.glob('**/recovery-decision.json'))) == 1


def test_interrupted_intent_without_receipt_cannot_recover(tmp_path):
    import hashlib
    from agrinet.rag.e344_recovery import SampleLedger
    from agrinet.rag.e35_ledger import E35Ledger, DeliveryUnresolved
    key = 'turn:0'
    directory = tmp_path / hashlib.sha256(key.encode()).hexdigest()[:16] / 'R0'
    ledger = E35Ledger(directory, work_id='sample:turn:0:R0', attempt_ordinal=0, intent_limit=1)
    ledger._append({'event': 'intent', 'key': key, 'request_id': 'unfinished', 'payload_sha256': 'unknown'})
    with pytest.raises(DeliveryUnresolved):
        SampleLedger(tmp_path, work_id='sample').call(kind='generation', key=key, payload={}, invoke=lambda: pytest.fail('replayed'))
    assert not list(tmp_path.glob('**/R1'))


def test_replacement_prefers_failed_cluster_then_undercoverage():
    from agrinet.rag.e344_prepare import replacement
    def row(i, cluster):
        return {'canonical_class_code':'x','image_sha256':str(i),'source_group_id':str(i),
                'near_duplicate_group_id':str(i),'cluster':cluster,'center_distance':i/10,
                'bindings':{'known':{}},'arm':'known','question_type':'open'}
    failed = row(1, 1)
    pool = [failed, row(2,0), row(3,1), row(4,2)]
    assert replacement(pool,failed,[failed],[])['image_sha256'] == '3'
    assert replacement(pool,failed,[failed,pool[2]],[pool[1]])['image_sha256'] == '4'
    assert replacement(pool,failed,pool,[]) is None


def test_pilot_exactly_eight_cells_and_diverse_classes():
    pool=[]
    for arm in ('known','simulated_unknown'):
        for question in ('open','option'):
            for domain in ('disease','pest'):
                for i in range(8):
                    pool.append({'arm':arm,'question_type':question,'domain':domain,
                                 'canonical_class_code':domain+str(i),'center_distance':i/10,
                                 'image_sha256':f'{arm}:{question}:{domain}:{i}'})
    selected=pilot(pool)
    from collections import Counter
    assert len(selected)==32
    assert set(Counter((r['arm'],r['question_type'],r['domain']) for r in selected).values())=={4}
    assert len({r['image_sha256'] for r in selected})==32


def test_protocol_bounds_reject_extra_classifier_and_injected_image():
    messages=trajectory()
    duplicated=json.loads(json.dumps(messages[:2]))
    duplicated[0]['tool_calls'][0]['id']='extra'
    duplicated[1]['tool_call_id']='extra'
    with pytest.raises(ValueError):
        validate_messages(messages[:2]+duplicated+messages[2:])
    for mode in ('name','semantic'):
        assert validate_arguments(RAG_SEARCH,{'retrieval_type':mode,'query':'apple','image':'query_image','top_k':3,'rationale':'compare'})
    assert validate_arguments(CLASSIFIER_PREDICT,{'label':'apple'})
    assert validate_arguments(CLASSIFIER_PREDICT,{'image':'query_image'})


@pytest.mark.parametrize('final', [
    '<think>analysis</think>Outside analysis.</think><answer>apple black rot</answer>',
    '<think>analysis</think><answer>apple black rot</answer>',
    '<think> </think><answer>apple black rot</answer>',
    '<think>Actual <think>nested</think> analysis</think><answer>apple black rot</answer>',
    'extra <think>Reasoning.</think><answer>apple black rot</answer>',
])
def test_strict_final_rejects_invalid_analysis(final):
    messages=trajectory();messages[-1]['content']=final
    with pytest.raises(ValueError):validate_messages(messages)


def test_quarantine_is_source_qualified():
    from agrinet.rag.evidence_quality import partition_evidence
    good={'artifact_id':'agri_disease_pest_wiki::N04056','metadata':{'english_name':'fig rust leaf'}}
    bad={'artifact_id':'Disease_pest_dataset_seg_wiki::N04056','metadata':{'english_name':'fig rust leaf'}}
    original={'evidence':[good,bad]}
    filtered,excluded=partition_evidence(original)
    assert filtered['evidence']==[good]
    assert excluded[0]['artifact_id']==bad['artifact_id']
    assert original['evidence']==[good,bad]
