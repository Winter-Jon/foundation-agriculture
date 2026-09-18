import json
from pathlib import Path
from agrinet.rag.e344_report import report


def test_report_counts_private_audit_and_unknown_delivery(tmp_path):
    pilot=tmp_path/'pilot'; sample=pilot/'samples'/'s'/'ledger'/'turn'/'R0'
    sample.mkdir(parents=True)
    source={'sample_id':'s','canonical_class_code':'a','arm':'known','question_type':'open','private':{'truth_name':'apple'}}
    (tmp_path/'pilot-source.jsonl').write_text(json.dumps(source)+'\n')
    (pilot/'lineage.jsonl').write_text('')
    (tmp_path/'deficits.json').write_text(json.dumps([{'canonical_class_code':'a','arm':'known','question_type':'open','target':6}]))
    (tmp_path/'isolation-audit.json').write_text('{}')
    (pilot/'report.json').write_text(json.dumps({'cells':{'known:open:disease':{'attempted':1,'qualified':0,'errors':{'unknown_delivery':1}}},'rag_calls':{},'retrieval_modes':{},'repeated_searches':0}))
    events=[{'event':'binding'}, {'event':'intent','key':'audit','request_id':'1','kind':'private_audit','time':'start'},
            {'event':'result','key':'audit','status':'delivered','time':'end','response':{'raw':{'usage':{'prompt_tokens':100,'completion_tokens':10}}}},
            {'event':'intent','key':'gen','request_id':'2','kind':'generation','time':'start'},
            {'event':'result','key':'gen','status':'unknown_delivery','time':'end'}]
    (sample/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result=report(tmp_path)
    assert result['provider_requests']==2
    assert result['usage']['prompt_tokens_observed']==100
    assert result['usage']['requests_with_unknown_usage']==1
    assert 'cost' not in result
    assert result['remaining_qualified_quota']==6
