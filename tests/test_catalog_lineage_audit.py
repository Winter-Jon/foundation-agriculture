import json
from pathlib import Path
from agrinet.rag.catalog_lineage_audit import audit


def test_identical_codes_in_two_sources_are_not_joined(tmp_path):
    base=tmp_path/'base.json'; source=tmp_path/'sources'; registry=tmp_path/'registry.jsonl'
    for name,label in [('first','fig rust leaf'),('second','grape black rot')]:
        p=source/name;p.mkdir(parents=True)
        (p/'knowledge_base.json').write_text(json.dumps({'description':{'N1':{'english_name':label}}}))
    base.write_text(json.dumps({'description':{f'{s}::N1':{'code':'N1','source_dataset':s,'english_name':'fig rust leaf'} for s in ['first','second']}}))
    registry.write_text(json.dumps({'canonical_code':'C2','canonical_english_name':'grape black rot','aliases':[]})+'\n')
    result=audit(base,source,registry,tmp_path/'out')
    assert result['source_name_matches']==1
    q=json.loads((tmp_path/'out/quarantine.json').read_text())
    assert list(q)==['second::N1']
    assert q['second::N1']['exact_alias_candidate_codes']==['C2']
    assert not result['automatic_relabeling']
