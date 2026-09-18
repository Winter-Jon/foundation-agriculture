import json
from agrinet.rag.repair_wiki_identity import build


def test_second_source_code_never_overrides_semantic_mapping(tmp_path):
    source=tmp_path/'sources';p=source/'second';p.mkdir(parents=True)
    (p/'knowledge_base.json').write_text(json.dumps({'description':{'N1':{'english_name':'grape black rot','chinese_name':'葡萄黑腐病','description':{'content':'Grape disease'}}}}))
    base=tmp_path/'base.json';base.write_text(json.dumps({'description':{'second::N1':{'code':'N1','source_dataset':'second','english_name':'fig rust','reference_images':[]}}}))
    registry=tmp_path/'registry.jsonl';registry.write_text('\n'.join(json.dumps(r) for r in [
        {'canonical_code':'N1','canonical_english_name':'fig rust','canonical_chinese_name':'无花果锈病','aliases':[]},
        {'canonical_code':'N2','canonical_english_name':'grape black rot','canonical_chinese_name':'葡萄黑腐病','aliases':[]}]))
    build(base,source,registry,tmp_path/'out')
    row=json.loads((tmp_path/'out/catalog.json').read_text())['description']['second::N1']
    assert row['source_code']=='N1' and row['code']=='N2'
    assert row['english_name']=='grape black rot'
    assert row['description']=={'content_1':'Grape disease'}
    assert row['similar_english_classes']==[]


def test_quarantine_cannot_be_bypassed_by_version_tag():
    from agrinet.rag.evidence_quality import partition_evidence
    evidence = {'artifact_id': 'Disease_pest_dataset_seg_wiki::N04056',
                'metadata': {'identity_version': 'source-scoped-wiki/v1', 'english_name': 'grape black rot'}}
    retained, excluded = partition_evidence({'evidence': [evidence]})
    assert not retained['evidence'] and len(excluded) == 1
    assert evidence['metadata']['english_name'] == 'grape black rot'


def test_missing_first_source_does_not_exclude_later_evidence():
    from agrinet.rag.repair_wiki_identity import source_contents
    assert source_contents({'content_1': None, 'content_2': 'GBIF morphology',
                            'content_4': 'EOL description', 'status_2': 'ok'}) == [
                                ('content_2', 'GBIF morphology'), ('content_4', 'EOL description')]
