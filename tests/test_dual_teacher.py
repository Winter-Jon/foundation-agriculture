import json
from PIL import Image
from agrinet.rag.dual_teacher import audit_payload, reference_attachments, parse_audit


def test_auditor_only_sees_query_and_public_reasoning(tmp_path):
    path=tmp_path/'query.jpg';Image.new('RGB',(12,12)).save(path)
    row={'image_path':str(path),'private':{'truth':'SECRET'}}
    payload=audit_payload(row,[{'role':'assistant','content':'observations'},
        {'role':'tool','content':'SECRET_TOOL'}, {'role':'user','content':[{'type':'image_url','image_url':{'url':'SECRET_REF'}}]}])
    encoded=json.dumps(payload)
    assert 'SECRET' not in encoded
    assert 'observations' in encoded
    assert sum(p['type']=='image_url' for p in payload['messages'][1]['content'])==1


def test_reference_dedup_and_query_exclusion(tmp_path):
    import hashlib
    path=tmp_path/'ref.jpg';Image.new('RGB',(12,12)).save(path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    response={'evidence':[{'artifact_id':'a','metadata':{'english_name':'name','local_reference_images':[str(path)]}}]}
    seen={}
    assert reference_attachments(response,tmp_path,seen,sha)==(None,[])
    message,records=reference_attachments(response,tmp_path,seen,'other')
    assert len(records)==1
    message,records=reference_attachments(response,tmp_path,seen,'other')
    assert not records and all(p['type']=='text' for p in message['content'])


def test_multimodal_export_keeps_reference_order(tmp_path):
    from agrinet.rag.dual_teacher import export_multimodal
    from agrinet.rag.e35_transport import transport_image
    path=tmp_path/'query.jpg';Image.new('RGB',(12,12)).save(path)
    image,_=transport_image(path,max_side=512)
    original=[{'role':'user','content':[{'type':'text','text':'query'},image]},
              {'role':'user','content':[{'type':'text','text':'REF1'},image]}]
    messages,images=export_multimodal(original,tmp_path/'out')
    assert len(images)==2
    assert messages[0]['content']=='query'+chr(10)+'<image>'+chr(10)+'REF1'+chr(10)+'<image>'
    assert isinstance(original[0]['content'],list)


def test_observation_then_real_tool_sequence(tmp_path, monkeypatch):
    import hashlib
    from agrinet.rag.e344_collect import collect_sample
    import agrinet.rag.e344_collect as collector
    path=tmp_path/'query.jpg';Image.new('RGB',(12,12)).save(path)
    class Ledger:
        def __init__(self,*a,**kw):pass
        def call(self,*,kind,key,payload,invoke):return key,invoke()
    monkeypatch.setattr(collector,'E35Ledger',Ledger)
    calls=[]
    def teacher(payload):
        calls.append(payload)
        n=len(calls)
        if n==1:
            assert payload['tool_choice']=='none'
            assert 'Observation stage only' in payload['messages'][-1]['content']
            m={'role':'assistant','content':'Visible leaf with brown spots. The underside and roots are not visible.'}
        elif n in (2,3):
            name='agrinet_classifier_predict' if n==2 else 'agrinet_rag_search'
            args={} if n==2 else {'retrieval_type':'name','query':'apple black rot','top_k':3,'rationale':'check lesion evidence'}
            m={'role':'assistant','content':'Compare lesion morphology.','tool_calls':[{'id':str(n),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        else:m={'role':'assistant','content':'<think>Brown lesions support black rot but host certainty is limited.</think><answer>apple black rot</answer>'}
        return {'choices':[{'message':m}]}
    row={'sample_id':'test','image_path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'question_type':'open','private':{'truth_name':'apple black rot'}}
    r=collect_sample(row,root=tmp_path/'sample',teacher=teacher,classifier=lambda r:{'top3':[{'name':'apple black rot','code':'N04001'}]},retrieve=lambda r,a:{'evidence':[]},private_audit=lambda r,t:{'semantic':'correct','quality':'pass'},reference_mode=True)
    assert r['training_eligible'] and len(calls)==4


def test_revised_audit_boundary(tmp_path):
    from agrinet.rag.dual_teacher_reaudit import revised_payload
    path=tmp_path/'query.jpg';Image.new('RGB',(12,12)).save(path)
    p=revised_payload({'image_path':str(path),'private':{'truth':'PRIVATE'}},[
        {'role':'assistant','content':'Compare REF1 and query'},
        {'role':'tool','content':'TOOL_SECRET'}])
    assert 'PRIVATE' not in json.dumps(p) and 'TOOL_SECRET' not in json.dumps(p)
    assert 'MUST NOT by itself cause failure' in p['messages'][0]['content']
    assert sum(x['type']=='image_url' for x in p['messages'][1]['content'])==1
