import pytest
from jsonschema import Draft7Validator
from agrinet.vlm.full_tool import tool_schemas,validate_arguments,rag_request,RAG_SEARCH,SYSTEM_PROMPT
from agrinet.rag.protocol_repair import repair_instruction


@pytest.mark.parametrize('mode',['visual','name','semantic'])
@pytest.mark.parametrize('image',['absent','query_image','/untrusted/path',None])
def test_schema_runtime_image_contract_agree(mode,image):
    args={'retrieval_type':mode,'query':'leaf spots','rationale':'compare symptoms','top_k':3}
    if image!='absent':args['image']=image
    valid=not list(Draft7Validator(tool_schemas()[1]['function']['parameters']).iter_errors(args))
    assert valid==(not validate_arguments(RAG_SEARCH,args))
    if valid:
        body=rag_request(args,'/bound/query.jpg')
        assert (body.get('image_path')=='/bound/query.jpg')==(mode=='visual')


def test_repair_preserves_analysis_without_answer_feedback():
    prompt=repair_instruction(['malformed_final'])
    assert 'Move the complete substantive analysis inside' in prompt
    assert 'do not change your diagnosis' in prompt
    assert 'No answer correction is supplied' in prompt
    assert 'Do not discard' in prompt
    assert '<think>analysis</think>' not in SYSTEM_PROMPT


def test_missing_image_historical_calls_now_bind_query():
    args={'retrieval_type':'visual','query':'moth','rationale':'compare morphology','top_k':3}
    assert rag_request(args,'/query.jpg')['image_path']=='/query.jpg'
