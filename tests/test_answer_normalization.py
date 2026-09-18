from agrinet.rag.answer_normalization import answer_key, normalized_review


def test_case_and_boundary_whitespace_only():
    assert answer_key('  Gryllus bimaculatus ') == answer_key('gryllus bimaculatus')
    assert answer_key('grape black rot') != answer_key('grape brown spot')
    assert answer_key('grape-black-rot') == answer_key('grape black rot')


def test_reassessment_preserves_original_and_quality_gate():
    import copy
    source={'private':{'truth_name':'slug'}}
    for quality in ['pass','fail']:
        t={'validation':{'answer':'Slug'},'private_audit':{'semantic':'incorrect','quality':quality,
           'raw':{'choices':[{'message':{'content':'{"semantic":"correct","quality":"'+quality+'"}'}}]}},'training_eligible':False}
        original=copy.deepcopy(t)
        result=normalized_review(source,t)
        assert result['training_eligible']==(quality=='pass')
        assert t==original

def test_formal_normalization_and_ambiguous_answers():
    from agrinet.rag.answer_normalization import resolve, class_correct
    for value in ['Ｇｒｙｌｌｕｓ　ｂｉｍａｃｕｌａｔｕｓ', 'Final answer: Gryllus_bimaculatus.', 'Gryllus   bimaculatus']:
        assert resolve(value)['canonical_code'] == 'N05026'
        assert resolve(value)['canonical_answer'] == 'gryllus bimaculatus'
    assert not class_correct('grape black rot or grape brown spot', 'grape black rot')
    assert not class_correct('unknown name', 'unknown name')
    assert not class_correct('grape black rot', 'grape brown spot disease')
