import pytest
from agrinet.rag.e344_pipeline import write, write_rows
from agrinet.rag.e344_prepare import digest
from agrinet.rag.full_collection_acceptance import accept, CHECKS


def fixture_data(tmp_path):
    root = tmp_path / 'live'
    root.mkdir()
    sources = [dict(sample_id=str(i), canonical_class_code='class', arm='known',
                    question_type='open', image_sha256=str(i), source_group_id=str(i),
                    near_duplicate_group_id=str(i)) for i in range(9)]
    write_rows(tmp_path / 'pilot-source.jsonl', sources)
    write_rows(root / 'data.jsonl', [{'messages': []} for _ in sources])
    write_rows(root / 'lineage.jsonl', [{'source': s} for s in sources])
    checks = [dict(row_index=i, **{k: True for k in CHECKS}) for i in range(9)]
    checks[0] = {'row_index': 0, 'error': 'max_length_exceeded'}
    write(root / 'template-check.json', {'data_sha256': digest(root / 'data.jsonl'), 'rows': checks})
    return root


def test_long_rows_excluded_and_surplus_cannot_fill_other_cells(tmp_path):
    result = accept(fixture_data(tmp_path))
    assert result['template_accepted'] == 8
    assert result['accepted_within_quota'] == 6
    assert result['target'] == 18
    assert result['deficit'] == 12


def test_stale_template_report_rejected(tmp_path):
    root = fixture_data(tmp_path)
    with (root / 'data.jsonl').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='template_input_hash_mismatch'):
        accept(root)


def test_combined_acceptance_fills_only_first_deficits(tmp_path):
    from agrinet.rag.dual_teacher_combined_acceptance import accept as combined_accept
    first_parent = tmp_path / 'first'
    first_parent.mkdir()
    first = fixture_data(first_parent)
    first_data = list(__import__('agrinet.rag.e344_prepare', fromlist=['rows']).rows(first / 'data.jsonl'))
    first_lineage = list(__import__('agrinet.rag.e344_prepare', fromlist=['rows']).rows(first / 'lineage.jsonl'))
    # Preserve two accepted rows and request one more in the same cell.
    write_rows(first / 'quota-data.jsonl', first_data[1:3])
    write_rows(first / 'quota-lineage.jsonl', first_lineage[1:3])
    write(first / 'quota-deficits.json', [
        dict(canonical_class_code='class', arm=arm, question_type=question, target=quota,
             available=2 if (arm, question) == ('known', 'open') else 0,
             deficit=1 if (arm, question) == ('known', 'open') else quota)
        for arm, question, quota in __import__('agrinet.rag.e344_prepare', fromlist=['PRIORITY']).PRIORITY
    ])
    supplement_parent = tmp_path / 'supplement'
    supplement = supplement_parent / 'live'
    supplement.mkdir(parents=True)
    source = dict(sample_id='s', canonical_class_code='class', arm='known', question_type='open',
                  image_sha256='s', source_group_id='s', near_duplicate_group_id='s')
    write_rows(supplement / 'data.jsonl', [{'messages': ['supplement']}])
    write_rows(supplement / 'lineage.jsonl', [{'source': source}])
    write(supplement / 'template-check.json', {'data_sha256': digest(supplement / 'data.jsonl'),
                                                'rows': [dict(row_index=0, **{k: True for k in CHECKS})]})
    out = tmp_path / 'combined'
    result = combined_accept(first, [supplement], out)
    assert result['accepted'] == 3
    assert result['deficit'] == 15
    assert [row['collection_origin'] for row in __import__('agrinet.rag.e344_prepare', fromlist=['rows']).rows(out / 'lineage.jsonl')] == ['first_round', 'first_round', 'supplement_1']
