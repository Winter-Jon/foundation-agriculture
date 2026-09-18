"""Formal v3 scoring normalization with approved, unambiguous class aliases."""
import copy
import importlib.util
from functools import lru_cache
from pathlib import Path
from agrinet.research.open_agri_v2_canonical.registry import load_registry

POLICY = 'formal-v3-canonical-answer/v1'
ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def normalizer():
    spec = importlib.util.spec_from_file_location('agrinet_formal_answer_normalizer', ROOT/'vlm/eval/tools/normalize_answers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def registry():
    path = ROOT/'datasets/AgriNet-1K/open_agri_v3/taxonomy'
    return load_registry(path/'canonical_label_registry.jsonl', path/'approval.json', require_approval=True)


def resolve(value):
    text = normalizer().normalize_text(value)
    reg = registry()
    code = reg.resolve_answer(text)
    return {'policy': POLICY, 'raw_answer': value, 'normalized_text': text,
            'canonical_code': code, 'canonical_answer': reg.display_name(code, 'en') if code else None,
            'registry_sha256': reg.digest}


def answer_key(value):
    result = resolve(value)
    return ('code', result['canonical_code']) if result['canonical_code'] else ('unresolved', result['normalized_text'])


def class_correct(answer, truth):
    predicted, expected = resolve(answer), resolve(truth)
    return bool(predicted['canonical_code'] and predicted['canonical_code'] == expected['canonical_code'])


def normalized_review(source, trajectory):
    """Reassess class correctness deterministically; keep independent quality."""
    from agrinet.rag.e320_private import parse_e320_private_audit
    result = copy.deepcopy(trajectory)
    answer = result.get('validation', {}).get('answer', '')
    if not answer or result.get('validation', {}).get('insufficient_evidence'):
        return result
    result['answer_normalization'] = resolve(answer)
    audit = result.get('private_audit', {})
    if audit.get('raw'):
        review = parse_e320_private_audit(audit['raw'])
        review['semantic'] = 'correct' if class_correct(answer, source['private']['truth_name']) else 'incorrect'
        result['private_audit'] = {**audit, **review}
        result['training_eligible'] = review['semantic'] == 'correct' and review.get('quality') == 'pass'
    return result
