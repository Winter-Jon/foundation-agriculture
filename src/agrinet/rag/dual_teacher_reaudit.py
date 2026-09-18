"""Bounded audit-only revision: no regeneration, no truth in auditor context."""
import argparse,copy,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
from agrinet.rag.dual_teacher import audit_payload,parse_audit
from agrinet.rag.e344_runtime import Runtime
from agrinet.rag.e344_recovery import SampleLedger
from agrinet.rag.e344_prepare import rows,digest
from agrinet.rag.e344_pipeline import write,write_rows
from agrinet.rag.answer_normalization import class_correct

POLICY='query-only-conditional-audit/v2'
GUIDANCE='''You audit logical coherence and diagnostic explanation quality, not label accuracy. Input: query image and the complete ordered reasoning. The sampling teacher DID receive the reference images and source text it cites. They are deliberately omitted from YOUR input to reduce cost. Treat its stated reference observations and source quotations as unverified premises for this audit. Their omission is NOT evidence of a defect and MUST NOT by itself cause failure. Do not claim to verify those premises. Check whether the inference follows from the stated premises, whether query-image observations have an unmistakable contradiction, whether exclusions are valid, and whether uncertainty is proportional. A species-level answer is allowed when the described comparison supports it; do not demand laboratory confirmation or exhaustive comparison with species outside the candidate set. A revision of initial observations is allowed if explained. Fail only for a material defect identifiable from the supplied query or reasoning. An uncertain fine detail is not a definite visual conflict. Do not infer an error from ranks alone unless rank substitutes for diagnostic support. No hidden truth is supplied. Return JSON: verdict pass/fail; issues (max 3 objects, type and reason, cite the relevant statement); reason (short). Types: observation_conflict, reasoning_inconsistency, unsupported_discrimination, invalid_exclusion, overconfidence. For pass issues must be empty. If reference-premise truth cannot be verified, mention this only as a scope limit, not as a failure. Do not supply corrections or alternative answers.'''


def revised_payload(row,messages):
    payload=audit_payload(row,messages)
    payload['messages'][0]['content']=GUIDANCE
    return payload


def run(config):
    root=Path(config['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    sources=list(rows(config['source']));assert len(sources)==8
    runtime=Runtime({},timeout=180)
    def one(row):
        original=Path(config['original_root'])/'samples'/row['sample_id']/'trajectory.json'
        t=json.loads(original.read_text());directory=root/'samples'/row['sample_id'];payload=revised_payload(row,t['messages'])
        ledger=SampleLedger(directory/'ledger',work_id=POLICY+row['sample_id'])
        try:
            rid,raw=ledger.call(kind='private_audit',key='audit',payload=payload,invoke=lambda:runtime.teacher(payload))
            review=parse_audit(raw)
            answer=t.get('validation',{}).get('answer','')
            correct=bool(answer) and class_correct(answer,row['private']['truth_name'])
            result={'sample_id':row['sample_id'],'original_sha256':digest(original),'policy':POLICY,
                    'old_quality':t.get('private_audit',{}).get('quality'),'class_correct':correct,
                    'eligible':correct and review['verdict']=='pass','review':review,'raw':raw,'request_id':rid,
                    'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
        except Exception as exc:
            result={'sample_id':row['sample_id'],'eligible':False,'error_type':type(exc).__name__}
        write(directory/'review.json',result);return result
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,sources))
    write_rows(root/'decisions.jsonl',[{k:v for k,v in r.items() if k!='raw'} for r in results])
    write(root/'report.json',{'policy':POLICY,'attempted':8,'eligible':sum(r['eligible'] for r in results),
        'audit_pass':sum(r.get('review',{}).get('verdict')=='pass' for r in results),'generation_requests':0,
        'original_trajectories_modified':False,'errors':sum('error_type' in r for r in results)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(yaml.safe_load(Path(a.config).read_text())['parameters'])
