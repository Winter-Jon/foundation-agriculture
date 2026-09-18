"""Offline correction of E344 qualification; original evidence remains immutable."""
import argparse
import json
from collections import Counter
from pathlib import Path
from agrinet.rag.e344_prepare import rows, digest
from agrinet.rag.e344_pipeline import write, write_rows
from agrinet.vlm.full_tool import validate_messages

HOLDS = {
    'e344-63325c06903ab34ae359c914': 'leaf_miner_vs_fungal_cause_unresolved',
    'e344-9dd35f04bccb1abbab860d60': 'walnut_vs_pear_discrimination_relies_on_classifier_rank',
}
# Confirmed from actual stored responses, keyed by source-qualified artifact ID.
QUARANTINE = {'Disease_pest_dataset_seg_wiki::N04056': 'fig_rust_label_with_grape_black_rot_content'}


def remediate(root, output):
    root, output = Path(root), Path(output)
    if (output/'report.json').exists():
        raise ValueError('remediation_version_already_exists')
    output.mkdir(parents=True, exist_ok=True)
    source=list(rows(root/'pilot-source.jsonl'))
    prior_data=list(rows(root/'pilot/data.jsonl')); prior_lineage=list(rows(root/'pilot/lineage.jsonl'))
    by_id={r['source']['sample_id']:(d,r) for d,r in zip(prior_data,prior_lineage,strict=True)}
    decisions=[]; retained=[];lineage=[];snapshots={}
    for row in source:
        sid=row['sample_id'];t=json.loads((root/'pilot/samples'/sid/'trajectory.json').read_text())
        reasons=[]
        try: validate_messages(t['messages'])
        except (ValueError,KeyError,TypeError) as exc: reasons.append('protocol:'+str(exc))
        if sid in HOLDS: reasons.append('review_hold:'+HOLDS[sid])
        exposures=[]
        for event in t['tool_trace']:
            for evidence in event['response'].get('evidence',[]):
                aid=evidence['artifact_id']
                if aid in QUARANTINE:
                    exposures.append(aid);snapshots[aid]=evidence
        eligible=sid in by_id and not reasons
        decisions.append({'sample_id':sid,'previously_exported':sid in by_id,'protocol_and_hold_pass':eligible,
                          'reasons':reasons,'quarantined_evidence_exposure':sorted(set(exposures)),
                          'visual_review_status':'not_independently_completed'})
        if eligible:
            data,old=by_id[sid];retained.append(data);lineage.append({**old,'remediation':'e344-offline-review-v2',
                'quality_status':'provisional_pending_independent_visual_review','quarantined_evidence_exposure':sorted(set(exposures))})
    write_rows(output/'provisional-data.jsonl',retained)
    write_rows(output/'lineage.jsonl',lineage)
    write_rows(output/'decisions.jsonl',decisions)
    write(output/'rag-quarantine.json',{'version':'e344-evidence-quarantine/v1','artifacts':QUARANTINE,'snapshots':snapshots})
    report={'version':'e344-offline-review-v2','source_sha256':digest(root/'pilot-source.jsonl'),
            'original_export_sha256':digest(root/'pilot/data.jsonl'),'reviewed':len(decisions),
            'previously_exported':len(by_id),'provisional_retained':len(retained),
            'removed_from_previous_export':[r for r in decisions if r['previously_exported'] and r['reasons']],
            'strict_protocol_failures':sum(any(x.startswith('protocol:') for x in r['reasons']) for r in decisions),
            'training_approved':False,'teacher_requests':0,'original_artifacts_modified':False,
            'remaining':'Independent image/evidence review; source-specific catalogue reconciliation; no semantic rewriting or automatic reinstatement of case-only failures.'}
    write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(remediate(a.root,a.output),ensure_ascii=False))
