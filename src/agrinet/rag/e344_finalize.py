"""Offline final pilot acceptance and inventory report; never calls providers."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from agrinet.rag.e344_prepare import rows, digest
from agrinet.rag.e344_pipeline import write
from agrinet.rag.e344_integrity import audit
from agrinet.rag.e344_report import report


def finalize(root):
    root=Path(root);pilot=root/'pilot'
    integrity=audit(root); summary=report(root)
    template=json.loads((pilot/'template-check.json').read_text())
    if not template['passed'] or len(template['rows'])!=integrity['exported']:
        raise ValueError('actual_template_acceptance_failed')
    source=list(rows(root/'pilot-source.jsonl'))
    outcomes=[json.loads((pilot/'samples'/r['sample_id']/'trajectory.json').read_text()) for r in source]
    if len(source)!=32 or len(outcomes)!=32:
        raise ValueError('pilot_incomplete')
    by_class=defaultdict(lambda:{'candidates':0,'clusters':Counter(),'selected':0,'attempted':0,'qualified':0})
    for r in rows(root/'cluster-inventory.jsonl'):
        item=by_class[r['canonical_class_code']];item['candidates']+=1;item['clusters'][r['cluster']]+=1
    for r in rows(root/'selected.jsonl'):by_class[r['canonical_class_code']]['selected']+=1
    for r,o in zip(source,outcomes):
        item=by_class[r['canonical_class_code']];item['attempted']+=1;item['qualified']+=bool(o.get('training_eligible'))
    write(pilot/'inventory-summary.json',dict(by_class))
    stats=Counter(o['disposition'] for o in outcomes)
    summary.update(dispositions=dict(stats),qualification_rate=integrity['exported']/32,
                   template_passed=True,integrity_passed=True,
                   template_length_range=[min(r['length'] for r in template['rows']),max(r['length'] for r in template['rows'])],
                   elapsed_seconds_total=sum(o['elapsed_seconds'] for o in outcomes),
                   case_only_name_failures_count=len(summary['case_only_name_failures']),
                   interpretation='Pilot completed; low yield and naming/analysis failures warrant a prompt/protocol follow-up before any full collection. No SFT effect measured.')
    write(pilot/'report.json',summary)
    with (pilot/'REPORT.md').open('a') as f:
        f.write('\n## Final acceptance and interpretation\n\n')
        f.write(f"All {integrity['exported']} exported records pass independent receipt/message/image integrity and actual Qwen3-VL/Hermes template validation. Length range: {summary['template_length_range']} tokens; no truncation.\n\n")
        f.write('Outcome categories: '+json.dumps(dict(stats))+'.\n\n')
        f.write(f"There are {len(summary['case_only_name_failures'])} case-only canonical-name failures. These remain excluded under the frozen wire contract and must not be interpreted as all being recognition errors. One observed quality failure reproduced the placeholder analysis text; the private gate rejected it. No semantic corrections were sent to the teacher.\n\n")
        f.write(f"Union candidate truth coverage: {summary['union_truth_covered']}/32. Remaining qualified deficit: {summary['remaining_qualified_quota']}. Naive remaining-attempt projection: {summary['projected_attempts_for_remaining_quota']:.1f}; this ignores per-class and inventory effects and is not a collection recommendation.\n\n")
        f.write('Per-class cluster stock is in inventory-summary.json; per-quota deficits in deficits.json. All source attempts and private reviews remain under samples/. Currency fees were waived by the user and are not reported.\n\n')
        f.write(summary['interpretation']+'\n')
    evidence={str(p.relative_to(root)):digest(p) for p in [root/'pilot-manifest.json',root/'selection-audit.json',root/'selection-replay-audit.json',root/'pilot-source-audit.json',root/'isolation-audit.json',root/'checkpoint-embedded-label-audit.json',pilot/'data.jsonl',pilot/'lineage.jsonl',pilot/'template-check.json',pilot/'integrity-audit.json',pilot/'report.json',pilot/'inventory-summary.json',pilot/'deficits.json']}
    write(pilot/'acceptance.json',{'passed':True,'pilot_rows':32,'qualified':integrity['exported'],'evidence_sha256':evidence,
        'monetary_cost_requirement':'waived_by_user','full_collection':False,'sft':False,
        'limitations':['pHash and source paths cannot prove acquisition-group independence','32 samples do not establish downstream SFT benefit','4 provider attempts have unknown usage']})
    return {'passed':True,'qualified':integrity['exported']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(finalize(a.root)))


if __name__=='__main__':main()
