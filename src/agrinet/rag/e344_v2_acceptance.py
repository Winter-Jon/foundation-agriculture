"""Evidence-driven v2 pilot acceptance without historical fixed conclusions."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from agrinet.rag.e344_prepare import rows,digest
from agrinet.rag.e344_pipeline import write
from agrinet.rag.e344_integrity import audit
from agrinet.rag.e344_report import report


def finalize(root):
    root=Path(root);p=root/'pilot'
    integrity=audit(root);summary=report(root)
    sources=list(rows(root/'pilot-source.jsonl'))
    assert len(sources)==32
    outcomes=[]
    for source in sources:
        directory=p/'samples'/source['sample_id']
        path=directory/'trajectory.json'
        if not path.exists():path=directory/'outcome.json'
        outcomes.append(json.loads(path.read_text()))
    template=json.loads((p/'template-check.json').read_text())
    assert integrity['passed'] and template['passed']
    assert len(template['rows'])==integrity['exported']
    assert json.loads((root/'pilot-source-audit.json').read_text())['passed']
    stock=defaultdict(lambda:{'candidates':0,'clusters':Counter(),'selected':0,'attempted':0,'qualified':0})
    for r in rows(root/'cluster-inventory.jsonl'):
        stock[r['canonical_class_code']]['candidates']+=1
        stock[r['canonical_class_code']]['clusters'][r['cluster']]+=1
    for r in rows(root/'selected.jsonl'):stock[r['canonical_class_code']]['selected']+=1
    for r,o in zip(sources,outcomes):
        stock[r['canonical_class_code']]['attempted']+=1
        stock[r['canonical_class_code']]['qualified']+=bool(o.get('training_eligible'))
    write(p/'inventory-summary.json',dict(stock))
    summary.update(dispositions=dict(Counter(o['disposition'] for o in outcomes)),
                   qualification_rate=integrity['exported']/32,
                   template_passed=True,integrity_passed=True,
                   workers=16,rag_dataset_id='wiki_source_scoped_v3',
                   template_length_range=[min(r['length'] for r in template['rows']),max(r['length'] for r in template['rows'])],
                   elapsed_seconds_total=sum(o.get('elapsed_seconds',0) for o in outcomes))
    # Protocol-failed trajectories lack final validation; count actual tool
    # responses rather than treating missing validation as zero searches.
    summary["rag_calls"] = dict(Counter(sum(
        event["call"]["function"]["name"] == "agrinet_rag_search"
        and "protocol_error" not in event["response"]
        for event in outcome.get("tool_trace", [])) for outcome in outcomes))
    write(p/'report.json',summary)
    with (p/'REPORT.md').open('a') as f:
        f.write(chr(10)+"Corrected actual RAG call distribution (including protocol failures): "+json.dumps(summary["rag_calls"])+chr(10))
        f.write(chr(10)+'## v2 acceptance'+chr(10)*2)
        f.write(json.dumps({k:summary[k] for k in ['dispositions','qualification_rate','workers','template_length_range','rag_dataset_id']},indent=2)+chr(10))
        f.write('All exported rows passed native receipt/message/image integrity and actual training template checks. Independent private audits are retained per sample. This does not establish downstream SFT benefit or a causal improvement over v1, since pilot images differ. No full collection or SFT launched. Monetary reporting waived.'+chr(10))
    paths=[root/'pilot-source-audit.json',root/'pilot-manifest.json',root/'rag-binding.json',p/'data.jsonl',p/'lineage.jsonl',p/'template-check.json',p/'integrity-audit.json',p/'report.json',p/'inventory-summary.json',p/'deficits.json']
    write(p/'acceptance.json',{'passed':True,'pilot_rows':32,'qualified':integrity['exported'],
          'evidence_sha256':{str(x.relative_to(root)):digest(x) for x in paths},
          'full_collection':False,'sft':False,'monetary_reporting':'waived'})
    return {'qualified':integrity['exported'],'dispositions':summary['dispositions']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(finalize(parser.parse_args().root)))
