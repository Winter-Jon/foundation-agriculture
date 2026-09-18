"""Verify exact multimodal transport and query-only audit boundaries offline."""
import argparse,json,copy,hashlib
from pathlib import Path
from agrinet.rag.dual_teacher import export_multimodal,audit_payload
from agrinet.rag.answer_normalization import resolve
from agrinet.rag.e344_prepare import rows
from agrinet.rag.e344_pipeline import write


def verify(root):
    root=Path(root);checks=[]
    for row,trace in zip(rows(root/'data.jsonl'),rows(root/'lineage.jsonl'),strict=True):
        source=trace['source'];p=Path(trace['original_trajectory_path']);t=json.loads(p.read_text())
        actual=copy.deepcopy(t['messages']);text=actual[-1]['content'];start=text.rindex('<answer>')+8;end=text.rindex('</answer>')
        actual[-1]['content']=text[:start]+resolve(text[start:end])['canonical_answer']+text[end:]
        expected,images=export_multimodal(actual,root/'images'/source['sample_id'])
        assert row['messages']==expected and row['images']==images
        for ref in t['reference_records']:
            assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
        assert len(t['reference_records'])<=6
        payload=audit_payload(source,t['messages']);user=payload['messages'][1]['content']
        assert sum(p['type']=='image_url' for p in user)==1
        public=json.loads(user[0]['text']);assert 'private' not in public
        delivered=[];audit_intents=[]
        for ledger in p.parent.glob('**/events.jsonl'):
            for e in rows(ledger):
                if e.get('event')=='result' and e.get('status')=='delivered':delivered.append(e['response'])
                if e.get('event')=='intent' and e.get('kind')=='private_audit':audit_intents.append(e)
        # A transport-uncertain audit may be legitimately recovered in R1.
        # Every recovery must preserve the same public audit payload; one or
        # more receipt records are therefore evidence of recovery, not a
        # second logical audit.
        assert audit_intents
        if t.get("private_audit", {}).get("audit_policy") == "query-only-conditional-audit/v2":
            from agrinet.rag.dual_teacher_reaudit import revised_payload
            payload = revised_payload(source,t["messages"])
        assert all(intent["payload"] == payload for intent in audit_intents)
        for request in t['requests']:
            if request.get('kind')!='private_audit':assert request['raw'] in delivered
        checks.append({'sample_id':source['sample_id'],'images':len(images),'actual_messages_and_images_preserved':True,'query_only_audit_payload_reconstructed':True,
                       'private_audit_recovery_attempts':len(audit_intents)})
    result={'passed':bool(checks),'exported':len(checks),'rows':checks}
    write(root/'integrity-check.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);verify(p.parse_args().root)
