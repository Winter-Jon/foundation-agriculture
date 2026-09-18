"""Independent final native-export evidence audit; no provider calls."""
import base64
import hashlib
import json
from pathlib import Path
from agrinet.rag.e344_prepare import rows, digest
from agrinet.rag.e344_export import export_one
from agrinet.rag.e344_pipeline import write


def audit(root):
    root=Path(root); pilot=root/'pilot'
    data=list(rows(pilot/'data.jsonl')); lineage=list(rows(pilot/'lineage.jsonl'))
    if len(data)!=len(lineage):
        raise ValueError('lineage_count_mismatch')
    results=[]
    for student, provenance in zip(data,lineage):
        source=provenance['source']; directory=pilot/'samples'/source['sample_id']
        trajectory=json.loads((directory/'trajectory.json').read_text())
        expected, _=export_one(source,trajectory)
        expected['images']=student['images']
        if expected!=student:
            raise ValueError('export_messages_changed')
        if hashlib.sha256(json.dumps(student,sort_keys=True).encode()).hexdigest()!=provenance['student_row_sha256']:
            raise ValueError('student_hash_mismatch')
        image=next(p for m in trajectory['messages'] if isinstance(m.get('content'),list)
                   for p in m['content'] if p.get('type')=='image_url')
        actual=base64.b64decode(image['image_url']['url'].split(',',1)[1],validate=True)
        if Path(student['images'][0]).read_bytes()!=actual:
            raise ValueError('teacher_student_image_mismatch')
        delivered=[]
        for ledger in directory.glob('**/events.jsonl'):
            for event in rows(ledger):
                if event.get('event')=='result' and event.get('status')=='delivered':
                    delivered.append(event['response'])
        public_raw=[request['raw'] for request in trajectory['requests'] if request.get('kind')!='private_audit']
        if any(response not in delivered for response in public_raw):
            raise ValueError('teacher_response_missing_receipt')
        generated=[r['choices'][0]['message'] for r in public_raw]
        assistants=[m for m in trajectory['messages'] if m['role']=='assistant']
        if len(generated)!=len(assistants):
            raise ValueError('assistant_message_count_mismatch')
        for original, retained in zip(generated,assistants):
            if {k:v for k,v in original.items() if k in {'role','content','tool_calls','reasoning_content'}}!=retained:
                raise ValueError('teacher_message_changed')
        results.append({'sample_id':source['sample_id'],'native_messages_exact':True,
                        'teacher_image_exact':True,'provider_receipts_verified':True})
    report={'passed':bool(results),'exported':len(results),'rows':results,
            'data_sha256':digest(pilot/'data.jsonl'),'lineage_sha256':digest(pilot/'lineage.jsonl')}
    write(pilot/'integrity-audit.json',report)
    return report
