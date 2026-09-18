"""Offline case-normalized scoring; preserve all original teacher messages."""
import argparse,json,hashlib,base64
from pathlib import Path
from agrinet.rag.answer_normalization import normalized_review,answer_key,resolve,POLICY
from agrinet.rag.e344_export import export_one
from agrinet.rag.e344_prepare import rows,digest
from agrinet.rag.e344_pipeline import write,write_rows


def run(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('immutable_rescore_output_exists')
    output.mkdir(parents=True)
    data=[];lineage=[];scores=[]
    for source in rows(root/'pilot-source.jsonl'):
        path=root/'pilot/samples'/source['sample_id']/'trajectory.json'
        original=json.loads(path.read_text());reviewed=normalized_review(source,original)
        score={'sample_id':source['sample_id'],'raw_answer':original.get('validation',{}).get('answer'),
               'canonical_answer':resolve(original.get('validation',{}).get('answer',''))['canonical_answer'], 'class_correct':reviewed.get('private_audit',{}).get('semantic')=='correct','original_eligible':original.get('training_eligible',False),
               'eligible':False,'source_trajectory_sha256':digest(path)}
        if reviewed.get('training_eligible'):
            try:student,trace=export_one(source,reviewed)
            except ValueError as exc:score['exclusion']=str(exc)
            else:
                image=next(part for message in original['messages'] if isinstance(message.get('content'),list)
                           for part in message['content'] if part.get('type')=='image_url')
                image_path=output/'images'/(source['sample_id']+'.jpg');image_path.parent.mkdir(exist_ok=True)
                image_path.write_bytes(base64.b64decode(image['image_url']['url'].split(',',1)[1],validate=True))
                student['images']=[str(image_path)]
                trace.update(original_trajectory_path=str(path),original_trajectory_sha256=digest(path),
                             normalization=POLICY,
                             student_row_sha256=hashlib.sha256(json.dumps(student,sort_keys=True).encode()).hexdigest())
                data.append(student);lineage.append(trace);score['eligible']=True
        scores.append(score)
    write_rows(output/'data.jsonl',data);write_rows(output/'lineage.jsonl',lineage);write_rows(output/'scores.jsonl',scores)
    write(output/'report.json',{'policy':POLICY,'attempted':len(scores),'qualified':len(data),
          'qualified_rate':len(data)/len(scores),'newly_qualified':sum(s['eligible'] and not s['original_eligible'] for s in scores),
          'teacher_requests':0,'original_trajectories_unchanged':True, 'export_final_answer_canonicalized':True})
    print(json.dumps({'qualified':len(data),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
