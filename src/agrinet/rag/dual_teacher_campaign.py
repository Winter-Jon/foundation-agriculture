"""Bounded eight-cell dual teacher smoke with exact multimodal export."""
import argparse,copy,json,hashlib,fcntl
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from agrinet.rag.e344_collect import collect_sample
from agrinet.rag.e344_runtime import Runtime
from agrinet.rag.e344_export import export_one
from agrinet.rag.e344_prepare import digest,rows
from agrinet.rag.e344_pipeline import write,write_rows
from agrinet.rag.dual_teacher import simple_audit,export_multimodal,diagnostic_evidence


def run(config):
    root=Path(config['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=list(rows(config['source']))
    sample_count = int(config.get('sample_count', 8))
    cells = Counter((r['arm'],r['question_type'],r['domain']) for r in sources)
    full_batch = config.get('collection_mode') == 'quota_batch'
    if len(sources)!=sample_count or sample_count < 1 or (not full_batch and (sample_count not in (8,32) or len(cells)!=8 or set(cells.values())!={sample_count//8})):
        raise ValueError('invalid_eight_cell_quota')
    if len({r['image_sha256'] for r in sources}) != sample_count: raise ValueError('duplicate_query_image')
    import requests
    dataset=Path('datasets/AgriNet-1K/wiki_source_scoped_v3')
    dataset_manifest=json.loads((dataset/'manifest.json').read_text())
    for name,sha in dataset_manifest['files'].items():
        if digest(dataset/name)!=sha:raise ValueError('dataset_hash_changed')
    with requests.Session() as session:
        session.trust_env=False
        health=session.get(config['rag_endpoint'].removesuffix('/search')+'/health',timeout=120)
        health.raise_for_status()
        if not Path(health.json()['database']).samefile(dataset/'index.db'):raise ValueError('wrong_retrieval_database')
    runtime=Runtime(json.loads(Path(config['bindings']).read_text()),rag_endpoint=config['rag_endpoint'],device=config['classifier_device'])
    frozen={'dataset_manifest_sha256':digest(dataset/'manifest.json'),'config':config,'source_sha256':digest(config['source']),'bindings_sha256':digest(config['bindings']),
            'implementation':{str(p):digest(p) for p in [Path(__file__),Path('src/agrinet/rag/dual_teacher.py'),Path('src/agrinet/rag/e344_collect.py'),Path('src/agrinet/rag/dual_teacher_reaudit.py'),Path('src/agrinet/rag/answer_normalization.py'),Path('src/agrinet/vlm/full_tool.py'),Path('src/agrinet/rag/protocol_repair.py')]}}
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=frozen:raise ValueError('campaign_binding_changed')
    write(manifest,frozen)
    def one(row):
        assert digest(row['image_path'])==row['image_sha256']
        directory=root/'samples'/row['sample_id']
        try:
            return collect_sample(row,root=directory,teacher=runtime.teacher,classifier=runtime.classifier,
                retrieve=lambda row,args:diagnostic_evidence(runtime.retrieve(row,args)),private_audit=lambda row,result:simple_audit(runtime.teacher,row,result),reference_mode=True)
        except Exception as exc:
            outcome={'training_eligible':False,'disposition':'execution_failure','exception_type':type(exc).__name__}
            write(directory/'outcome.json',outcome);return outcome
    workers = int(config.get('workers', 16))
    if not 1 <= workers <= 32: raise ValueError('invalid_worker_limit')
    with ThreadPoolExecutor(max_workers=min(workers,len(sources))) as pool:
        outcomes=list(pool.map(one,sources))
    data=[];lineage=[]
    for source,result in zip(sources,outcomes):
        if not result.get('training_eligible'):continue
        # Validate original native evidence using the single-query projection;
        # export actual multimodal context afterward, preserving every reference.
        projected=copy.deepcopy(result)
        first_image = next(i for i,m in enumerate(projected['messages']) if m['role']=='user' and isinstance(m.get('content'),list))
        projected['messages']=[m for i,m in enumerate(projected['messages']) if i==first_image or not(m['role']=='user' and isinstance(m.get('content'),list))]
        try:
            student,provenance=export_one(source,projected)
            actual=copy.deepcopy(result['messages']);actual[-1]['content']=student['messages'][-1]['content']
            messages,images=export_multimodal(actual,root/'images'/source['sample_id'])
            student.update(messages=messages,images=images)
            provenance.update(original_trajectory_sha256=digest(root/'samples'/source['sample_id']/'trajectory.json'),original_trajectory_path=str(root/'samples'/source['sample_id']/'trajectory.json'),
                              reference_records=result.get('reference_records',[]),student_row_sha256=hashlib.sha256(json.dumps(student,sort_keys=True).encode()).hexdigest())
            data.append(student);lineage.append(provenance)
        except ValueError as exc:
            result['training_eligible']=False;result['export_error']=str(exc)
    write_rows(root/'data.jsonl',data);write_rows(root/'lineage.jsonl',lineage)
    write(root/'report.json',{'attempted':len(sources),'qualified':len(data),'outcomes':[{'id':r['sample_id'],'disposition':o.get('disposition'),'eligible':o.get('training_eligible'),'error':o.get('export_error',o.get('exception_type'))} for r,o in zip(sources,outcomes)]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    run(yaml.safe_load(Path(a.config).read_text())['parameters'])
