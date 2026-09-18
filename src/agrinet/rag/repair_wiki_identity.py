"""Rebuild source-scoped catalogue; ambiguous classes keep external identities."""
import argparse
import hashlib
import json
from pathlib import Path
from agrinet.rag.e344_prepare import rows,digest
from agrinet.rag.e344_pipeline import write,write_rows
from agrinet.rag.catalog_lineage_audit import norm


def source_contents(description):
    """Keep all populated original source slots, including when slot 1 failed."""
    fields = sorted(k for k in description if k == "content" or
                    (k.startswith("content_") and k[8:].isdigit()))
    return [(k, description[k]) for k in fields
            if isinstance(description[k], str) and description[k].strip()]


def build(base,source_root,registry,output):
    base,source_root,registry,output=map(Path,(base,source_root,registry,output))
    if (output/'catalog.json').exists():raise ValueError('immutable_output_exists')
    original=json.loads(base.read_text())['description']
    labels=list(rows(registry));by_code={r['canonical_code']:r for r in labels};aliases={}
    for r in labels:
        for name in [r['canonical_english_name'],r['canonical_chinese_name']]+[a['value'] for a in r.get('aliases',[]) if a.get('scope')=='answer']:
            key=norm(name) if name.isascii() else name.strip()
            if key:aliases.setdefault(key,set()).add(r['canonical_code'])
    sources={};result={};decisions=[];excluded=[]
    for aid,current in sorted(original.items()):
        dataset=current['source_dataset'];code=current['code']
        if dataset not in sources:sources[dataset]=json.loads((source_root/dataset/'knowledge_base.json').read_text())['description']
        source=sources[dataset].get(code)
        if not source:excluded.append({'artifact_id':aid,'reason':'missing_original_source'});continue
        candidates=set()
        for name in [source.get('english_name',''),source.get('chinese_name','')]:
            key=norm(name) if name.isascii() else name.strip()
            candidates.update(aliases.get(key,set()))
        # Only primary-source codes share the approved registry namespace.
        if dataset=='agri_disease_pest_wiki' and code in by_code:candidates={code}
        target=next(iter(candidates)) if len(candidates)==1 else None
        label=by_code.get(target,{})
        description=source.get('description',{})
        parts=source_contents(description)
        if not parts:
            excluded.append({'artifact_id':aid,'reason':'missing_original_description'});continue
        fixed={**current,'source_code':code,'canonical_class_code':target,
               'code':target or 'EXT_'+hashlib.sha256(aid.encode()).hexdigest()[:24],
               'english_name':label.get('canonical_english_name',source['english_name']),
               'chinese_name':label.get('canonical_chinese_name',source.get('chinese_name','')),
               'description':{f'content_{i}':text for i,(_,text) in enumerate(parts,1)},
               'description_sources':[dict(field=k, source=description.get('source_'+k.split('_')[-1]),
                    url=description.get('url_'+k.split('_')[-1])) for k,_ in parts],
               'alias_en':[],'alias_cn':[],
               'similar_english_classes':[],'similar_chinese_classes':[],
               'identity_version':'source-scoped-wiki/v1'}
        # No migrated similarity names survive; they may share the same collision.
        result[aid]=fixed
        decisions.append({'artifact_id':aid,'source_code':code,'canonical_class_code':target,
                          'original_name':source['english_name'],'repaired_name':fixed['english_name'],
                          'previous_name':current['english_name'],'method':'primary_namespace' if dataset=='agri_disease_pest_wiki' and target else 'unique_source_name_alias' if target else 'external_identity',
                          'candidate_codes':sorted(candidates)})
    output.mkdir(parents=True,exist_ok=True)
    write_rows(output/'mapping.jsonl',decisions);write_rows(output/'excluded.jsonl',excluded)
    write(output/'catalog.json',{'description':result})
    report={'input_rows':len(original),'retained':len(result),'excluded':len(excluded),
            'canonical':sum(r['canonical_class_code'] is not None for r in decisions),
            'external':sum(r['canonical_class_code'] is None for r in decisions),
            'base_sha256':digest(base),'registry_sha256':digest(registry),
            'source_hashes':{s:digest(source_root/s/'knowledge_base.json') for s in sources},
            'catalog_sha256':digest(output/'catalog.json'),'live_index_changed':False,
            'limitations':['Original descriptions restored; independent biological image review remains necessary','Similarity links cleared pending identity-safe recomputation']}
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','source-root','registry','output']:p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();print(json.dumps(build(a.base,a.source_root,a.registry,a.output)))
