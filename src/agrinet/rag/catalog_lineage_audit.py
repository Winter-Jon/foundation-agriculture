"""Audit source-scoped wiki identities without assuming code namespaces match."""
import argparse
import json
import re
from pathlib import Path
from agrinet.rag.e344_prepare import digest, rows
from agrinet.rag.e344_pipeline import write, write_rows


def norm(text):
    return re.sub(r'[^a-z0-9]+',' ',str(text).casefold()).strip()


def audit(base, source_root, registry, output):
    base,source_root,registry,output=map(Path,(base,source_root,registry,output))
    output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(base.read_text())['description']
    sources={}; aliases={}
    for row in rows(registry):
        names=[row['canonical_english_name']]+[a['value'] for a in row.get('aliases',[]) if a.get('language')=='en' and a.get('scope')=='answer']
        for name in names: aliases.setdefault(norm(name),set()).add(row['canonical_code'])
    decisions=[]; retained={}; quarantine={}
    for aid,row in sorted(catalog.items()):
        dataset=row.get('source_dataset') or aid.split('::')[0]
        if dataset not in sources:
            path=source_root/dataset/'knowledge_base.json'
            sources[dataset]=json.loads(path.read_text())['description'] if path.exists() else {}
        original=sources[dataset].get(row['code'])
        source_name=original.get('english_name') if original else None
        candidates=sorted(aliases.get(norm(source_name),set())) if source_name else []
        matches=bool(source_name) and norm(source_name)==norm(row.get('english_name'))
        reason='source_name_matches' if matches else 'source_name_changed_requires_mapping_review' if original else 'missing_source_record'
        decision={'artifact_id':aid,'source_code':row['code'],'current_name':row.get('english_name'),
                  'original_source_name':source_name,'exact_alias_candidate_codes':candidates,'disposition':reason}
        decisions.append(decision)
        if matches:retained[aid]=row
        else:quarantine[aid]=decision
    write_rows(output/'mapping-audit.jsonl',decisions)
    write(output/'quarantine.json',quarantine)
    write(output/'provisional-catalog.json',{'description':retained})
    result={'rows':len(catalog),'source_name_matches':len(retained),'quarantined_for_review':len(quarantine),
            'base_sha256':digest(base),'registry_sha256':digest(registry),
            'source_hashes':{s:digest(source_root/s/'knowledge_base.json') for s in sources if (source_root/s/'knowledge_base.json').exists()},
            'automatic_relabeling':False,'live_index_changed':False,
            'limitation':'Name difference is a review signal, not proof of wrong biology. Exact aliases suggest mappings but do not verify description/image identity.'}
    write(output/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['base','source-root','registry','output']:p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();print(json.dumps(audit(a.base,a.source_root,a.registry,a.output)))
