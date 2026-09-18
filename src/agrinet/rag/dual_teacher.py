"""Versioned public-only diagnostic auditor and bounded reference-image transport."""
import hashlib
import json
from pathlib import Path
from agrinet.rag.e35_transport import transport_image

VERSION = 'dual-teacher-reference/v2'
SAMPLER_GUIDANCE = '''Before calling any tool, describe visible host/organ, lesions or insect morphology and what cannot be assessed. Do not diagnose yet. After tools, explicitly explain any revised observation. Use name search for ONE canonical class name only, semantic for a specific unresolved distinction, and visual for similar examples. Each extra search must resolve a new evidence gap. Briefly screen the deduplicated union, then compare the genuinely closest alternatives using visible support, conflicts and unknowns. Compare all four options when supplied. Rank is not a visible trait. Absence and not-visible are different. Reference images are visible ONLY when attached with a REF identifier; paths are not images. Cite supplied artifact IDs or REF IDs when relying on evidence. Reference examples do not prove species identity. Retrieved text can be generic or incomplete: do not invent diagnostic distinctions. If decisive distinctions remain unobservable, use INSUFFICIENT_EVIDENCE.'''
AUDITOR_GUIDANCE = '''Audit the diagnostic explanation using ONLY the query image and supplied public reasoning. You are not given reference images or tool evidence, so do not claim to verify reference-image descriptions or source quotations. Check visible-observation conflicts, unexplained changes after retrieval, coherent comparisons, valid exclusions, nearest-alternative choice and proportionate uncertainty. Tool rank is not visual evidence. Do not judge by agreement with a hidden label (none is provided). Return JSON with verdict pass/fail, issues (at most 3 objects with type and short reason), and a short reason. Issue types: observation_conflict, reasoning_inconsistency, unsupported_discrimination, invalid_exclusion, overconfidence. No correction or target class is requested.'''


def audit_payload(row, messages):
    """Never pass private truth, reference images, or tool results to auditor."""
    public=[]
    for message in messages:
        if message['role']=='assistant':
            public.append({k:message[k] for k in ('content','tool_calls') if k in message})
    image,_=transport_image(Path(row['image_path']),max_side=1024)
    return {'model':'gpt-5.6-sol','temperature':0,'max_tokens':384,
            'response_format':{'type':'json_object'},'messages':[
                {'role':'system','content':AUDITOR_GUIDANCE},
                {'role':'user','content':[{'type':'text','text':json.dumps({
                    'question':'Identify the agricultural disease or pest in the query image.',
                    'options':row.get('public_options',[]),'reasoning_in_order':public},ensure_ascii=False)},image]}]}


def parse_audit(raw):
    result=json.loads(raw['choices'][0]['message']['content'])
    if result.get('verdict') not in ('pass','fail') or not isinstance(result.get('issues'),list):
        raise ValueError('invalid_simple_audit')
    if result['verdict']=='pass' and result['issues']:
        raise ValueError('contradictory_simple_audit')
    return result


def reference_attachments(response, image_root, seen, query_sha, query_path=None):
    """One image per hit, <=6 distinct images overall; retain exact attachments."""
    root=Path(image_root).resolve();parts=[];records=[]
    from PIL import Image
    import numpy as np
    from scipy.fft import dctn
    def phash(im):
        values=dctn(np.asarray(im.convert("L").resize((32,32)),dtype=float),norm="ortho")[:8,:8]
        return values > np.median(values.flatten()[1:])
    query_hash = phash(Image.open(query_path)) if query_path else None
    for hit in response.get('evidence',[]):
        metadata=hit['metadata'];paths=metadata.get('local_reference_images') or []
        preferred=metadata.get('matched_image')
        if preferred:paths=[preferred]+list(paths)
        for original in paths:
            path=root/Path(original).name
            if not path.is_file():continue
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha==query_sha:continue
            if query_hash is not None:
                with Image.open(path) as im:
                    if np.count_nonzero(query_hash != phash(im)) <= 6:continue
            if sha in seen:
                parts.append({'type':'text','text':f"{hit['artifact_id']}: reference already supplied as {seen[sha]}"});break
            if len(seen)>=6:break
            ref='REF'+str(len(seen)+1);seen[sha]=ref
            image,_=transport_image(path,max_side=512)
            parts.extend([{'type':'text','text':f"{ref} | {hit['artifact_id']} | {metadata['english_name']}"},image])
            records.append({'reference_id':ref,'artifact_id':hit['artifact_id'],'path':str(path),'sha256':sha})
            break
    return ({'role':'user','content':parts} if parts else None),records


def simple_audit(teacher, row, result):
    from agrinet.rag.answer_normalization import class_correct
    from agrinet.rag.dual_teacher_reaudit import revised_payload, POLICY
    payload=revised_payload(row,result['messages'])
    raw=teacher(payload);review=parse_audit(raw)
    return {'semantic':'correct' if class_correct(result['validation']['answer'],row['private']['truth_name']) else 'incorrect',
            'quality':review['verdict'],'issues':review['issues'],'reason':review.get('reason',''),
            'raw':raw,'audit_policy':POLICY}


def export_multimodal(messages, output):
    """Render every actually seen image in order; never drop reference evidence."""
    import base64
    import copy
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    exported=copy.deepcopy(messages);images=[]
    for message in exported:
        if not isinstance(message.get('content'),list):continue
        text=[]
        for part in message['content']:
            if part['type']=='text':text.append(part['text'])
            elif part['type']=='image_url':
                raw=base64.b64decode(part['image_url']['url'].split(',',1)[1],validate=True)
                path=output/(hashlib.sha256(raw).hexdigest()+'.jpg')
                if path.exists() and path.read_bytes()!=raw:raise ValueError('image_hash_collision')
                path.write_bytes(raw);images.append(str(path));text.append('<image>')
            else:raise ValueError('unsupported_multimodal_part')
        message['content']=chr(10).join(text)
    merged=[]
    for message in exported:
        if message["role"] == "user" and merged and merged[-1]["role"] in {"tool", "user"}:
            merged[-1]["content"] += chr(10) + message["content"]
        else:
            merged.append(message)
    return merged,images


def diagnostic_evidence(response):
    """Extract source sentences, never manufacture class distinctions."""
    import copy,re
    result=copy.deepcopy(response)
    for item in result.get('evidence',[]):
        meta=item['metadata'];original=meta.get('public_description','')
        sentences=re.split(r'(?<=[.!?。]) +', original.replace(chr(10), ' '))
        terms=re.compile(r'lesion|symptom|spot|wilt|wing|antenna|foreleg|hindleg|larva|adult|leaf|leaves|fruit|host|stem|病斑|症状|叶|成虫|幼虫',re.I)
        selected=[s.strip() for s in sentences if terms.search(s) and len(s.strip())>=20][:8]
        meta['diagnostic_evidence']={'artifact_id':item['artifact_id'],'extractive_quotes':selected,
            'limitations':'Source excerpts; not an independently validated differential diagnosis. Missing traits remain unknown.'}
        # Keep original description: exact evidence and context remain available.
    return result
