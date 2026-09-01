#!/usr/bin/env python3
"""Collect exact Hermes v2 targets; dry-run by default and fail closed on delivery."""
from __future__ import annotations

import argparse, base64, json, mimetypes, os, signal, urllib.request
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import cell_of, load_hermes_1to1_specification, long_direct_audit_errors, strict_rag_trajectory_errors
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from agrinet.rag.hermes_protocol import parse_hermes_tool_calls
from agrinet.rag.tool_schema import TOOL_NAME, tools_list, validate_tool_arguments


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=Path, required=True); p.add_argument('--route', choices=('direct','rag'), required=True)
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--limit', type=int, default=1); p.add_argument('--offset', type=int, default=0); p.add_argument('--target-id')
    p.add_argument('--rag-api', default='http://127.0.0.1:8077'); p.add_argument('--model', default='gpt-5.6-terra'); p.add_argument('--oracle', action='store_true')
    p.add_argument('--request-timeout', type=int, default=120, help='Hard wall-clock timeout for one teacher or retrieval request.')
    p.add_argument('--dry-run', action='store_true'); return p.parse_args()


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
    return {'type':'image_url','image_url':{'url':f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}'}}


class RequestTimeout(TimeoutError):
    pass


def post(url: str, payload: dict[str, Any], headers: dict[str,str] | None = None, *, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload,ensure_ascii=False).encode(), headers=headers or {'Content-Type':'application/json'}, method='POST')
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout))
    def timed_out(*_: Any) -> None: raise RequestTimeout(f"request exceeded {timeout}s: {url}")
    signal.signal(signal.SIGALRM, timed_out)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]: signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def teacher(messages: list[dict[str,Any]], model: str, *, timeout: int = 120) -> str:
    key, base = os.environ['YUNWU_API_KEY'], os.environ.get('YUNWU_API_BASE_URL','https://yunwu.ai/v1').rstrip('/')
    response = post(f'{base}/chat/completions', {'model':model,'messages':messages,'temperature':0.2,'max_tokens':2048}, {'Authorization':f'Bearer {key}','Content-Type':'application/json'}, timeout=timeout)
    return str(response['choices'][0]['message'].get('content') or '')


def prompt(target: dict[str,Any], route: str) -> str:
    language = 'Chinese' if target['language']=='zh' else 'English'
    answer = 'only the public option letter' if target['question_type']=='option' else 'only the canonical public class name'
    if route == 'direct':
        if target['language'] == 'zh':
            return f'''生成一条中文农业视觉对比答案，只使用图像。严格返回 <think>...</think><answer>...</answer>。<think> 内必须全部使用中文：以 1.、2.、3.、4. 开头分别写四条完整的可见特征或候选比较；随后用一句中文说明为何可见特征支持所选类别；再写三句完整中文句子，分别以“排除”或“不考虑”开头，各自只命名一个不同的近邻类别，并明确说明图中可见的排除理由。每条观察和排除都要写出颜色、形状、部位、纹理、病斑或虫体等具体视觉细节，避免短语、重复或英文。不要提及检索、工具、标签、百科、置信度或隐藏信息。答案中只包含 {answer}。{public_option_question(target)} Internal quality-control target: {target["canonical_name"]}. Use it only to check the final semantic choice; never mention that it was supplied or call it a label/target.'''
        return f'''Generate one {language} agricultural visual-comparison answer. Use only the image. Return exactly <think>...</think><answer>...</answer>. Inside <think>, begin four separate lines exactly with 1., 2., 3., and 4.; each line is a visible-trait candidate. Then select one from visible evidence. Add three separate full sentences, each beginning with I exclude or I rule out, each naming exactly one different nearby alternative and one visible reason. Do not combine multiple alternatives in one exclusion sentence. Do not mention retrieval, tools, labels, wiki, confidence, or hidden information. Answer contains {answer}. {public_option_question(target)} Internal quality-control target: {target["canonical_name"]}. Use it only to check the final semantic choice; never mention that it was supplied or call it a label/target.'''
    oracle = f" Private oracle quality target: {target['canonical_name']}. Use it only to verify the final semantic choice after public retrieval evidence supports it; never mention this target or its source." if target.get("generation_route") == "oracle_grounded" else ""
    return f'''Generate phase one of a Hermes agricultural RAG trace in {language}. Inspect the image, then return exactly a <think> block followed immediately by one <tool_call> block; do not output markdown or bare JSON. The think must list three or four numbered descriptive visual candidates, not a final diagnosis. The tool call must have this complete shape: <tool_call>{{"name":"{TOOL_NAME}","arguments":{{"query":"leaf color and lesion shape","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"visible host organ color and lesion traits"}}}}</tool_call>. Replace only the example query and rationale with image-based visible traits. Do not use a class name, answer, code, or hidden label.{oracle}'''


def answer_body(content: str) -> str:
    return content.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip() if "<answer>" in content else ""


@lru_cache(maxsize=1)
def canonical_answer_names() -> dict[str, set[str]]:
    """Return public English/Chinese names for exact Open-answer matching."""
    catalog = Path(__file__).resolve().parents[4] / "outputs/milvus/wiki_similar_classes_siglip2_report.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entries = payload.get("classes") if isinstance(payload, dict) else []
    result: dict[str, set[str]] = {}
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict) or not item.get("code"):
            continue
        result[str(item["code"])] = {
            str(item.get(key) or "").strip().casefold()
            for key in ("english_name", "chinese_name")
        } - {""}
    return result


@lru_cache(maxsize=1)
def verified_open_answer_aliases() -> dict[str, set[str]]:
    """Conservative public common-name equivalents for Open answers.

    The catalog's pest entries use scientific names, while the teacher often
    supplies an unambiguous public English common name.  Keep this mapping
    deliberately small and one-to-one; ambiguous names (for example cotton
    bollworm and fall armyworm) are intentionally absent.
    """
    return {
        "N05001": {"house cricket"},
        "N05004": {"black cutworm"},
        "N05005": {"citrus longhorned beetle"},
        "N05007": {"cotton aphid"},
        "N05008": {"oriental fruit fly"},
        "N05009": {"cabbage aphid"},
        "N05011": {"mediterranean fruit fly"},
        "N05016": {"long-winged conehead"},
        "N05018": {"banana weevil"},
        "N05019": {"codling moth"},
        "N05021": {"potato leafhopper"},
        "N05025": {"oriental mole cricket"},
        "N05026": {"two-spotted cricket"},
        "N05028": {"great orange tip"},
        "N05032": {"cottony cushion scale"},
        "N05044": {"mourning cloak"},
        "N05046": {"strawberry root weevil"},
        "N05048": {"citrus leafminer"},
        "N05049": {"indian cabbage white"},
        "N05050": {"diamondback moth"},
        "N05062": {"beet armyworm"},
        "N05066": {"two-spotted spider mite"},
        "N05067": {"great green bush-cricket"},
        "N05072": {"common birdwing"},
    }


def answer_matches_target(target: dict[str, Any], content: str) -> bool:
    answer = answer_body(content)
    if target.get("question_type") == "option":
        return answer == target.get("correct_option")
    names = {str(target.get(key) or "").strip().casefold() for key in ("canonical_name", "canonical_name_en", "canonical_name_zh", "class_name", "label_name")} - {""}
    code = str(target.get("canonical_class") or "")
    names |= canonical_answer_names().get(code, set())
    names |= verified_open_answer_aliases().get(code, set())
    return not names or answer.casefold() in names


def private_oracle_text_in_messages(messages: list[dict[str, Any]]) -> bool:
    """Reject explicit teacher-only Oracle wording from a persisted trace."""
    markers = ("private oracle", "oracle quality target", "internal quality-control target")
    return any(any(marker in str(message.get("content") or "").casefold() for marker in markers) for message in messages)


def known_rejection_record(*, target: dict[str, Any], row: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Persist a completed trajectory for future audit without promoting it.

    Completed Direct/RAG calls are known outcomes even if a quality or answer
    gate rejects them.  Keeping the entire post-sanitisation row makes later
    contract changes auditable.  Unknown delivery remains a target/error-only
    record because no complete trajectory is available.
    """
    return {
        "target": target,
        "errors": sorted(set(errors)),
        "status": "known_strict_rejection",
        "trajectory": row,
    }


def public_retrieval_evidence(raw: dict[str, Any], *, query: str, retrieval_type: str) -> dict[str, Any]:
    """Persist an intelligible public catalog view, never internal retrieval IDs."""
    hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or raw.get("results") or []
    if not isinstance(hits, list):
        raise ValueError("retrieval returned no public result list")
    results: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        english_name = str(hit.get("english_name") or hit.get("name") or "").strip()
        chinese_name = str(hit.get("chinese_name") or "").strip()
        if not (english_name or chinese_name):
            continue
        item: dict[str, Any] = {"rank": int(hit.get("rank") or len(results) + 1)}
        if english_name:
            item["name"] = english_name
        if chinese_name:
            item["name_zh"] = chinese_name
        for source, destination in (("alias_en", "aliases_en"), ("alias_cn", "aliases_zh")):
            aliases = hit.get(source)
            cleaned = [str(value).strip() for value in aliases if str(value).strip()] if isinstance(aliases, list) else []
            if cleaned:
                item[destination] = cleaned
        if isinstance(hit.get("distance"), (int, float)):
            item["similarity"] = round(float(hit["distance"]), 6)
        results.append(item)
    if not results:
        raise ValueError("retrieval returned no named public results")
    return {"source": "AgriNet public reference catalog", "retrieval_type": retrieval_type, "query": query, "results": results}


def oracle_capacity_errors(collection_root: Path, targets: list[dict[str, Any]]) -> list[str]:
    """Fail before a teacher request if an Oracle cell would exceed its final cap."""
    cap = int(load_hermes_1to1_specification()["oracle_rag_cap_per_cell"])
    existing: Counter[tuple[str, str, str]] = Counter()
    for path in collection_root.glob("*/accepted.jsonl"):
        for row in read_jsonl(path):
            metadata = row.get("metadata") or {}
            if metadata.get("generation_route") == "oracle_grounded":
                existing[cell_of(row)] += 1
    requested = Counter(cell_of(target) for target in targets)
    return [f"oracle_cap_preflight:{'/'.join(cell)}:{existing[cell]}+{count}>{cap}" for cell, count in sorted(requested.items()) if existing[cell] + count > cap]


def public_option_question(target: dict[str, Any]) -> str:
    if target.get("question_type") != "option":
        return ""
    labels = target.get("candidate_labels") or []
    if len(labels) != 4:
        raise ValueError("Option target lacks four public candidates")
    names = [str(item.get("chinese_name") if target.get("language") == "zh" else item.get("name") or "") for item in labels]
    if any(not name for name in names):
        raise ValueError("Option target has an unnamed public candidate")
    instruction = "请选择图中病虫害的规范名称，只在答案标签中输出选项字母。" if target.get("language") == "zh" else "Select the canonical name shown in the image; output only the option letter in the answer tag."
    return "\n" + instruction + "\n" + "\n".join(f"{letter}. {name}" for letter, name in zip("ABCD", names))


def main() -> int:
    a=args(); all_targets=read_jsonl(a.targets)
    if a.oracle and a.route != 'rag':
        raise ValueError('--oracle is permitted only for RAG collection')
    targets = [target for target in all_targets if target.get("target_id") == a.target_id] if a.target_id else all_targets[a.offset:a.offset+a.limit]
    if a.target_id and len(targets) != 1:
        raise ValueError(f"target ID must resolve exactly once: {a.target_id}")
    if a.oracle:
        targets = [{**target, "generation_route": "oracle_grounded", "label_visible_to_teacher": True, "oracle_reason": "semantic_recollection_pilot"} for target in targets]
        capacity_errors = oracle_capacity_errors(a.output_dir.parent, targets)
        if capacity_errors:
            raise ValueError("; ".join(capacity_errors))
    if not a.dry_run and any((a.output_dir / name).exists() for name in ("accepted.jsonl", "rejected.jsonl")):
        raise RuntimeError(f"refusing collection output reuse: {a.output_dir}")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    preview={'route':a.route,'targets':len(targets),'dry_run':a.dry_run,'target_ids':[x['target_id'] for x in targets],'tools':tools_list() if a.route=='rag' else []}
    (a.output_dir/'preview.json').write_text(json.dumps(preview,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if a.dry_run: print(json.dumps(preview,ensure_ascii=False)); return 0
    rows=[]; rejected=[]
    for target in targets:
        try:
            image=Path(target['query_image']); first=teacher([{'role':'system','content':prompt(target,a.route)},{'role':'user','content':[{'type':'text','text':'<image> Identify this.'},image_part(image)]}],a.model, timeout=a.request_timeout)
            if a.route=='direct':
                row={'sample_id':target['target_id'],'images':[target['query_image']],'messages':[{'role':'user','content':'<image> Identify this.'},{'role':'assistant','content':first}],'metadata':target}
                errors=long_direct_audit_errors(row,forbidden_hashes=set())
                if not answer_matches_target(target, first): errors.append('answer_target_mismatch')
            else:
                calls=parse_hermes_tool_calls(first[first.find('<tool_call>'):],tool_name=TOOL_NAME,validate_arguments=validate_tool_arguments)
                if len(calls)!=1: raise ValueError('invalid first Hermes tool call')
                raw_evidence=post(f"{a.rag_api.rstrip('/')}/search/{calls[0]['arguments']['retrieval_type']}", {'image_path':str(image),'top_k':calls[0]['arguments']['top_k'],'text':calls[0]['arguments']['query']}, timeout=a.request_timeout)
                evidence=public_retrieval_evidence(raw_evidence, query=calls[0]['arguments']['query'], retrieval_type=calls[0]['arguments']['retrieval_type'])
                oracle_note = (' Private oracle quality target is '+target['canonical_name']+'. Use it only after public evidence supports it; never mention the target or source.' if target.get('generation_route') == 'oracle_grounded' else '')
                final_prompt='Public retrieval evidence:\n'+json.dumps(evidence,ensure_ascii=False)+'\nReturn exactly <think>...</think><answer>...</answer>. Inside think, list three numbered candidates on separate lines beginning 1., 2., and 3.; each candidate line must pair a candidate name with a concrete visible trait (color, lesion shape, organ, body form, wing pattern, texture, or damage), never a bare class-name list. Then write one sentence confirming the chosen candidate from public evidence and the visible trait. Then write two separate sentences; each must begin Evidence excludes or Evidence rules out, name one different candidate, and state a specific retrieved/visible morphology or symptom conflict. A lower rank or similarity score alone is never an exclusion reason. '+('Answer only A-D.' if target['question_type']=='option' else 'Answer only the canonical public class name in either its English or Chinese public form.')+public_option_question(target)+oracle_note
                final=teacher([{'role':'system','content':final_prompt},{'role':'user','content':[{'type':'text','text':'<image> Identify this.'},image_part(image)]}],a.model, timeout=a.request_timeout)
                row={'sample_id':target['target_id'],'images':[target['query_image']],'tools':tools_list(),'messages':[{'role':'user','content':'<image> Identify this.'},{'role':'assistant','content':first.split('<tool_call>',1)[0].strip()},{'role':'tool_call','content':json.dumps({'name':TOOL_NAME,'arguments':calls[0]['arguments']},ensure_ascii=False)},{'role':'tool','content':json.dumps(evidence,ensure_ascii=False)},{'role':'assistant','content':final}],'metadata':target}
                errors=strict_rag_trajectory_errors(row)
                if private_oracle_text_in_messages(row['messages']): errors.append('private_oracle_text_leak')
                if not answer_matches_target(target, final): errors.append('answer_target_mismatch')
            (rows if not errors else rejected).append(row if not errors else known_rejection_record(target=target, row=row, errors=errors))
        except Exception as exc: rejected.append({'target':target,'errors':['unknown_delivery_or_runtime_error'],'error':str(exc)})
        # A long cell must survive a later stalled request.  Rewriting a small
        # JSONL checkpoint is deliberate: it keeps a complete lineage ledger.
        write_jsonl(a.output_dir/'accepted.jsonl',rows); write_jsonl(a.output_dir/'rejected.jsonl',rejected)
    print(json.dumps({'accepted':len(rows),'rejected':len(rejected),'output':str(a.output_dir)},ensure_ascii=False)); return 2 if rejected else 0

if __name__=='__main__': raise SystemExit(main())
