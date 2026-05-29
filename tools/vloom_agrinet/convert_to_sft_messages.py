import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

REQUIRED_RESULT_FIELDS = {
    "task_domain",
    "final_label",
    "candidate_labels",
    "query_visual_evidence",
    "positive_reference_alignment",
    "negative_reference_contrast",
    "wiki_evidence",
    "agricultural_interpretation",
    "uncertainty",
}

QUERY_TEMPLATES = {
    "disease": [
        "What is the name of the plant disease shown in this image?",
        "Can you identify the disease affecting this plant?",
        "What type of plant disease does this image show?",
        "Please identify the plant disease shown in this image.",
        "Classify the plant disease shown in the image.",
        "Determine the disease type visible in this plant.",
        "What plant disease is depicted here?",
        "Identify the specific disease shown in this image.",
        "Based on the symptoms, what disease is affecting this plant?",
        "What is the correct diagnosis for this plant?",
        "Which disease matches the symptoms shown?",
        "Can you diagnose the plant condition shown here?",
        "What pathogen-related condition is visible here?",
        "What disease name fits this plant's condition?",
        "What disease is shown in the image?",
    ],
    "pest": [
        "What is the name of the pest shown in this image?",
        "Can you identify the insect or pest in this image?",
        "What type of agricultural pest does this image show?",
        "Please identify the pest shown in this image.",
        "Classify the pest shown in the image.",
        "Determine the pest species visible here.",
        "What pest is depicted in this image?",
        "Identify the specific pest shown here.",
        "Based on the morphology, what pest is this?",
        "What is the correct species name of this pest?",
        "Which pest matches the image?",
        "Can you name this agricultural pest?",
        "What harmful organism is shown here?",
        "What pest name fits this specimen?",
        "What pest is shown in the image?",
    ],
}

QUERY_TEMPLATES_ZH = {
    "disease": [
        "这张图片显示的植物病害叫什么名字？",
        "你能识别出影响这种植物的病害吗？",
        "这张图片显示的是什么类型的植物病害？",
        "请识别这张图片中显示的植物病害。",
        "对图片中显示的植物病害进行分类。",
        "确定这种植物上可见的病害类型。",
        "这张图片展示的是什么植物病害？",
        "识别这张图片中显示的特定病害。",
        "根据症状判断，这是什么植物病害？",
        "这种植物的正确诊断是什么？",
        "哪种病害与显示的症状相符？",
        "你能诊断出图片中的植物状况吗？",
        "这里可见的是什么病原体相关病症？",
        "什么病害名称符合这种植物的状况？",
        "图片中显示的是什么病害？",
    ],
    "pest": [
        "这张图片中的害虫叫什么名字？",
        "你能识别出这张图片中的昆虫或害虫吗？",
        "这张图片显示的是什么类型的农业害虫？",
        "请识别这张图片中显示的害虫。",
        "对图片中显示的害虫进行分类。",
        "确定这里可见的害虫种类。",
        "这张图片展示的是什么害虫？",
        "识别这里显示的特定害虫。",
        "根据形态特征，这是什么害虫？",
        "这种害虫的正确物种名称是什么？",
        "哪种害虫与图片相符？",
        "你能命名这种农业害虫吗？",
        "这里显示的是什么有害生物？",
        "什么害虫名称符合这个标本？",
        "图片中显示的是什么害虫？",
    ],
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

SYSTEM_PROMPT_ZH = (
    "User 与 Assistant 之间的对话。用户提出问题，Assistant 进行解答。"
    "Assistant 首先在脑海中思考推理过程，然后向用户提供答案。"
    "推理过程和答案分别用 <think></think> 和 <answer></answer> 标签包裹，"
    "即：<think> 推理过程 </think><answer> 答案 </answer>"
)


def normalize_name(name: str) -> str:
    """Normalize a class name: remove underscores, extra spaces, strip."""
    if not name:
        return ""
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def build_name_mapping(template_vars: Dict[str, Any], lang: str = "en") -> Dict[str, str]:
    """Build a mapping from code to normalized name."""
    mapping: Dict[str, str] = {}
    for detail in template_vars.get("candidate_label_details", []):
        code = detail.get("code", "")
        if lang == "zh":
            raw_name = detail.get("chinese_name", "") or detail.get("name", "")
        else:
            raw_name = detail.get("name", "")
        if code and raw_name:
            mapping[str(code)] = normalize_name(raw_name)
    gt_label = template_vars.get("gt_label", "")
    gt_name = ""
    for detail in template_vars.get("candidate_label_details", []):
        if detail.get("code") == gt_label:
            if lang == "zh":
                gt_name = normalize_name(detail.get("chinese_name", "") or detail.get("name", ""))
            else:
                gt_name = normalize_name(detail.get("name", ""))
            break
    if gt_label and gt_name:
        mapping[str(gt_label)] = gt_name
    return mapping


def extract_think_and_json(raw_response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", raw_response, flags=re.DOTALL | re.IGNORECASE)
    think_text = think_match.group(1).strip() if think_match else ""

    remaining = raw_response
    if think_match:
        remaining = raw_response[think_match.end():]

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", remaining, flags=re.DOTALL | re.IGNORECASE)
    json_text = fence_match.group(1).strip() if fence_match else remaining.strip()

    parsed = None
    if json_text:
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            brace_match = re.search(r"\{.*\}", json_text, flags=re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass

    return think_text, parsed if isinstance(parsed, dict) else None


def pick_query(task_domain: str, candidate_names: List[str], key: str) -> str:
    templates = QUERY_TEMPLATES.get(task_domain, QUERY_TEMPLATES["disease"])
    rng = random.Random(key)
    template = rng.choice(templates)
    if "{candidates}" in template:
        names_str = ", ".join(candidate_names)
        return template.format(candidates=names_str)
    return template


def pick_query_zh(task_domain: str, candidate_names: List[str], key: str) -> str:
    templates = QUERY_TEMPLATES_ZH.get(task_domain, QUERY_TEMPLATES_ZH["disease"])
    rng = random.Random(key)
    template = rng.choice(templates)
    if "{candidates}" in template:
        names_str = ", ".join(candidate_names)
        return template.format(candidates=names_str)
    return template


def _clean_field_text(text: str, lang: str = "en") -> str:
    """Remove reference terminology from a text field."""
    if lang == "zh":
        text = re.sub(r"正例参考", "典型症状", text)
        text = re.sub(r"负例参考", "其他情况", text)
        text = re.sub(r"同类代表图像?", "典型症状", text)
        text = re.sub(r"对比样本", "其他情况", text)
        text = re.sub(r"参考图像?", "相关样本", text)
        text = re.sub(r"参考样本", "相关样本", text)
        return text
    text = re.sub(r"\bpositive reference[s]?\b", "typical symptoms", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnegative reference[s]?\b", "other conditions", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsame-class representative[s]?\b", "typical examples", text, flags=re.IGNORECASE)
    text = re.sub(r"\breference image[s]?\b", "related samples", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe reference[s]?\b", "the typical cases", text, flags=re.IGNORECASE)
    return text


def build_finer1_think(
    result: Dict[str, Any],
    name_mapping: Dict[str, str],
    original_think: str = "",
    lang: str = "en",
    wiki_zh_by_code: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build a FineR1-style think block using normalized names."""

    final_label = result.get("final_label", "")
    final_label_name = name_mapping.get(str(final_label), str(final_label))
    candidates = result.get("candidate_labels", [])
    candidate_names = [name_mapping.get(str(c), str(c)) for c in candidates]

    if lang == "en":
        section_titles = {
            "visual_features": "Analysis of Visual Features:",
            "candidates": "Candidate Subcategories:",
            "comparison": "Comparison Process:",
            "knowledge": "Knowledge Verification:",
            "final": "Final Prediction:",
            "based_on": "Based on the observed features, the following candidates are considered:",
            "comparison_with": "Comparison with {}:",
            "uncertainty": "Uncertainty assessment:",
            "prediction_is": "The prediction is:",
        }
    else:
        section_titles = {
            "visual_features": "视觉特征分析：",
            "candidates": "候选子类别：",
            "comparison": "对比过程：",
            "knowledge": "知识验证：",
            "final": "最终预测：",
            "based_on": "根据观察到的特征，考虑以下候选：",
            "comparison_with": "与{}的对比：",
            "uncertainty": "不确定性评估：",
            "prediction_is": "预测结果为：",
        }

    if original_think and len(original_think) > 100:
        has_structure = any(kw in original_think for kw in [
            "Analysis of", "Candidate", "Comparison", "Final Prediction", "Visual Features",
            "视觉特征", "候选子类别", "对比过程", "知识验证", "最终预测", "不确定性评估",
            "同类代表", "负例区分"
        ])
        if has_structure:
            is_original_zh = any(kw in original_think for kw in ["视觉特征", "候选子类别", "对比过程", "最终预测"])
            if (lang == "zh" and is_original_zh) or (lang == "en" and not is_original_zh):
                cleaned = original_think
                for code, name in name_mapping.items():
                    cleaned = re.sub(rf"\b{re.escape(code)}\b", name, cleaned)
                cleaned = _clean_field_text(cleaned, lang=lang)
                return cleaned

    lines = []

    evidence = result.get("query_visual_evidence", [])
    if evidence:
        lines.append(section_titles["visual_features"])
        for i, feat in enumerate(evidence, 1):
            lines.append(f"{i}. {_clean_field_text(feat, lang=lang)}")
        lines.append("")

    lines.append(section_titles["candidates"])
    lines.append(section_titles["based_on"])
    lines.append("")
    for name in candidate_names:
        lines.append(f"- {name}")
    lines.append("")

    lines.append(section_titles["comparison"])
    pos_alignments = result.get("positive_reference_alignment", [])
    if pos_alignments:
        lines.append(section_titles["comparison_with"].format(final_label_name))
        for align in pos_alignments:
            lines.append(f"  - {_clean_field_text(align, lang=lang)}")
        lines.append("")

    contrasts = result.get("negative_reference_contrast", [])
    if contrasts:
        for c in contrasts:
            candidate_code = c.get("candidate", "")
            candidate_name = name_mapping.get(str(candidate_code), str(candidate_code))
            why = c.get("why_less_likely", "")
            if candidate_name and why:
                lines.append(section_titles["comparison_with"].format(candidate_name))
                lines.append(f"  - {_clean_field_text(why, lang=lang)}")
                lines.append("")

    wiki = result.get("wiki_evidence", [])
    if lang == "zh" and wiki_zh_by_code and final_label in wiki_zh_by_code:
        zh_wiki = wiki_zh_by_code[final_label]
        if zh_wiki:
            wiki = zh_wiki
    if wiki and wiki != ["insufficient wiki knowledge"]:
        lines.append(section_titles["knowledge"])
        for w in wiki:
            lines.append(f"  - {_clean_field_text(w, lang=lang)}")
        lines.append("")

    interpretation = result.get("agricultural_interpretation", "")
    uncertainty = result.get("uncertainty", "unknown")
    lines.append(section_titles["final"])
    if interpretation:
        lines.append(_clean_field_text(interpretation, lang=lang))
    lines.append("")
    lines.append(f"{section_titles['uncertainty']} {uncertainty}")
    lines.append("")
    lines.append(f"{section_titles['prediction_is']} {final_label_name}")

    return "\n".join(lines)


def extract_teacher_trace(
    key: str,
    row: Dict[str, Any],
    name_mapping: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Extract a clean teacher trace record from VLooM result."""
    result = row.get("result") or {}
    raw_response = row.get("raw_response", "")
    template_vars = row.get("template_vars", {})

    if not result:
        think_text, result = extract_think_and_json(raw_response)
        if not result:
            return None
    else:
        think_text = row.get("think", "")
        if not think_text:
            think_text, _ = extract_think_and_json(raw_response)

    missing = sorted(REQUIRED_RESULT_FIELDS - set(result))
    if missing:
        return None

    final_label = result.get("final_label")
    candidates = result.get("candidate_labels", [])
    if final_label not in candidates:
        return None

    final_label_name = name_mapping.get(str(final_label), str(final_label))
    candidate_names = [name_mapping.get(str(c), str(c)) for c in candidates]

    teacher_think = build_finer1_think(result, name_mapping, think_text)

    return {
        "sample_id": key,
        "task_domain": result.get("task_domain", "unknown"),
        "query_image": template_vars.get("query_image", ""),
        "final_label_code": str(final_label),
        "final_label_name": final_label_name,
        "candidate_codes": [str(c) for c in candidates],
        "candidate_names": candidate_names,
        "think": teacher_think,
        "structured_result": result,
        "raw_response": raw_response,
        "uncertainty": result.get("uncertainty", "unknown"),
    }


def convert_to_student_sft(
    trace: Dict[str, Any],
    key: str,
    lang: str = "en",
    name_mapping: Optional[Dict[str, str]] = None,
    wiki_zh_by_code: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a teacher trace to student-safe SFT messages format."""
    task_domain = trace.get("task_domain", "unknown")
    candidate_codes = trace.get("candidate_codes", [])
    final_label_code = trace.get("final_label_code", "")
    query_image = trace.get("query_image", "")

    if name_mapping is not None:
        candidate_names = [name_mapping.get(str(c), str(c)) for c in candidate_codes]
        final_label_name = name_mapping.get(str(final_label_code), str(final_label_code))
    else:
        candidate_names = trace.get("candidate_names", [])
        final_label_name = trace.get("final_label_name", "")

    if lang == "zh" and (name_mapping is not None or wiki_zh_by_code is not None):
        structured_result = trace.get("structured_result", {})
        if structured_result:
            # Don't reuse English think for Chinese SFT — synthesize from structured fields
            teacher_think = build_finer1_think(
                structured_result,
                name_mapping or {},
                original_think="",
                lang=lang,
                wiki_zh_by_code=wiki_zh_by_code,
            )
        else:
            teacher_think = trace.get("think", "")
    else:
        teacher_think = trace.get("think", "")

    if lang == "zh":
        system_prompt = SYSTEM_PROMPT_ZH
        query_text = pick_query_zh(task_domain, candidate_names, key)
    else:
        system_prompt = SYSTEM_PROMPT
        query_text = pick_query(task_domain, candidate_names, key)

    user_parts: List[str] = []
    if query_image:
        user_parts.append("<image>")
    user_parts.append(query_text)
    user_content = "\n".join(user_parts)

    student_think = _clean_field_text(teacher_think, lang=lang)
    if lang == "zh":
        student_think = re.sub(r"典型症状", "已知特征", student_think)
        student_think = re.sub(r"其他情况", "替代选项", student_think)
        student_think = re.sub(r"相关样本", "已记录案例", student_think)
    else:
        student_think = re.sub(r"\btypical symptoms\b", "known characteristics", student_think, flags=re.IGNORECASE)
        student_think = re.sub(r"\bother conditions\b", "alternatives", student_think, flags=re.IGNORECASE)
        student_think = re.sub(r"\brelated samples\b", "documented cases", student_think, flags=re.IGNORECASE)
        student_think = re.sub(r"\bthe typical cases\b", "documented cases", student_think, flags=re.IGNORECASE)

    assistant_content = f"<think>\n{student_think}\n</think>\n\n<answer>{final_label_name}</answer>"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    record: Dict[str, Any] = {"messages": messages}
    if query_image:
        record["images"] = [query_image]

    return record


def _think_len(sft_record: Dict[str, Any]) -> int:
    for msg in sft_record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            m = re.search(r"<think>\s*(.*?)\s*</think>", content, flags=re.DOTALL)
            if m:
                return len(m.group(1).strip())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VLooM contrast CoT results to FineR1-style teacher traces and student SFT messages"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to VLooM result JSON")
    parser.add_argument("--output-sft-en", type=Path, required=True, help="Output English student SFT JSONL path")
    parser.add_argument("--output-sft-zh", type=Path, required=True, help="Output Chinese student SFT JSONL path")
    parser.add_argument("--output-traces", type=Path, help="Output teacher traces JSONL path (optional)")
    parser.add_argument("--bilingual-wiki", type=Path, default=Path("outputs/vlm_data/disease_pest/wiki_bilingual_structured.json"))
    parser.add_argument("--min-think-len", type=int, default=50, help="Minimum think text length to accept")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, args.log_level.upper()),
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    logger.info("Loading results from %s", args.input)
    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    wiki_zh_by_code: Dict[str, List[str]] = {}
    if args.bilingual_wiki.exists():
        with args.bilingual_wiki.open("r", encoding="utf-8") as f:
            wiki_data = json.load(f)
        for entry in wiki_data:
            code = entry.get("code", "")
            if not code:
                continue
            cn_contents = []
            for content in entry.get("contents", []):
                cn = content.get("cn", "")
                if cn:
                    cn_contents.append(cn)
            if cn_contents:
                wiki_zh_by_code[str(code)] = cn_contents
        logger.info("Loaded bilingual wiki for %d classes", len(wiki_zh_by_code))

    traces_accepted = 0
    traces_rejected = 0
    sft_en_accepted = 0
    sft_en_rejected = 0
    sft_zh_accepted = 0
    sft_zh_rejected = 0

    out_sft_en = args.output_sft_en.open("w", encoding="utf-8")
    out_sft_zh = args.output_sft_zh.open("w", encoding="utf-8")
    out_traces = args.output_traces.open("w", encoding="utf-8") if args.output_traces else None

    try:
        for key, row in payload.items():
            template_vars = row.get("template_vars", {})
            name_mapping_en = build_name_mapping(template_vars, lang="en")
            name_mapping_zh = build_name_mapping(template_vars, lang="zh")

            trace = extract_teacher_trace(key, row, name_mapping_en)
            if trace is None:
                traces_rejected += 1
                continue

            traces_accepted += 1
            if out_traces:
                out_traces.write(json.dumps(trace, ensure_ascii=False) + "\n")

            sft_en = convert_to_student_sft(trace, key, lang="en")
            if sft_en and _think_len(sft_en) >= args.min_think_len:
                out_sft_en.write(json.dumps(sft_en, ensure_ascii=False) + "\n")
                sft_en_accepted += 1
            else:
                sft_en_rejected += 1

            sft_zh = convert_to_student_sft(trace, key, lang="zh", name_mapping=name_mapping_zh, wiki_zh_by_code=wiki_zh_by_code)
            if sft_zh and _think_len(sft_zh) >= args.min_think_len:
                out_sft_zh.write(json.dumps(sft_zh, ensure_ascii=False) + "\n")
                sft_zh_accepted += 1
            else:
                sft_zh_rejected += 1
    finally:
        out_sft_en.close()
        out_sft_zh.close()
        if out_traces:
            out_traces.close()

    logger.info("Teacher traces: %d accepted / %d rejected", traces_accepted, traces_rejected)
    logger.info("Student SFT EN: %d accepted / %d rejected -> %s", sft_en_accepted, sft_en_rejected, args.output_sft_en)
    logger.info("Student SFT ZH: %d accepted / %d rejected -> %s", sft_zh_accepted, sft_zh_rejected, args.output_sft_zh)
    if args.output_traces:
        logger.info("Teacher traces written to %s", args.output_traces)


if __name__ == "__main__":
    main()
