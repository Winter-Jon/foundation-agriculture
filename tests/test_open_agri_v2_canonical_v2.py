from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/data/build_open_agri_v2_canonical_v2.py"
SPEC = importlib.util.spec_from_file_location("canonical_v2_builder", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
AUDIT_SCRIPT = ROOT / "scripts/data/audit_open_agri_v2_canonical_v2.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("canonical_v2_audit", AUDIT_SCRIPT)
assert AUDIT_SPEC and AUDIT_SPEC.loader
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)


class Registry:
    digest = "d" * 64

    def canonical_for_source(self, value: str) -> str:
        return {"N00002": "N00001"}.get(value, value)

    def display_name(self, code: str, language: str) -> str:
        return {("N00001", "en"): "canonical leaf spot", ("N00001", "zh"): "规范叶斑病"}[code, language]

    def resolve_answer(self, value: object) -> str | None:
        return "N00001" if str(value).lower().replace("_", " ") in {"legacy leaf spot", "canonical leaf spot", "规范叶斑病"} else None


def direct(*, question_type: str = "open") -> dict:
    user = "<image> identify" if question_type == "open" else "<image> choose\nA. legacy leaf spot\nB. legacy leaf spot\nC. legacy leaf spot\nD. legacy leaf spot"
    return {
        "sample_id": "direct", "images": ["image.jpg"],
        "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": user}, {"role": "assistant", "content": "<think>x</think><answer>legacy leaf spot</answer>" if question_type == "open" else "<think>x</think><answer>A</answer>"}],
        "metadata": {"class_code": "N00002", "v2_image_sha256": "a" * 64, "route": "direct", "language": "en", "question_type": question_type},
    }


def rag(calls: int) -> dict:
    messages = [{"role": "user", "content": "<image> identify"}]
    for index in range(calls):
        messages.extend((
            {"role": "assistant", "content": "<think>plan</think>"},
            {"role": "tool_call", "content": json.dumps({"name": "agrinet_rag_search", "arguments": {"query": "leaf", "retrieval_type": "visual", "image": "query_image", "top_k": 1, "rationale": "x"}})},
            {"role": "tool_response", "content": json.dumps({"results": [{"name": "legacy leaf spot", "name_zh": "规范叶斑病"}]} if index < calls - 1 else {"error": "tool_budget_exhausted", "results": []})},
        ))
    messages.append({"role": "assistant", "content": "<think>x</think><answer>legacy leaf spot</answer>"})
    return {"sample_id": f"rag-{calls}", "images": ["image.jpg"], "tools": "[]", "messages": messages,
            "metadata": {"class_code": "N00002", "v2_image_sha256": "b" * 64, "route": "rag", "language": "en", "question_type": "open"}}


def test_convert_rewrites_open_and_option_labels(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    builder.write_jsonl(path, [direct(), direct(question_type="option")])
    open_row, audit = builder.convert_record(direct(), registry=Registry(), source_path=path, source_line=1, source_kind="test")
    assert audit["action"] == "include"
    assert "<answer>canonical leaf spot</answer>" in open_row["messages"][-1]["content"]
    option_row, _ = builder.convert_record(direct(question_type="option"), registry=Registry(), source_path=path, source_line=2, source_kind="test")
    assert "A. canonical leaf spot" in option_row["messages"][1]["content"]
    assert option_row["messages"][-1]["content"].endswith("<answer>A</answer>")


def test_short_budget_terminal_isolated_and_five_turn_retained(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    builder.write_jsonl(path, [rag(2), rag(6)])
    short, short_audit = builder.convert_record(rag(2), registry=Registry(), source_path=path, source_line=1, source_kind="test")
    kept, kept_audit = builder.convert_record(rag(6), registry=Registry(), source_path=path, source_line=2, source_kind="test")
    assert short is None and short_audit["action"] == "isolate"
    assert kept is not None and kept_audit["action"] == "include"
    assert [message["role"] for message in kept["messages"]].count("tool") == 6


def test_supplement_requires_complete_views() -> None:
    rows = []
    for route in ("direct", "rag"):
        for question in ("open", "option"):
            for language in ("en", "zh"):
                row = direct(question_type=question) if route == "direct" else rag(1)
                row["sample_id"] = f"{route}-{question}-{language}"
                row["metadata"].update({"image_sha256": "c" * 64, "route": route, "language": language, "question_type": question})
                rows.append(row)
    accepted = builder.complete_supplement_rows(rows, [], selected_normal={"c" * 64}, selected_oracle=set())
    assert len(accepted) == 8
    assert not builder.complete_supplement_rows(rows[:-1], [], selected_normal={"c" * 64}, selected_oracle=set())


def test_class_coverage_separates_image_and_route_gaps() -> None:
    known = {
        "N00001": {"canonical_english_name": "one", "canonical_chinese_name": "一", "domain": "disease"},
        "N00002": {"canonical_english_name": "two", "canonical_chinese_name": "二", "domain": "pest"},
    }
    rows = [
        {"metadata": {"canonical_class_code": "N00001", "v2_image_sha256": image, "route": "direct", "question_type": "open", "language": "en"}}
        for image in ("a", "b", "c", "d")
    ]
    rows.extend(
        {"metadata": {"canonical_class_code": "N00002", "v2_image_sha256": image, "route": "rag", "question_type": "open", "language": "en"}}
        for image in ("e", "f", "g")
    )
    coverage, summary = audit.class_coverage(rows, known, image_floor=4, route_row_floor=4)
    by_code = {item["canonical_class_code"]: item for item in coverage}
    assert by_code["N00001"]["unique_training_images"] == 4
    assert by_code["N00001"]["rag_route_gap"]
    assert by_code["N00002"]["image_floor_gap"]
    assert by_code["N00002"]["direct_route_gap"]
    assert by_code["N00002"]["rag_row_floor_gap"]
    assert summary["image_floor_gaps"] == ["N00002"]
