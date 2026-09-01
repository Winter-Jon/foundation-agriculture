"""Shared, label-blind terminal-answer contract for HCV SFT and evaluation.

This text is deliberately centralized: terminal-rejection examples are useful
only if their observation exactly matches the deployed evaluator state.
"""
from __future__ import annotations

from typing import Any


def final_answer_only_correction(sample: dict[str, Any]) -> str:
    """Return the public instruction after no further retrieval is available."""
    if sample.get("question_type") == "option":
        return (
            "Tool budget exhausted: do not call any more tools. Give the final answer now. "
            "Use the trained final form <think>brief evidence</think><answer>A</answer>: the <answer> must contain exactly one option letter A, B, C, or D. Do not output tool_call."
            if sample.get("language") != "zh" else
            "工具调用次数已用尽：不要再调用工具。现在使用训练时的最终格式 <think>简短证据</think><answer>A</answer>；<answer> 中只能输出唯一选项字母 A、B、C 或 D，不要输出 tool_call。"
        )
    return (
        "Tool budget exhausted: do not call any more tools. Give the final answer now. "
        "Use the trained final form <think>brief evidence</think><answer>class name</answer>, copying only a class name or alias from retrieved evidence; no tool_call."
        if sample.get("language") != "zh" else
        "工具调用次数已用尽：不要再调用工具。现在使用训练时的最终格式 <think>简短证据</think><answer>类别名</answer>，<answer> 只能使用检索证据中的类别名或别名；不要输出 tool_call。"
    )
