"""Swift-compatible Hermes Agent data and wire-format helpers."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

TOOL_CALL_RE = re.compile(r"^\s*<tool_call>\s*(\{.*?\})\s*</tool_call>\s*$", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE)
PRE_TOOL_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*$", re.DOTALL | re.IGNORECASE)


def json_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def tools_json(value: Any) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(tool, dict) for tool in parsed):
        raise ValueError("tools must be a JSON list of objects")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def parse_hermes_tool_calls(content: Any, *, tool_name: str, validate_arguments: Callable[[dict[str, Any]], list[str]]) -> list[dict[str, Any]]:
    if not isinstance(content, str):
        return []
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
    if not matches:
        return []
    residue = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", content, flags=re.DOTALL).strip()
    if residue:
        return []
    calls = []
    for raw in matches:
        try:
            call = json_object(raw, field="tool_call")
        except ValueError:
            return []
        if call.get("name") != tool_name or not isinstance(call.get("arguments"), dict):
            return []
        if validate_arguments(call["arguments"]):
            return []
        calls.append({"name": tool_name, "arguments": call["arguments"]})
    return calls


def canonicalize_hermes_tool_turn(
    content: Any,
    *,
    tool_name: str,
    validate_arguments: Callable[[dict[str, Any]], list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    """Return one Swift-Hermes-compatible assistant tool turn.

    SGLang may emit the SFT planning turn and the following XML tool call in a
    single completion, and it preserves its own JSON whitespace.  ms-swift
    instead turns the native ``tool_call`` record into a newline-wrapped XML
    block with ``json.dumps(..., ensure_ascii=False)``.  Canonicalize that
    block before it is fed back as assistant history, while preserving the
    model-generated planning text verbatim.
    """
    if not isinstance(content, str):
        return "", []
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL))
    if len(matches) != 1 or content[matches[0].end():].strip():
        return "", []
    planning = content[:matches[0].start()]
    if planning.strip() and not is_pre_tool_think(planning):
        return "", []
    raw_call = matches[0].group(1)
    calls = parse_hermes_tool_calls(
        f"<tool_call>{raw_call}</tool_call>", tool_name=tool_name, validate_arguments=validate_arguments)
    if len(calls) != 1:
        return "", []
    canonical_call = json.dumps(calls[0], ensure_ascii=False)
    return f"{planning}<tool_call>\n{canonical_call}\n</tool_call>", calls


def is_pre_tool_think(content: Any) -> bool:
    return isinstance(content, str) and bool(PRE_TOOL_THINK_RE.fullmatch(content))


def is_final_answer(content: Any) -> bool:
    return isinstance(content, str) and bool(FINAL_ANSWER_RE.fullmatch(content))


def normalize_training_messages(messages: Any, *, tool_name: str, validate_arguments: Callable[[dict[str, Any]], list[str]], require_tool_calls: bool) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    pending_call = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("messages must contain objects")
        role, content = str(message.get("role") or ""), message.get("content")
        if role == "tool_response":
            role = "tool"
        if role == "tool_call":
            call = json_object(content, field="tool_call")
            if call.get("name") != tool_name or not isinstance(call.get("arguments"), dict):
                raise ValueError("tool_call has invalid name or arguments")
            errors = validate_arguments(call["arguments"])
            if errors:
                raise ValueError(f"tool_call arguments invalid: {', '.join(errors)}")
            content = json.dumps({"name": tool_name, "arguments": call["arguments"]}, ensure_ascii=False, separators=(",", ":"))
            pending_call = True
        elif role == "tool":
            json_object(content, field="tool response")
            if not pending_call:
                raise ValueError("tool response must immediately follow a tool call")
            pending_call = False
        elif role == "assistant":
            if pending_call:
                raise ValueError("tool call must immediately precede its tool response")
            if index != len(messages) - 1 and not is_pre_tool_think(content):
                raise ValueError("non-final assistant message must be a pure <think> planning turn")
            if index == len(messages) - 1 and not is_final_answer(content):
                raise ValueError("final assistant message must be <think> followed by <answer>")
        elif role not in {"user", "system"}:
            raise ValueError(f"unsupported message role: {role}")
        normalized.append({"role": role, "content": content})
    if pending_call:
        raise ValueError("tool call has no tool response")
    first_non_system = next((message for message in normalized if message["role"] != "system"), None)
    if first_non_system is None or first_non_system.get("role") != "user":
        raise ValueError("first non-system message must be user")
    if require_tool_calls and not any(message["role"] == "tool_call" for message in normalized):
        raise ValueError("RAG trajectory requires at least one tool_call")
    return normalized


def swift_hermes_system_prompt(base_system: str, tools: Any) -> str:
    """Render the current ms-swift ``HermesAgentTemplate`` tools prompt.

    Keep this helper byte-compatible with
    ``swift.agent_template.hermes.HermesAgentTemplate._format_tools`` so the
    evaluator does not hand-author a near-but-different tools context.  In
    particular, Hermes serializes each tool object on its own line rather than
    placing the tool list in one JSON array.
    """
    if isinstance(tools, str):
        parsed = json.loads(tools)
    else:
        parsed = tools
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(tool, dict) for tool in parsed):
        raise ValueError("tools must be a JSON list of objects")
    # BaseAgentTemplate.wrap_tool: preserve an already OpenAI-wrapped function
    # schema, otherwise add the same wrapper used by ms-swift.
    wrapped = [tool if "type" in tool and "function" in tool else {"type": "function", "function": tool} for tool in parsed]
    tool_descs = [json.dumps(tool, ensure_ascii=False) for tool in wrapped]
    return (
        f"{base_system or ''}\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        + "\n".join(tool_descs)
        + "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    )


def swift_hermes_tool_response_message(content: Any) -> dict[str, str]:
    """Render one retrieval result as the ms-swift Hermes user turn."""
    serialized = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return {"role": "user", "content": f"<tool_response>\n{serialized}\n</tool_response>"}


def hermes_system_prompt(base_system: str, tools: Any) -> str:
    """Backward-compatible name for the ms-swift-compatible renderer."""
    return swift_hermes_system_prompt(base_system, tools)
