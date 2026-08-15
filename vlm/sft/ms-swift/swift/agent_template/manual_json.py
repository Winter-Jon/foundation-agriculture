# Copyright (c) ModelScope Contributors. All rights reserved.
"""Strict manual-JSON tool-call formatting for protocol-facing SFT.

The evaluator expects the first assistant turn to be a bare JSON object, rather
than a model/vendor-specific tool marker.  This adapter keeps that wire format
in the supervised target while retaining the normal tool-response ordering.
"""

import json
import re
from typing import List, Optional, Union

from .base import BaseAgentTemplate


class ManualJSONAgentTemplate(BaseAgentTemplate):
    """Format function calls as ``{"name": ..., "arguments": ...}``."""

    def _format_tool_calls(self, tool_call_messages) -> str:
        calls = []
        for message in tool_call_messages:
            call = self._parse_tool_call(message['content'])
            calls.append({
                'name': call['name'],
                'arguments': call['arguments'],
            })
        # One call is the protocol contract. Keep deterministic compact JSON;
        # support multiple calls without adding non-protocol marker text.
        if len(calls) == 1:
            return json.dumps(calls[0], ensure_ascii=False, separators=(',', ':'))
        return json.dumps(calls, ensure_ascii=False, separators=(',', ':'))

    def _format_tools(
        self,
        tools: List[Union[str, dict]],
        system: Optional[str] = None,
        user_message=None,
    ) -> str:
        # Keep tool instructions language-neutral: the dataset already carries
        # language-specific rationale/final-answer constraints.
        names = [self._parse_tool(tool, 'en').name_for_model for tool in tools]
        user_text = str((user_message or {}).get('content') or '')
        chinese = bool(re.search(r'[\u4e00-\u9fff]', user_text))
        language_contract = (
            ' This is a Chinese query. Except for the internal English retrieval query, all visible reasoning, '
            'field labels, and the final answer must be in Chinese. Use the headings 证据、排除的候选、不确定性.'
            if chinese else
            ' This is an English query. All visible reasoning and the final answer must be in English; do not use Chinese.'
        )
        suffix = (
            '\n\nYou are an agricultural visual recognition assistant using manual JSON tool-calling. '
            'Every assistant message must be exactly one bare JSON tool-call object or a final '
            '<think>...</think> followed by <answer>...</answer>. The first assistant message '
            'must be a single retrieval JSON call; never mix a call with explanation. '
            'Use this exact shape: {"name":"agrinet_rag_search","arguments":{"query":"...",'
            '"retrieval_type":"visual","image":"query_image","top_k":3,'
            '"rationale":"Visual Observation: ...; Candidate Analysis: ..."}}. '
            f'The tool name must be one of: {",".join(names)}. '
            'After a tool response, emit another bare JSON call or the grounded final response.'
            + language_contract
        )
        return (system or '') + suffix
