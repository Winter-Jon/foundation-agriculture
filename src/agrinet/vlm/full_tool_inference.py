"""Public full-tool state machine and faithful Hermes multimodal wire format."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from agrinet.rag.dual_teacher import SAMPLER_GUIDANCE, reference_attachments
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.hermes_protocol import swift_hermes_system_prompt
from agrinet.rag.protocol_repair import repair_instruction
from agrinet.vlm.full_tool import (SYSTEM_PROMPT, CLASSIFIER_PREDICT, RAG_SEARCH,
                                   tool_schemas, validate_arguments, validate_messages)

OBSERVE = 'Observation stage only. In this turn describe visible morphology, affected organ and what is not visible in 2-4 sentences. Do not name a disease, choose an option, emit answer tags, or call a tool. A separate next turn will request classification and retrieval.'
CONTINUE = 'Observation stage complete. Now call agrinet_classifier_predict, then perform 1-3 RAG searches. Do not give a final answer before reading their results.'
FINALIZE = 'All three permitted RAG searches have returned. No further tool calls are available. Now give the final response using only the observations and evidence already received: <think>your substantive analysis</think><answer>one canonical English name or INSUFFICIENT_EVIDENCE</answer>.'


def wire_messages(messages):
    """Merge attached references into the corresponding tool response before wrapping."""
    merged = []
    for original in messages:
        m = copy.deepcopy(original)
        if m['role'] == 'user' and merged and merged[-1]['role'] in {'user', 'tool'}:
            previous = merged[-1]
            def parts(value):
                return value if isinstance(value, list) else [{'type': 'text', 'text': value}]
            previous['content'] = parts(previous['content']) + [{'type': 'text', 'text': '\n'}] + parts(m['content'])
        else:
            merged.append(m)
    result = []
    for m in merged:
        if m['role'] == 'system':
            m['content'] = swift_hermes_system_prompt(m['content'], tool_schemas())
        elif m['role'] == 'assistant' and m.get('tool_calls'):
            content = m.get('content') or ''
            for call in m.pop('tool_calls'):
                fn = call['function']
                value = {'name': fn['name'], 'arguments': json.loads(fn['arguments'])}
                content += '<tool_call>\n' + json.dumps(value, ensure_ascii=False) + '\n</tool_call>'
            m['content'] = content
        elif m['role'] == 'tool':
            content = m['content']
            if isinstance(content, list):
                content = [{'type': 'text', 'text': '<tool_response>\n'}] + content + [{'type': 'text', 'text': '\n</tool_response>'}]
            else:
                content = '<tool_response>\n' + content + '\n</tool_response>'
            m = {'role': 'user', 'content': content}
        result.append(m)
    return result


def parse_completion(text, ordinal):
    matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.S))
    if not matches:
        if '<tool_call' in text:
            raise ValueError('malformed_tool_call')
        return {'role': 'assistant', 'content': text}
    if len(matches) != 1 or text[matches[0].end():].strip():
        raise ValueError('exactly_one_call_per_turn_required')
    value = json.loads(matches[0].group(1))
    errors = validate_arguments(value.get('name'), value.get('arguments'))
    if errors:
        raise ValueError(str(errors))
    return {'role': 'assistant', 'content': text[:matches[0].start()], 'tool_calls': [{
        'id': f'call_{ordinal}', 'type': 'function', 'function': {
            'name': value['name'], 'arguments': json.dumps(value['arguments'], ensure_ascii=False)}}]}


def evaluate(row, *, generate, classifier, retrieve, image_root='datasets/AgriNet-1K/wiki/images'):
    image, _ = transport_image(Path(row['image_path']), max_side=1024)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT + '\n' + SAMPLER_GUIDANCE},
                {'role': 'user', 'content': [{'type': 'text', 'text': row['question']}, image]},
                {'role': 'user', 'content': OBSERVE}]
    observed = False
    predicted = False
    searches = 0
    repaired = False
    seen = {}
    history, modes, refs = [], [], []
    outputs = []
    for ordinal in range(9):
        text = generate(wire_messages(messages))
        outputs.append({'turn': ordinal, 'content': text})
        calls = []
        try:
            message = parse_completion(text, ordinal)
            messages.append(message)
            calls = message.get('tool_calls', [])
            if not observed:
                if calls or len(text.strip()) < 40 or '<answer>' in text:
                    raise ValueError('invalid_observation_stage')
                observed = True
                messages.append({'role': 'user', 'content': CONTINUE})
                continue
            if calls:
                fn = calls[0]['function']
                name, arguments = fn['name'], json.loads(fn['arguments'])
                if (name == CLASSIFIER_PREDICT and predicted) or (name == RAG_SEARCH and (not predicted or searches >= 3)):
                    raise ValueError('tool_order_or_count')
                response = classifier() if name == CLASSIFIER_PREDICT else retrieve(arguments)
                messages.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps(response, ensure_ascii=False)})
                history.append(name)
                predicted |= name == CLASSIFIER_PREDICT
                if name == RAG_SEARCH:
                    searches += 1
                    modes.append(arguments['retrieval_type'])
                    attachment, records = reference_attachments(response, image_root, seen, row['image_sha256'], row['image_path'])
                    if attachment:
                        messages.append(attachment)
                    refs.extend(records)
                    if searches == 3:
                        # Public equivalent of collection's tool_choice=none.
                        messages.append({'role': 'user', 'content': FINALIZE})
                continue
            validation = validate_messages(messages)
            return {'prediction': text, 'protocol_error': None, 'tool_history': history,
                    'rag_calls': searches, 'retrieval_modes': modes, 'reference_records': refs,
                    'protocol_repairs': int(repaired), 'insufficient_evidence': validation['insufficient_evidence'],
                    'completion_trace': outputs}
        except (ValueError, KeyError, TypeError) as exc:
            if repaired:
                return {'prediction': '', 'protocol_error': str(exc), 'tool_history': history,
                        'rag_calls': searches, 'retrieval_modes': modes, 'protocol_repairs': 1, 'completion_trace': outputs}
            if not messages or messages[-1].get('role') != 'assistant':
                messages.append({'role': 'assistant', 'content': text})
            for call in calls:
                messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                 'content': json.dumps({'protocol_error': [str(exc)]})})
            messages.append({'role': 'user', 'content': repair_instruction([str(exc)])})
            repaired = True
    return {'prediction': '', 'protocol_error': 'turn_loop_exhausted', 'tool_history': history, 'rag_calls': searches, 'completion_trace': outputs}
