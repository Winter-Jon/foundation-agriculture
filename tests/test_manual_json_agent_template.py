import json

from swift.agent_template.manual_json import ManualJSONAgentTemplate
from swift.agent_template.hermes import HermesAgentTemplate
from swift.template.base import Template
from swift.template.template_inputs import StdTemplateInputs
from swift.loss_scale.manual_json import ManualJSONLossScale


def test_manual_json_tool_call_is_bare_json():
    template = ManualJSONAgentTemplate()
    content = template._format_tool_calls([{
        'content': json.dumps({
            'name': 'agrinet_rag_search',
            'arguments': {'query': 'leaf spot', 'top_k': 5},
        })
    }])
    parsed = json.loads(content)
    assert parsed == {
        'name': 'agrinet_rag_search',
        'arguments': {'query': 'leaf spot', 'top_k': 5},
    }
    assert '<tool_call>' not in content
    assert '✿FUNCTION✿' not in content
    assert 'Action:' not in content


def test_manual_json_is_required_for_the_native_json_sft_wire_contract():
    """Hermes rewrites role=tool_call to XML; native evaluation must not train it."""
    message = {'content': json.dumps({
        'name': 'agrinet_rag_search',
        'arguments': {'query': 'leaf spot', 'top_k': 3},
    })}
    hermes_content = HermesAgentTemplate()._format_tool_calls([message])
    native_content = ManualJSONAgentTemplate()._format_tool_calls([message])
    assert '<tool_call>' in hermes_content
    assert json.loads(native_content)['name'] == 'agrinet_rag_search'
    assert '<tool_call>' not in native_content


def test_manual_json_preserves_tool_response_order():
    template = ManualJSONAgentTemplate()
    assistant, response = template._format_tool_responses(
        '{"name":"agrinet_rag_search","arguments":{"query":"x"}}',
        [{'content': '{"results":[1,2]}'}],
    )
    assert json.loads(assistant)['name'] == 'agrinet_rag_search'
    assert response == ['{"results":[1,2]}']


def test_manual_json_preprocess_converts_only_adjacent_tool_call():
    template = Template.__new__(Template)
    template._agent_template = 'manual_json'
    template._agent_template_cache = {}
    template.template_meta = None
    inputs = StdTemplateInputs(messages=[
        {'role': 'user', 'content': 'query'},
        {'role': 'assistant', 'content': '{"name":"agrinet_rag_search","arguments":{"query":"x"}}'},
        {'role': 'tool', 'content': '{"status":"success"}'},
        {'role': 'assistant', 'content': 'final answer'},
    ])
    template._preprocess_function_call(inputs)
    assert inputs.messages[1]['role'] == 'assistant'
    assert inputs.messages[1]['content'] == '{"name":"agrinet_rag_search","arguments":{"query":"x"}}'
    assert inputs.messages[2]['role'] == 'tool'
    assert inputs.messages[3]['role'] == 'assistant'


def test_manual_json_tool_call_does_not_receive_thinking_prefix():
    template = Template.__new__(Template)
    template._agent_template = 'manual_json'
    template.mode = 'train'
    template._loss_scale = 'default'
    template._loss_scale_cache = {}
    template.template_meta = type('Meta', (), {
        'non_thinking_prefix': '<think>\\n\\n</think>\\n\\n',
    })()
    inputs = StdTemplateInputs(messages=[
        {'role': 'assistant', 'content': '{"name":"agrinet_rag_search","arguments":{}}'},
        {'role': 'tool', 'content': '{}'},
        {'role': 'assistant', 'content': 'final'},
    ])
    template._add_non_thinking_prefix(inputs)
    assert inputs.messages[0]['content'].startswith('{"name":"agrinet_rag_search"')
    assert inputs.messages[2]['content'].startswith('<think>')


def test_manual_json_loss_scale_weights_only_tool_json():
    scale = ManualJSONLossScale()
    values, weights = scale.get_loss_scale(
        '{"name":"agrinet_rag_search","arguments":{"query":"x"}}')
    assert values[0].startswith('{"name":"agrinet_rag_search"')
    assert weights == [3.0]
    _, final_weights = scale.get_loss_scale('<think>Evidence</think><answer>class</answer>')
    assert final_weights == [1.0]


def test_manual_json_language_contract_follows_user_query():
    template = ManualJSONAgentTemplate()
    text = template._format_tools([], user_message={'content': '请识别图中的病虫害'})
    assert 'all visible reasoning' in text
    assert '证据、排除的候选、不确定性' in text
