import logging
import json
from typing import Any, Dict, List, Optional
import jinja2
from .basic_agent import BasicAgent
from ..tools import *
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register_agent("tool")
class ToolAgent(BasicAgent):
    """
    Agent capable of executing tools.
    Expects tool calls in JSON format: {"tool": "name", "args": {...}}
    """
    def __init__(self, config, template_root, llm_client, log_interval=10):
        super().__init__(config, template_root, llm_client, log_interval=log_interval)
        self.tools = ToolRegistry()
        
        # Load format configuration (configurable for data collection)
        tool_call_fmt = getattr(config, 'tool_call_format', None) or {}
        think_fmt = getattr(config, 'think_format', None) or {}
        
        # Default format tags (standard format)
        self.tool_call_open = tool_call_fmt.get('open', '<tool_call>')
        self.tool_call_close = tool_call_fmt.get('close', '</tool_call>')
        self.think_open = think_fmt.get('open', '[think]')
        self.think_close = think_fmt.get('close', '[/think]')
        
        # Setup custom chat template if configured
        self.custom_template = None
        custom_tmpl_str = getattr(config, 'custom_chat_template', None)
        if custom_tmpl_str:
            if self.jinja_env is None:
                self.jinja_env = jinja2.Environment()
            
            # Ensure 'tojson' filter is available
            if 'tojson' not in self.jinja_env.filters:
                self.jinja_env.filters['tojson'] = json.dumps
                
            try:
                self.custom_template = self.jinja_env.from_string(custom_tmpl_str)
                logger.info("Initialized custom chat template for ToolAgent")
            except Exception as e:
                logger.error(f"Failed to compile custom_chat_template: {e}")
        
        logger.debug(f"Format config: tool_call={self.tool_call_open}...{self.tool_call_close}, think={self.think_open}...{self.think_close}")
        
        # Load tools based on config
        requested_tools = getattr(config, 'tools', [])
        tool_config_map = getattr(config, 'tool_config', {})

        import importlib

        for tool_name in requested_tools:
            tool_cls = None
            
            # 1. Try to get tool class from global registry
            tool_cls = ToolRegistry.get_tool_class(tool_name)
            
            # 2. If not found, try dynamic import via class_path from config
            if not tool_cls:
                tool_cfg = tool_config_map.get(tool_name, {})
                class_path = tool_cfg.get("class_path")
                if class_path:
                    try:
                        module_path, class_name = class_path.rsplit(".", 1)
                        # Importing the module should trigger the @register_tool decorator
                        module = importlib.import_module(module_path)
                        
                        # Check registry again
                        tool_cls = ToolRegistry.get_tool_class(tool_name)
                        if tool_cls:
                            logger.info(f"Dynamically loaded tool '{tool_name}' from {class_path}")
                        else:
                            # Fallback: try to get class directly if decorator didn't work/exist
                            # and manually register it (though decorator is preferred)
                            tool_cls = getattr(module, class_name)
                            # We don't manually register to global registry here to avoid side effects,
                            # just use it for this instance.
                            logger.info(f"Loaded tool class directly from {class_path}")
                    except Exception as e:
                        logger.error(f"Failed to dynamically import tool '{tool_name}' from {class_path}: {e}")

            # 3. Instantiate and register instance to this agent
            if tool_cls:
                kwargs = tool_config_map.get(tool_name, {}).copy()
                kwargs.pop("class_path", None)
                
                try:
                    tool_instance = tool_cls(**kwargs)
                    self.register_tool(tool_instance)
                except Exception as e:
                    logger.error(f"Failed to instantiate tool {tool_name} with kwargs {kwargs}: {e}")
                    raise e
            else:
                logger.warning(f"Unknown tool requested: {tool_name}. Skipping auto-instantiation.")
        
    def register_tool(self, tool: BaseTool):
        self.tools.register(tool)

    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        # 1. Prepare Prompt
        image_path = str(item.img_path) if item.img_path else ""
        templates = self.config.templates or self.config.template_files or {}
        context = item.template_vars
        prompt = self._build_prompt(templates, context)

        should_log = self.lifecycle_logger.log_cycle_start(self.config.name, item.img_name)

        # 2. Initialize Conversation
        conv = self.create_conversation()
        
        # Build tools schema as JSON string for output
        tool_list = self.tools.list_tools()
        tools_json_str = json.dumps(tool_list, ensure_ascii=False)
        
        # Add available tools description to prompt
        # Only inject tools manually if NO custom template is used (template handles it)
        if not self.custom_template:
            tool_schemas = json.dumps(tool_list, indent=2)
            if tool_schemas != "[]":
                prompt = f"Available Tools:\n{tool_schemas}\n\n{prompt}"

        # Build standard messages array for output
        output_messages = []
        
        # User message with <image> placeholder
        user_content = f"<image>{prompt}"
        output_messages.append({"role": "user", "content": user_content})

        initial_msgs = await self.build_user_message(
            text=prompt,
            image_path=item.img_path,
            image_kwargs=kwargs
        )
        for msg in initial_msgs:
            conv.add_message(msg["role"], msg["content"])

        max_turns = 5
        current_turn = 0
        final_response = ""
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_history = []  # Track tool calls for output
        output_images = [image_path]  # Track all images (input + from tool responses)

        while current_turn < max_turns:
            current_turn += 1
            logger.debug(f"ToolAgent Turn {current_turn}")

            # Chat
            try:
                messages_to_send = conv.get_messages()
                self.lifecycle_logger.log_question(messages_to_send, should_log=should_log)
                
                # Apply custom chat template if configured
                if self.custom_template:
                    try:
                        # Render the FULL prompt including history
                        full_prompt = self.custom_template.render(
                            messages=messages_to_send, 
                            tools=tool_list,
                            add_vision_id=True,
                            add_generation_prompt=True
                        )
                        # Wrap rendered prompt in a single user message for transport
                        # (Assumes backend handles raw/pre-formatted string in user role or transparently)
                        messages_to_send = [{"role": "user", "content": full_prompt}]
                    except Exception as e:
                        logger.error(f"Error rendering custom chat template: {e}")
                
                llm_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ["min_pixels", "max_pixels", "max_image_size"]}

                response = await self.llm.chat(
                    model=model_name,
                    messages=messages_to_send,
                    **llm_kwargs
                )
            except Exception as e:
                logger.error(f"LLM error: {e}")
                break

            content = response.choices[0].message.content.strip()
            final_response = content
            
            # Track usage
            if response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens

            conv.add_message("assistant", content)
            
            self.lifecycle_logger.log_response(model_name, content, should_log=should_log)

            # Check for tool call
            tool_call = self._parse_tool_call(content)
            
            if tool_call:
                tool_name = tool_call.get("tool") or tool_call.get("name")
                tool_args = tool_call.get("args") or tool_call.get("arguments", {})
                logger.debug(f"Detected tool call: {tool_name}({tool_args})")
                
                # Extract thinking content before tool call (if any)
                think_content = self._extract_thinking(content)
                
                self.lifecycle_logger.log_tool_use(tool_name, tool_args, should_log=should_log)
                # Pass validation flag to tool (e.g. to disable noise)
                is_validating = kwargs.get('validate', False)
                if not is_validating and hasattr(item, 'meta_info'):
                     is_validating = item.meta_info.get('validate', False)
                
                result = await self.tools.execute_tool(tool_name, validate=is_validating, **tool_args)
                self.lifecycle_logger.log_tool_result(result, should_log=should_log)
                
                # Track for output
                tool_history.append({"tool": tool_name, "args": tool_args, "result": result})
                
                # Add tool_call to output messages (Modern Format with tool_calls)
                tool_args_str = json.dumps(tool_args, ensure_ascii=False)
                
                # Generate ID if not present
                call_id = tool_call.get("id", f"call_{len(tool_history)}")
                
                tool_call_obj = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args_str}
                }
                
                output_messages.append({
                    "role": "assistant",
                    "content": think_content if think_content else "",
                    "tool_calls": [tool_call_obj]
                })
                
                # Add tool_response to output messages (Modern Format: role=tool)
                tool_response_clean = {k: v for k, v in result.items() if k != "visualization"} if isinstance(result, dict) else result
                tool_response_content = json.dumps(tool_response_clean, ensure_ascii=False)
                
                output_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_response_content
                })
                
                # If tool response contains visualization, add it to images list
                if isinstance(result, dict) and result.get("visualization"):
                    output_images.append(result["visualization"])
                
                # Build feedback message (potentially multimodal)
                feedback_content = await self._build_tool_feedback(result, kwargs)
                conv.add_message("user", feedback_content)
            else:
                # No tool call, add final assistant response
                output_messages.append({"role": "assistant", "content": content})
                break

        # Get image path for output (already defined at start)
        image_name = context.get("image_name", f"{item.img_name}.jpg")
        
        # Build result in standard message format
        result_dict = {
            item.img_name: {
                "tools": tools_json_str,
                "messages": output_messages,
                "images": output_images,
                "metadata": {
                    "image_name": image_name,
                    "image_id": context.get("image_id"),
                    "task_type": self.config.name,
                    "template_vars": context,
                    "result": final_response,
                    "raw_response": final_response,
                    "usage": total_usage,
                    "tool_history": tool_history
                }
            }
        }
        self.lifecycle_logger.log_cycle_end(item.img_name, should_log=should_log)
        return result_dict

    async def _build_tool_feedback(self, result: Any, image_kwargs: Dict) -> Any:
        """
        Build feedback content for tool result.
        If result contains 'visualization' path, returns multimodal content (text + image).
        Otherwise, returns text-only content.
        """
        if isinstance(result, dict) and result.get("visualization"):
            vis_path = result["visualization"]
            # Convert visualization to base64 data URL
            from ..core.image_processor import image_to_data_url
            image_url = await image_to_data_url(vis_path)
            
            # Construct multimodal content: image + text
            text_part = f"Tool Output:\n```json\n{json.dumps(result, indent=2)}\n```"
            
            if image_url:
                return [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text_part}
                ]
            else:
                return f"Tool Output: {result}"
        else:
            # Plain text feedback
            return f"Tool Output: {result}"

    def _build_prompt(self, templates, context):
        """Helper to build prompt string from templates."""
        # Reuse logic from BasicAgent but extracted
        rendered_parts = {}
        for key, rel_path in templates.items():
            rendered_parts[key] = self.render_template(rel_path, context)
            
        if "full_prompt" in rendered_parts:
            return rendered_parts["full_prompt"]
        elif "usr" in rendered_parts:
             return rendered_parts["usr"]
        else:
            return "\n\n".join(rendered_parts.values())

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """Try to parse tool call from text. Supports configurable formats:
        - Configured format (default: <tool_call>...json...</tool_call>)
        - Legacy format [tool_call]...[/tool_call]
        - Fallback: Direct JSON
        """
        import re
        
        def try_parse_json(text_block: str) -> Optional[Dict]:
            """Helper to parse JSON and validate tool call structure."""
            # Clean markdown code blocks
            clean = text_block.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(clean)
                if isinstance(data, dict):
                    if "tool" in data or "name" in data:
                        return data
                    elif "function" in data:
                        func = data["function"]
                        return {"name": func.get("name"), "args": func.get("arguments")}
            except json.JSONDecodeError:
                pass
            return None
        
        # 1. Try configured format (e.g., <tool_call>...</tool_call> or [tool_call]...[/tool_call])
        open_tag = re.escape(self.tool_call_open)
        close_tag = re.escape(self.tool_call_close)
        pattern = f'{open_tag}(.*?){close_tag}'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            result = try_parse_json(match.group(1))
            if result:
                return result
        
        # 2. Try standard <tool_call> if different from configured
        if self.tool_call_open != '<tool_call>':
            std_match = re.search(r'<tool_call>(.*?)</tool_call>', text, flags=re.DOTALL)
            if std_match:
                result = try_parse_json(std_match.group(1))
                if result:
                    return result
        
        # 3. Try legacy [tool_call] format if different from configured
        if self.tool_call_open != '[tool_call]':
            legacy_match = re.search(r'\[tool_call\](.*?)\[/tool_call\]', text, flags=re.DOTALL)
            if legacy_match:
                result = try_parse_json(legacy_match.group(1))
                if result:
                    return result
        
        # 4. Fallback: Remove thinking blocks and try to parse JSON directly
        # Remove configured think format
        think_open = re.escape(self.think_open)
        think_close = re.escape(self.think_close)
        text_clean = re.sub(f'{think_open}.*?{think_close}', '', text, flags=re.DOTALL).strip()
        # Also remove standard formats
        text_clean = re.sub(r'\[think\].*?\[/think\]', '', text_clean, flags=re.DOTALL).strip()
        text_clean = re.sub(r'<think>.*?</think>', '', text_clean, flags=re.DOTALL).strip()
        
        # Try direct JSON parse
        result = try_parse_json(text_clean)
        if result:
            return result
        
        # Try to extract JSON object from text
        json_match = re.search(r'\{[^{}]*\}', text_clean)
        if json_match:
            result = try_parse_json(json_match.group())
            if result:
                return result
        
        return None

    def _extract_thinking(self, text: str) -> Optional[str]:
        """Extract thinking content from response using configured format.
        Uses configurable format (default: [think]...[/think]) to avoid conflicts with Qwen3-VL's native thinking.
        Does NOT include any tool_call content - that goes to separate tool_call message.
        """
        import re
        
        # Try configured format first
        open_tag = re.escape(self.think_open)
        close_tag = re.escape(self.think_close)
        pattern = f'{open_tag}(.*?){close_tag}'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return f"{self.think_open}\n{match.group(1).strip()}\n{self.think_close}"
        
        # Try legacy [think] format if different from configured
        if self.think_open != '[think]':
            legacy_match = re.search(r'\[think\](.*?)\[/think\]', text, flags=re.DOTALL)
            if legacy_match:
                return f"{self.think_open}\n{legacy_match.group(1).strip()}\n{self.think_close}"
        
        return None


