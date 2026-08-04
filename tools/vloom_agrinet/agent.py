import json
import re
import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from vloom.agents.basic_agent import BasicAgent
from vloom.agents.registry import AgentRegistry


@AgentRegistry.register_agent("optional_image_basic")
class OptionalImageBasicAgent(BasicAgent):
    """Basic VLooM agent that also supports text-only dry-run records."""

    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        self._extra_image_paths = item.template_vars.get("image_paths", [])
        try:
            result = await super().run(item, model_name, **kwargs)
            self._normalize_result_shape(result)
            self._repair_markdown_json(result)
            return result
        finally:
            self._extra_image_paths = []

    @staticmethod
    def _normalize_result_shape(result: Dict[str, Any]) -> None:
        """Expose stable project fields from the vendored BasicAgent metadata envelope."""
        for row in result.values():
            if not row:
                continue
            metadata = row.get("metadata") or {}
            mappings = {
                "result": "result",
                "raw_response": "raw_response",
                "template_vars": "template_vars",
                "thinking": "think",
            }
            for metadata_key, public_key in mappings.items():
                if public_key not in row and metadata_key in metadata:
                    row[public_key] = metadata[metadata_key]

    async def build_user_message(
        self,
        text: str,
        image_path: Optional[Path] = None,
        image_kwargs: Dict = {},
    ):
        content = [{"type": "text", "text": text}]
        final_img_kwargs = {**(self.config.image_config or {}), **(image_kwargs or {})}

        if image_path:
            image_url = self._sync_image_to_data_url(Path(image_path), final_img_kwargs.get("max_image_size", 1024))
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        for extra_path in getattr(self, "_extra_image_paths", []):
            if not extra_path:
                continue
            image_url = self._sync_image_to_data_url(Path(extra_path), final_img_kwargs.get("max_image_size", 1024))
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        message = {"role": "user", "content": content}
        for param in ["min_pixels", "max_pixels"]:
            if param in final_img_kwargs:
                message[param] = final_img_kwargs[param]
        return [message]

    @staticmethod
    def _sync_image_to_data_url(image_path: Path, max_size: Optional[int] = 1024) -> Optional[str]:
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=95)
            encoded = base64.b64encode(output.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    @staticmethod
    def _repair_markdown_json(result: Dict[str, Any]) -> None:
        for row in result.values():
            if not row or row.get("result") is not None:
                continue
            raw = row.get("raw_response")
            if not raw:
                continue
            parsed, think_text, repair_log = OptionalImageBasicAgent._parse_jsonish(raw)
            if parsed is not None:
                row["result"] = parsed
                if think_text:
                    row["think"] = think_text
                if repair_log:
                    row.setdefault("repair_log", []).extend(repair_log)

    @staticmethod
    def _extract_think(text: str) -> tuple[str, str]:
        """Extract <think>...</think> content and remaining text."""
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            think_text = think_match.group(1).strip()
            remaining = text[:think_match.start()] + text[think_match.end():]
            return think_text, remaining.strip()
        return "", text

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Apply common repairs to JSON-like text."""
        # Remove markdown fences
        text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove leading/trailing whitespace and common prefixes
        text = text.strip()
        text = re.sub(r"^(?:Here is the JSON[:：]?\s*|JSON[:：]?\s*|Response[:：]?\s*)", "", text, flags=re.IGNORECASE)
        # Remove line comments //...
        text = re.sub(r"//[^\n]*", "", text)
        # Remove trailing commas before } or ]
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        # Replace single-quoted string delimiters with double quotes (simple heuristic)
        # Only do this for keys and simple values, not inside already-double-quoted strings
        # A safer approach: replace 'key': or 'value' patterns when not inside double quotes
        def _fix_quotes(m):
            s = m.group(0)
            # If already wrapped in double quotes, leave as-is
            if s.startswith('"') and s.endswith('"'):
                return s
            # Replace surrounding single quotes with double
            if s.startswith("'") and s.endswith("'"):
                inner = s[1:-1]
                # Escape any existing double quotes inside
                inner = inner.replace('"', '\\"')
                return f'"{inner}"'
            return s
        # Match JSON string literals: either double-quoted or single-quoted
        text = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', _fix_quotes, text)
        return text.strip()

    @staticmethod
    def _parse_jsonish(text: str) -> tuple[Optional[Dict[str, Any]], str, list[str]]:
        """Parse JSON from LLM response, extracting think text and applying repairs.

        Returns: (parsed_dict, think_text, repair_log)
        """
        think_text, remaining = OptionalImageBasicAgent._extract_think(text)
        repair_log: list[str] = []

        candidates = [remaining.strip()]
        # Extract brace-wrapped content as fallback
        brace_match = re.search(r"\{.*\}", remaining, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        # Also try cleaned version of remaining and brace match
        cleaned_remaining = OptionalImageBasicAgent._clean_json_text(remaining)
        if cleaned_remaining not in candidates:
            candidates.append(cleaned_remaining)
        if brace_match:
            cleaned_brace = OptionalImageBasicAgent._clean_json_text(brace_match.group(0))
            if cleaned_brace not in candidates:
                candidates.append(cleaned_brace)

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                # Try progressively cleaning
                cleaned = OptionalImageBasicAgent._clean_json_text(candidate)
                if cleaned != candidate:
                    try:
                        parsed = json.loads(cleaned)
                        repair_log.append(f"cleaned_json: {exc} -> success")
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            if isinstance(parsed, dict):
                return parsed, think_text, repair_log

        # If all parsing failed, check for truncation and log it
        if "{" in remaining and "}" not in remaining:
            repair_log.append("truncation_detected: JSON starts with { but no closing }")
        if "[" in remaining and "]" not in remaining:
            repair_log.append("truncation_detected: JSON starts with [ but no closing ]")

        return None, think_text, repair_log
