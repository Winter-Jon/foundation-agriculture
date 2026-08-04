import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


def fix_json_content(content: str) -> Optional[str]:
    """尝试修复JSON内容"""
    try:
        # 移除可能的markdown格式
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试解析
        json.loads(content)
        return content
    except json.JSONDecodeError:
        lines = content.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
            if line.endswith('}'):
                break
        
        if json_lines:
            try:
                fixed_content = '\n'.join(json_lines)
                json.loads(fixed_content)
                return fixed_content
            except json.JSONDecodeError:
                pass
        
        return None


def parse_json(content: str) -> Dict[str, Any]:
    """解析JSON，如果失败则尝试修复"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fixed = fix_json_content(content)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        raise


class BasePostprocessor(ABC):
    """输出后处理器基类。"""

    @abstractmethod
    def parse(self, content: str, config: Dict[str, Any]) -> Any:
        pass


class JsonPostprocessor(BasePostprocessor):
    """JSON 输出后处理器。"""

    def parse(self, content: str, config: Dict[str, Any]) -> Any:
        return parse_json(content)


class RegexPostprocessor(BasePostprocessor):
    """正则输出后处理器。支持按字段提取。"""

    _FLAG_MAP = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }

    def parse(self, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        fields_cfg = config.get("fields", {})
        if not fields_cfg:
            raise ValueError("Regex postprocess requires non-empty 'fields' config.")

        result: Dict[str, Any] = {}

        for field_name, field_cfg in fields_cfg.items():
            pattern = field_cfg.get("pattern")
            if not pattern:
                raise ValueError(f"Regex field '{field_name}' requires 'pattern'.")

            flags = self._parse_flags(field_cfg.get("flags"))
            group = field_cfg.get("group", 1)
            required = field_cfg.get("required", True)
            all_matches = field_cfg.get("all_matches", False)
            transform = field_cfg.get("transform")

            if all_matches:
                matches = re.findall(pattern, content, flags=flags)
                values = [self._extract_group(m, group) for m in matches]
                values = [v for v in values if v is not None]
                if not values and required:
                    raise ValueError(f"Required regex field '{field_name}' not found.")
                if transform:
                    values = [self._apply_transform(v, transform) for v in values]
                result[field_name] = values
                continue

            match = re.search(pattern, content, flags=flags)
            if not match:
                if required:
                    raise ValueError(f"Required regex field '{field_name}' not found.")
                result[field_name] = None
                continue

            value = self._extract_group(match, group)
            if value is None and required:
                raise ValueError(f"Regex field '{field_name}' has no valid capture group.")

            if value is not None and transform:
                value = self._apply_transform(value, transform)

            result[field_name] = value

        return result

    def _parse_flags(self, flags_cfg: Any) -> int:
        if not flags_cfg:
            return 0
        if isinstance(flags_cfg, str):
            flags_cfg = [flags_cfg]

        flags = 0
        for name in flags_cfg:
            if not isinstance(name, str):
                continue
            flags |= self._FLAG_MAP.get(name.upper(), 0)
        return flags

    def _extract_group(self, match: Any, group: Any) -> Optional[str]:
        if isinstance(match, str):
            return match.strip()

        if isinstance(match, tuple):
            if isinstance(group, int):
                idx = group - 1
                if idx < 0 or idx >= len(match):
                    return None
                return str(match[idx]).strip()
            return None

        if isinstance(group, int):
            try:
                return match.group(group).strip()
            except (IndexError, AttributeError):
                return None

        if isinstance(group, str):
            try:
                return match.group(group).strip()
            except (IndexError, AttributeError):
                return None

        return None

    def _apply_transform(self, value: str, transform: str) -> Any:
        t = transform.lower()
        if t == "strip":
            return value.strip()
        if t == "int":
            return int(value)
        if t == "float":
            return float(value)
        if t == "int_list":
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        if t == "json":
            return json.loads(value)
        return value


_POSTPROCESSORS: Dict[str, BasePostprocessor] = {
    "json": JsonPostprocessor(),
    "regex": RegexPostprocessor(),
}


def parse_output(content: str, postprocess_config: Optional[Dict[str, Any]] = None) -> Any:
    """根据配置解析模型输出。默认按 JSON 解析。"""
    cfg = postprocess_config or {"type": "json"}
    ptype = str(cfg.get("type", "json")).lower()

    processor = _POSTPROCESSORS.get(ptype)
    if processor is None:
        raise ValueError(f"Unsupported postprocess type: {ptype}")
    return processor.parse(content, cfg)
