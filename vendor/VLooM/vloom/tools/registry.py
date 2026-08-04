from typing import Dict, Type, Optional, List, Any
import logging
from .base import BaseTool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Registry for managing available tools.
    Supports both global class registration and instance-level management.
    """
    _tool_classes: Dict[str, Type[BaseTool]] = {}

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    @classmethod
    def register_tool(cls, name: str = None):
        """Decorator to register a tool class globally."""
        def decorator(tool_cls: Type[BaseTool]):
            tool_name = name or getattr(tool_cls, "name", tool_cls.__name__)
            if tool_name in cls._tool_classes:
                logger.warning(f"Tool class {tool_name} already registered. Overwriting.")
            cls._tool_classes[tool_name] = tool_cls
            return tool_cls
        return decorator

    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[BaseTool]]:
        """Get a registered tool class by name."""
        return cls._tool_classes.get(name)

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered. Overwriting.")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of tool schemas."""
        return [t.to_schema() for t in self._tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found.")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {str(e)}"

