import json
from typing import Optional
from .base import LossScale

class ManualJSONWeight2LossScale(LossScale):
    """Mildly emphasize valid first-turn retrieval JSON supervision."""
    is_binary = False
    tool_call_weight = 2.0
    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            try: value = json.loads(context)
            except (TypeError, ValueError, json.JSONDecodeError): value = None
            if isinstance(value, dict) and value.get('name') == 'agrinet_rag_search' and isinstance(value.get('arguments'), dict):
                return [context], [self.tool_call_weight]
        return [context], [1.0]
