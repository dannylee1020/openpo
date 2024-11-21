from types import SimpleNamespace
from typing import Any, Dict


class ChatCompletionOutput(SimpleNamespace):
    """
    converts a response from endpoint into an object with attribute access to follow OpenAI API format.
    """

    def __init__(self, response_dict: Dict[str, Any]):
        def convert_dict(obj):
            if isinstance(obj, dict):
                return ChatCompletionOutput(obj)
            elif isinstance(obj, list):
                return [convert_dict(item) for item in obj]
            return obj

        super().__init__(**{k: convert_dict(v) for k, v in response_dict.items()})
