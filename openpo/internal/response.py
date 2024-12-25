from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


@dataclass
class ChatCompletionOutputMessage:
    role: str
    content: str
    tool_calls: Any = None
    refusal: str = ""


@dataclass
class ChatCompletionOutputUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionOutputComplete:
    finish_reason: str
    index: int
    message: ChatCompletionOutputMessage
    logprobs: Optional[Any] = None


class ChatCompletionOutput(SimpleNamespace):
    """
    Converts a response from endpoint into an object with attribute access to follow OpenAI API format.
    """

    def __init__(self, response_dict: Dict[str, Any]):
        # Convert usage stats if present
        if "usage" in response_dict:
            response_dict["usage"] = ChatCompletionOutputUsage(**response_dict["usage"])

        # Convert choices if present
        if "choices" in response_dict:
            choices = []
            for choice in response_dict["choices"]:
                # Convert the message within the choice
                if "message" in choice:
                    choice["message"] = ChatCompletionOutputMessage(**choice["message"])
                choices.append(ChatCompletionOutputComplete(**choice))
            response_dict["choices"] = choices

        super().__init__(**response_dict)
