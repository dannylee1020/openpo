import os
from typing import Any, Dict, List, Optional

from ..internal.response import ChatCompletionOutput, ChatCompletionStreamOutput
from .resources.provider.huggingface import HuggingFace
from .resources.provider.openrouter import OpenRouter

ENV_MAPPING = {
    "huggingface": "HF_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class OpenPO:
    """Main client class for interacting with various LLM providers.

    This class serves as the primary interface for making completion requests to different
    language model providers. It currently supports HuggingFace and OpenRouter as providers.

    Parameters:
        provider (str): The name of the LLM provider to use ('huggingface' or 'openrouter').
        api_key (str): The API key for authentication with the chosen provider.

    Raises:
        ValueError: If no API key is provided either through initialization or environment variables.
    """

    def __init__(
        self,
        provider: Optional[str] = "huggingface",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{ENV_MAPPING[provider]}")
        if not self.api_key:
            raise ValueError("No API key is provided")

    def completions(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[ChatCompletionOutput | ChatCompletionStreamOutput]:
        """Generate completions using the specified LLM provider.

        Args:
            models (List[str]): List of model identifiers to use for generation.
            messages (List[Dict[str, Any]]): List of message dictionaries containing
                the conversation history and prompts.
            params (Optional[Dict[str, Any]]): Additional model parameters for the request (e.g., temperature, max_tokens).

        Returns:
            The response from the LLM provider containing the generated completions.
        """
        if self.provider == "huggingface":
            llm = HuggingFace(api_key=self.api_key)
            res = llm.generate(models=models, messages=messages, params=params)

            return res
        else:
            llm = OpenRouter(api_key=self.api_key)
            res = llm.generate(models=models, messages=messages, params=params)

            return res
