import json
import os
from typing import Any, Dict, List, Optional

from vllm import LLM

from .internal.error import AuthenticationError, ProviderError
from .resources.batch.batch import Batch
from .resources.completion.completion import Completion
from .resources.eval.eval import Evaluation
from .resources.provider import Anthropic, HuggingFace, OpenAI, OpenRouter


class OpenPO:
    """
    Main client class for interacting with various LLM providers.

    This class serves as the primary interface for making completion requests to different
    language model providers. OpenPO takes optional api_key arguments for initialization.

    """

    def __init__(
        self,
        hf_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        self._completion = Completion(self)
        self._eval = Evaluation(self)
        self._batch = Batch(self)

    def _get_model_provider(self, model: str) -> str:
        try:
            return model.split("/")[0]
        except IndexError:
            raise ValueError("Invalid model format. Expected format: provider/model-id")

    def _get_model_id(self, model: str) -> str:
        try:
            return model.split("/", 1)[1]
        except IndexError:
            raise ValueError("Invalid model format. Expected format: provider/model-id")

    def _get_provider_instance(self, provider: str):
        if provider == "huggingface":
            if not self.hf_api_key:
                raise AuthenticationError("HuggingFace")
            return HuggingFace(api_key=self.hf_api_key)

        if provider == "openrouter":
            if not self.openrouter_api_key:
                raise AuthenticationError("OpenRouter")
            return OpenRouter(api_key=self.openrouter_api_key)

        if provider == "openai":
            if not self.openai_api_key:
                raise AuthenticationError("OpenAI")
            return OpenAI(api_key=self.openai_api_key)

        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise AuthenticationError("Anthropic")
            return Anthropic(api_key=self.anthropic_api_key)

        raise ProviderError(provider, "Unsupported model provider")

    @property
    def completion(self):
        """Access the chat completion functionality for LLM response.
        This property provides access to completion interfacce.

        Returns:
            Completion: An instance of the Completion class that provides method
                        for generatng response from LLM.
        """
        return self._completion

    @property
    def evaluate(self):
        """Access the evaluation functionality for LLM responses.
        This property provides access to the evaluation interface.

        Returns:
            Evaluation: An instance of the Evaluation class that provides methods
                       for evaluating and comparing LLM outputs.
        """
        return self._eval

    @property
    def batch(self):
        """Access the batch processing functionality for LLM operations.
        This property provides access to the batch processing interface

        Returns:
            Batch: An instance of the Batch class that provides methods for
                  processing multiple LLM requests efficiently.
        """
        return self._batch
