import json
import os
from typing import Any, Dict, List, Optional

from .internal.error import AuthenticationError, ProviderError
from .internal.response import ChatCompletionOutput, ChatCompletionStreamOutput
from .resources.batch.batch import Batch
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

    def completions(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[ChatCompletionOutput | ChatCompletionStreamOutput]:
        """Generate completions using the specified LLM provider.

        Args:
            models (List[str]): List of model identifiers to use for generation. Follows <provider>/<model-identifier> format.
            messages (List[Dict[str, Any]]): List of message dictionaries containing
                the conversation history and prompts.
            params (Optional[Dict[str, Any]]): Additional model parameters for the request (e.g., temperature, max_tokens).

        Returns:
            The response from the LLM provider containing the generated completions.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during completion generation.
            ValueError: If the model format is invalid.
        """
        responses = []

        for m in models:
            try:
                provider = self._get_model_provider(model=m)
                model_id = self._get_model_id(model=m)
                llm = self._get_provider_instance(provider=provider)

                res = llm.generate(model=model_id, messages=messages, params=params)
                responses.append(res)
            except (AuthenticationError, ValueError) as e:
                # Re-raise authentication and validation errors as is
                raise e
            except Exception as e:
                raise ProviderError(
                    provider=provider,
                    message=f"Failed to execute chat completions: {str(e)}",
                )

        return responses

    @property
    def eval(self):
        return self._eval

    @property
    def batch(self):
        return self._batch
