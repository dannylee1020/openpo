import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .internal.response import ChatCompletionOutput, ChatCompletionStreamOutput
from .resources.provider.anthropic import Anthropic
from .resources.provider.huggingface import HuggingFace
from .resources.provider.openai import OpenAI
from .resources.provider.openrouter import OpenRouter


class OpenPO:
    """
    Main client class for interacting with various LLM providers.

    This class serves as the primary interface for making completion requests to different
    language model providers.
    """

    def _get_model_provider(self, model: str) -> str:
        return model.split("/")[0]

    def _get_model_identifier(self, model: str) -> str:
        return model.split("/", 1)[1]

    def _get_provider_instance(self, provider: str):
        if provider == "huggingface":
            return HuggingFace(api_key=os.getenv("HF_API_KEY"))
        else:
            return OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

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
        """
        responses = []

        for m in models:
            try:
                provider = self._get_model_provider(model=m)
                model_id = self._get_model_identifier(model=m)
                llm = self._get_provider_instance(provider=provider)

                res = llm.generate(model=model_id, messages=messages, params=params)
                responses.append(res)
            except Exception as e:
                raise Exception(f"Failed to execute chat completions: {e}")

        return responses

    def eval_single(
        self,
        model: str,
        data: List[List[str, str]],
        prompt: Optional[str] = None,
    ):
        """Use single LLM-as-a-judge method to evaluate responses for building preference.

        Args:
            model (str): Model identifier to use for annotation. Follows <provider>/<model-identifier> format.
            data (List): pairwise responses dataset to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns: The annotated data with preferred, rejected, confidence_score and reason.
        """

        provider = self._get_model_provider(model)
        model_id = self._get_model_identifier(model)

        if provider == "openai":
            llm = OpenAI()
        elif provider == "anthropic":
            llm = Anthropic()
        else:
            raise ValueError("provider not supported for annotation")
        try:
            res = llm.generate(
                model=model_id,
                data=data,
            )

            return res
        except Exception as e:
            raise (f"error annotating dataset: {e}")
