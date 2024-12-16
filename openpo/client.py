import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .internal.error import AuthenticationError, ProviderError
from .internal.response import ChatCompletionOutput, ChatCompletionStreamOutput
from .resources.provider.anthropic import Anthropic
from .resources.provider.huggingface import HuggingFace
from .resources.provider.openai import OpenAI
from .resources.provider.openrouter import OpenRouter


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

    def _get_model_consensus(
        self,
        res_a: List[Dict],
        res_b: List[Dict],
    ) -> List[int]:

        matching_indices = []
        for i, (a, b) in enumerate(zip(res_a, res_b)):
            if a.get("q_index") == b.get("q_index") and a["rank"] == b["rank"]:
                matching_indices.append(a.get("q_index", i))

        return matching_indices

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

    def eval_single(
        self,
        model: str,
        questions: List[str],
        responses: List[List[str]],
        prompt: Optional[str] = None,
    ):
        """Use single LLM-as-a-judge method to evaluate responses for building preference data.

        Args:
            model (str): Model identifier to use as a judge. Follows provider/model-identifier format.
            questions (List(str)): Questions for each response pair.
            responses (List[List[str]]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns (Dict): The evaluation data for responses with preferred, rejected, confidence_score and reason.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during evaluation.
            ValueError: If the model format is invalid or provider is not supported.
        """
        try:
            provider = self._get_model_provider(model)
            model_id = self._get_model_id(model)

            if provider not in ["openai", "anthropic"]:
                raise ProviderError(provider, "Provider not supported for evaluation")

            llm = self._get_provider_instance(provider=provider)
            res = llm.generate(
                model=model_id,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )

            if provider == "anthropic":
                result = res.content[0].input['"evaluation']
            result = json.loads(res.choices[0].message.content)["evaluation"]

            return {"evaluation": result}
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider=provider, message=f"Error during evaluation: {str(e)}"
            )

    def eval_multi(
        self,
        models: List[str],
        questions: List[str],
        responses: List[List],
        prompt: Optional[str] = None,
    ):
        """Use multiple LLMs as a judge for model consensus to evaluate responses for building preference data.

        Args:
            models (List): List of models to use as a judge. Follows provider/model-identifier format.
            questions (List(str)): Questions for each response pair.
            responses (List[List[str]]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns (Dict): The evaluation data for responses that all models agree on.

            - preference: Evaluation data on the input responses.
            - q_index: Index of questions that reached consensus by the models.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during evaluation.
            ValueError: If the model format is invalid or required models are missing.
        """
        try:
            judge_a = self._get_provider_instance("anthropic")
            judge_o = self._get_provider_instance("openai")

            a_model = ""
            o_model = ""

            for m in models:
                provider = self._get_model_provider(m)
                if provider == "anthropic":
                    a_model = self._get_model_id(m)
                elif provider == "openai":
                    o_model = self._get_model_id(m)
                else:
                    raise ProviderError(
                        provider, "Provider not supported for evaluation"
                    )

            if not a_model or not o_model:
                raise ValueError("Both Anthropic and OpenAI models must be provided")

            res_a = judge_a.generate(
                model=a_model,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )
            parsed_res_a = res_a.content[0].input["evaluation"]

            res_o = judge_o.generate(
                model=o_model,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )
            parsed_res_o = json.loads(res_o.choices[0].message.content)["evaluation"]

            idx = self._get_model_consensus(
                parsed_res_a,
                parsed_res_o,
            )

            return {
                "evaluation": [parsed_res_o[i] for i in idx],
                "q_index": idx,
            }
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider="eval-multi",
                message=f"Error during multi-model evaluation: {str(e)}",
            )
