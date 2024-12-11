import json
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

    def _get_model_id(self, model: str) -> str:
        return model.split("/", 1)[1]

    def _get_provider_instance(self, provider: str):
        if provider == "huggingface":
            return HuggingFace(api_key=os.getenv("HF_API_KEY"))
        else:
            return OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    def _get_model_consensus(
        self,
        res_a: List[Dict],
        res_b: List[Dict],
    ) -> List[int]:
        if len(res_a) != len(res_b):
            raise ValueError(
                "responses for evaluation do not have identical number of responses."
            )

        matching_indices = []
        for i, (a, b) in enumerate(zip(res_a, res_b)):
            if a["rank"] == b["rank"]:
                matching_indices.append(i)

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
        """
        responses = []

        for m in models:
            try:
                provider = self._get_model_provider(model=m)
                model_id = self._get_model_id(model=m)
                llm = self._get_provider_instance(provider=provider)

                res = llm.generate(model=model_id, messages=messages, params=params)
                responses.append(res)
            except Exception as e:
                raise Exception(f"Failed to execute chat completions: {e}")

        return responses

    def eval_single(
        self,
        model: str,
        data: List[List],
        prompt: Optional[str] = None,
    ):
        """Use single LLM-as-a-judge method to evaluate responses for building preference data.

        Args:
            model (str): Model identifier to use as a judge. Follows provider/model-identifier format.
            data (List[List]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns: The evaluation data for responses with preferred, rejected, confidence_score and reason.
        """

        provider = self._get_model_provider(model)
        model_id = self._get_model_id(model)

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
                prompt=prompt if prompt else None,
            )
            if provider == "anthropic":
                return res.content[0].input

            return res.choices[0].message.content
        except Exception as e:
            raise (f"error annotating dataset: {e}")

    def eval_multi(
        self,
        models: List[str],
        data: List[List],
        prompt: Optional[str] = None,
    ):
        """Use multiple LLMs as a judge for model consensus to evaluate responses for building preference data.

        Args:
            models (List): List of models to use as a judge. Follows provider/model-identifier format.
            data (List[List]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns: The evaluation data for responses that all models agree on.
        """

        judge_a = Anthropic()
        judge_o = OpenAI()

        a_model = ""
        o_model = ""

        for m in models:
            provider = self._get_model_provider(m)

            if provider == "anthropic":
                a_model = self._get_model_id(m)
            else:
                o_model = self._get_model_id(m)

        res_a = judge_a.generate(
            model=a_model,
            data=data,
            prompt=prompt if prompt else None,
        )
        parsed_res_a = res_a.content[0].input["preference"]

        res_o = judge_o.generate(
            model=o_model,
            data=data,
            prompt=prompt if prompt else None,
        )
        parsed_res_o = json.loads(res_o.choices[0].message.content)["preference"]

        idx = self._get_model_consensus(
            parsed_res_a,
            parsed_res_o,
        )

        return {"preference": [parsed_res_o[i] for i in idx]}
