import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from .base import LLMProvider


class HuggingFace(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not provided")

        self.client = InferenceClient(api_key=self.api_key)

    def generate(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        PARAMS

        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[int] = None,
        stream: Optional[bool] = False,
        stream_options: Optional[dict] = None,
        temperature: Optional[float] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        tool_choice: Optional[str] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        pref_params: Optional[List[Dict[str, Any]]] = None,
        """

        try:
            if params is None:
                params = {}

            if params.get("response_format"):
                params.update(
                    {
                        "response_format": {
                            "type": "json",
                            "value": params["response_format"].model_json_schema(),
                        }
                    }
                )

            response = []
            pref_params = params.pop("pref_params", [])

            for i in range(len(models)):
                if i < len(pref_params):
                    params.update(pref_params[i])

                res = self.client.chat_completion(
                    model=models[i], messages=messages, **params
                )
                response.append(res)

            return response
        except Exception as e:
            raise Exception(f"Error calling LLM from HuggingFace: {e}")
