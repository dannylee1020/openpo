import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from huggingface_hub import AsyncInferenceClient, InferenceClient
from pydantic import BaseModel

from openpo.internal import helper, prompt
from openpo.internal.error import APIError
from openpo.internal.response import ChatCompletionOutput


class Completions:
    def __init__(self, client: Union[InferenceClient, Dict]):
        """
        Initialize Completions with either HuggingFace InferenceClient or custom API client configuration.
        """
        if not client.get("base_url", ""):
            self.client = client["inference_client"]
            self.async_client = AsyncInferenceClient(api_key=client["api_key"])
            self.custom_api = False
        elif client.get("base_url", ""):
            self.base_url = client["base_url"]
            self.headers = client["headers"]
            self.custom_api = True
        else:
            raise ValueError(
                "Invalid client configuration: Missing required parameters"
            )

    def _make_api_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> ChatCompletionOutput:
        try:
            with httpx.Client() as client:
                response = client.post(
                    endpoint, headers=self.headers, data=params, timeout=30.0
                )
                response.raise_for_status()

                return ChatCompletionOutput(response.json())

        except httpx.HTTPStatusError as e:
            raise APIError(
                f"API request to {endpoint} failed",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
                error=str(e),
            )

        except httpx.RequestError as e:
            raise APIError(f"Network error during API request: {str(e)}", error=str(e))

    async def _make_async_api_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> ChatCompletionOutput:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint, headers=self.headers, data=params, timeout=30.0
                )
                response.raise_for_status()
                return ChatCompletionOutput(response.json())

        except httpx.HTTPStatusError as e:
            raise APIError(
                f"API request failed",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
                error=str(e),
            )

        except httpx.RequestError as e:
            raise APIError(
                f"Network error during async API request: {str(e)}", error=str(e)
            )

    async def _concurrent_preference_calls(
        self,
        params: Dict[str, Any],
        pref_params: Optional[Dict],
    ) -> tuple[ChatCompletionOutput, ChatCompletionOutput]:
        pref_task1 = self._make_async_api_request(self.base_url, json.dumps(params))

        # update to custom values
        params["temperature"] = pref_params.get("temperature", 1.2)
        params["frequency_penalty"] = pref_params.get("frequency_penalty", 0.0)
        params["presence_penalty"] = pref_params.get("presence_penalty", 0.0)
        pref_task2 = self._make_async_api_request(self.base_url, json.dumps(params))

        pref_result1, pref_result2 = await asyncio.gather(pref_task1, pref_task2)
        return pref_result1, pref_result2

    async def _concurrent_hf_preference_calls(
        self,
        model: str,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, str],
        pref_params: Optional[Dict] = None,
    ) -> tuple[Any, Any]:

        task1 = self.async_client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )
        if pref_params:
            kwargs["temperature"] = pref_params.get("temperature", 1.2)
            kwargs["frequency_penalty"] = pref_params.get("frequency_penalty", 0.0)
            kwargs["presence_penalty"] = pref_params.get("presence_penalty", 0.0)

        task2 = self.async_client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )

        result1, result2 = await asyncio.gather(task1, task2)

        return result1, result2

    def create_preference(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        diff_frequency: Optional[float] = 0.0,
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
        pref_params: Optional[dict] = {},
    ):
        """
        Conditionally outputs two responses for human preference based on diff_frequency parameter.
        if base_url is not provided, it defaults to calling HuggingFace inference API.
        """

        if not self.custom_api:
            params = {
                "response_format": (
                    {"type": "json", "value": response_format.model_json_schema()}
                    if response_format
                    else None
                ),
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "stream_options": stream_options,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "tool_choice": tool_choice,
                "tool_prompt": tool_prompt,
                "tools": tools,
            }

            if helper.should_run(diff_frequency):
                res_1, res_2 = asyncio.run(
                    self._concurrent_hf_preference_calls(
                        model=model,
                        messages=messages,
                        pref_params=pref_params,
                        kwargs=params,
                    )
                )
                return [res_1, res_2]

            # For single response case
            return self.client.chat_completion(model=model, messages=messages, **params)
        else:
            # Custom API case
            if response_format:
                messages = [
                    {
                        "role": "system",
                        "content": prompt.JSON_PROMPT.format(
                            messages[0]["content"],
                            response_format.model_json_schema()["required"],
                        ),
                    },
                    *messages[1:],
                ]

            params = {
                "model": model,
                "messages": messages,
                "response_format": (
                    {"type": "json_object"} if response_format else None
                ),
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "stream_options": stream_options,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "tool_choice": tool_choice,
                "tool_prompt": tool_prompt,
                "tools": tools,
            }

            if helper.should_run(diff_frequency):
                # Make two concurrent preference calls
                pref_result1, pref_result2 = asyncio.run(
                    self._concurrent_preference_calls(params, pref_params)
                )
                return [pref_result1, pref_result2]

            # For single response case
            return self._make_api_request(self.base_url, json.dumps(params))

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
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
    ):
        """
        Create a chat completion using either HuggingFace or custom API.
        """

        if not self.custom_api:
            return self.client.chat_completion(
                **{
                    "model": model,
                    "messages": messages,
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "response_format": (
                        {
                            "type": "json",
                            "value": response_format.model_json_schema(),
                        }
                        if response_format
                        else None
                    ),
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "stream_options": stream_options,
                    "temperature": temperature,
                    "top_logprobs": top_logprobs,
                    "top_p": top_p,
                    "tool_choice": tool_choice,
                    "tool_prompt": tool_prompt,
                    "tools": tools,
                }
            )
        else:
            # For custom API, inject schema into prompt if response_format is provided
            if response_format:
                messages = [
                    {
                        "role": "system",
                        "content": prompt.JSON_PROMPT.format(
                            messages[0]["content"],
                            response_format.model_json_schema()["required"],
                        ),
                    },
                    *messages[1:],
                ]

            params = {
                "model": model,
                "messages": messages,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "n": n,
                "presence_penalty": presence_penalty,
                "response_format": {"type": "json_object"} if response_format else None,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "tool_choice": tool_choice,
                "tool_prompt": tool_prompt,
                "tools": tools,
            }

            return self._make_api_request(self.base_url, json.dumps(params))
