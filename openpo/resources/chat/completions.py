import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import httpx
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from openpo.internal import helper, prompt
from openpo.internal.error import APIError, InvalidStreamError
from openpo.internal.response import ChatCompletionOutput, ChatCompletionStreamOutput


class Completions:
    def __init__(self, client: Union[InferenceClient, Dict]):
        """
        Initialize Completions with either HuggingFace InferenceClient or custom API client configuration.
        """
        if not client.get("base_url", ""):
            self.client = client["inference_client"]
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
    ) -> Union[ChatCompletionOutput, Generator[ChatCompletionStreamOutput, None, None]]:
        if params.get("stream", False):

            def stream_generator():
                with httpx.stream(
                    method="POST",
                    url=endpoint,
                    headers=self.headers,
                    data=json.dumps(params),
                    timeout=45.0,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            if not line.startswith("data:"):
                                return None

                            if "[DONE]" in line:
                                break

                            chunk = json.loads(line[6:])
                            try:
                                yield ChatCompletionStreamOutput(chunk)
                            except json.JSONDecodeError:
                                continue

            return stream_generator()
        else:
            try:
                with httpx.Client() as client:
                    response = client.post(
                        endpoint,
                        headers=self.headers,
                        data=json.dumps(params),
                        timeout=45.0,
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
                raise APIError(
                    f"Network error during API request: {str(e)}", error=str(e)
                )

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
                res_1 = self.client.chat_completion(
                    model=model,
                    messages=messages,
                    **params,
                )

                params["temperature"] = pref_params.get("temperature", 1.3)
                params["frequency_penalty"] = pref_params.get("frequency_penalty", 0.2)

                res_2 = self.client.chat_completion(
                    model=model,
                    messages=messages,
                    **params,
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
                res_1 = self._make_api_request(self.base_url, params)

                params["temperature"] = pref_params.get("temperature", 1.3)
                params["frequency_penalty"] = pref_params.get("frequency_penalty", 0.2)

                res_2 = self._make_api_request(self.base_url, params)

                return [res_1, res_2]

            # For single response case
            return self._make_api_request(self.base_url, params)

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

            return self._make_api_request(self.base_url, params)
