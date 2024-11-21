import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from openpo.internal import helper, prompt
from openpo.internal.error import APIError
from openpo.internal.response import ChatCompletionOutput


class ExtractModel(BaseModel):
    main_answer: str
    key_points: str


class Completions:
    def __init__(self, client: Union[InferenceClient, Dict]):
        """
        Initialize Completions with either HuggingFace InferenceClient or custom API client configuration.

        Args:
            client: Either a HuggingFace InferenceClient instance or a dictionary containing custom API configuration
        """
        if isinstance(client, InferenceClient):
            self.client = client
            self.custom_api = False
        elif isinstance(client, dict):
            self.base_url = client["base_url"]
            self.headers = client["headers"]
            self.custom_api = True
        else:
            raise ValueError(
                "Client must be either InferenceClient or a configuration dictionary"
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
                # Convert custom API response to OpenAI format
                return ChatCompletionOutput(response.json())

        except httpx.HTTPStatusError as e:
            raise APIError(
                f"API request failed: {str(e)}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
            )

        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")

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
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        pref_response_format: Optional[dict] = None,
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
        Conditionally outputs two responses for human preference based on diff_frequency parameter.
        if base_url is not provided, it defaults to calling HuggingFace inference API.
        """

        if not self.custom_api:
            if helper.should_run(diff_frequency):
                # Extract key points and main answer
                res = self.client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt.EXTRACT_PROMPT},
                        *messages[1:],
                    ],
                    response_format={
                        "type": "json",
                        "value": ExtractModel.model_json_schema(),
                    },
                    max_tokens=max_tokens,
                )

                clean_res = helper.clean_str(res.choices[0].message.content)
                content = json.loads(clean_res)

                pref_response = self.client.chat_completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt.PREF_PROMPT.format(
                                messages[0]["content"],
                                content["main_answer"],
                                content["key_points"],
                            ),
                        },
                        *messages[1:],
                    ],
                    response_format=(
                        {
                            "type": "json",
                            "value": pref_response_format.model_json_schema(),
                        }
                        if pref_response_format
                        else None
                    ),
                    temperature=temperature or 0.2,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    max_tokens=max_tokens,
                    n=n,
                    seed=seed,
                    stop=stop,
                    stream=stream,
                    stream_options=stream_options,
                    top_logprobs=top_logprobs,
                    top_p=top_p,
                    tool_choice=tool_choice,
                    tool_prompt=tool_prompt,
                    tools=tools,
                )

                return pref_response

            # For single response case
            completion = self.client.chat_completion(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                response_format=(
                    {
                        "type": "json",
                        "value": response_format.model_json_schema(),
                    }
                    if response_format
                    else None
                ),
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=stream_options,
                temperature=temperature,
                top_logprobs=top_logprobs,
                top_p=top_p,
                tool_choice=tool_choice,
                tool_prompt=tool_prompt,
                tools=tools,
            )

            return completion

        else:
            # Custom API case
            if helper.should_run(diff_frequency):
                # First call for extraction
                extract_params = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt.EXTRACT_PROMPT_JSON.format(
                                ExtractModel.model_json_schema()["required"]
                            ),
                        },
                        *messages[1:],
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": max_tokens,
                }

                res = self._make_api_request(self.base_url, json.dumps(extract_params))
                clean_res = helper.clean_str(res.choices[0].message.content)
                content = json.loads(clean_res)

                # Second call for preference
                pref_content = prompt.PREF_PROMPT_JSON.format(
                    prompt.PREF_PROMPT.format(
                        messages[0]["content"],
                        content["main_answer"],
                        content["key_points"],
                    ),
                    (
                        pref_response_format.model_json_schema()["required"]
                        if pref_response_format
                        else ""
                    ),
                )

                pref_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": pref_content},
                        *messages[1:],
                    ],
                    "response_format": (
                        {"type": "json_object"} if pref_response_format else None
                    ),
                    "temperature": temperature or 0.2,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_tokens": max_tokens,
                    "n": n,
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

                return self._make_api_request(self.base_url, json.dumps(pref_params))

            # For single response case
            single_content = prompt.SINGLE_PROMPT_JSON.format(
                messages[0]["content"],
                (
                    response_format.model_json_schema()["required"]
                    if response_format
                    else ""
                ),
            )

            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": single_content},
                    *messages[1:],
                ],
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

        Args:
            model: The model to use for completion
            messages: The messages to generate completion for
            ... (other optional parameters)

        Returns:
            For HuggingFace: Original InferenceClient response (OpenAI format)
            For custom API: ChatCompletion object with OpenAI-style attributes

        Raises:
            APIError: If the API request fails
        """
        if not self.custom_api:
            try:
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
            except Exception as e:
                raise APIError(f"HuggingFace API request failed: {str(e)}")
        else:
            # For custom API, inject schema into prompt if response_format is provided
            if response_format:
                messages[0]["content"] = prompt.SINGLE_PROMPT_JSON.format(
                    messages[0]["content"],
                    response_format.model_json_schema(),
                )

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
