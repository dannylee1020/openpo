import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from openpo.internal import prompt
from openpo.internal.error import APIError
from openpo.internal.response import ChatCompletionOutput, ChatCompletionStreamOutput

from .base import LLMProvider


class OpenRouter(LLMProvider):
    def __init__(self, api_key):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not provided")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

    def generate(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        params
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
                messages = [
                    {
                        "role": "system",
                        "content": prompt.JSON_PROMPT.format(
                            messages[0]["content"],
                            params["response_format"].model_json_schema()["required"],
                        ),
                    },
                    *messages[1:],
                ]

                params["response_format"] = None

            response = []
            pref_params = params.pop("pref_params", [])
            for i in range(len(models)):
                if i < len(pref_params):
                    params.update(pref_params[i])

                params = {
                    "model": models[i],
                    "messages": messages,
                    **params,
                }

                res = self._make_api_request(endpoint=self.url, params=params)
                response.append(res)

            return response
        except Exception:
            raise
