import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from openpo.internal import prompt
from openpo.internal.error import APIError
from openpo.internal.response import ChatCompletionOutput, ChatCompletionStreamOutput

from .base import LLMProvider


class OpenRouter(LLMProvider):
    """
    A provider class for interacting with OpenRouter's API service.

    This class implements the LLMProvider interface to handle text generation requests
    through OpenRouter's API. It manages API authentication and provides methods for
    generating text completions with support for both streaming and non-streaming responses.

    Attributes:
        url (str): The OpenRouter API endpoint URL.
        api_key (str): The OpenRouter API key for authentication.
        headers (Dict[str, str]): HTTP headers used for API requests.

    Args:
        api_key (str): The API key for OpenRouter. If not provided,
            attempts to read from OPENROUTER_API_KEY environment variable.

    Raises:
        ValueError: If no API key is provided or found in environment variables.
    """

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
        model: str,
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate text completions using specified OpenRouter models.

        Args:
            models (List[str]): List of model identifiers to use for generation.
            messages (List[Dict[str, Any]]): List of message dictionaries containing
                the conversation history.
            params (Optional[Dict[str, Any]]): Additional parameters for the generation:
                - frequency_penalty (Optional[float]): Penalty for token frequency
                - logit_bias (Optional[List[float]]): Token biases for generation
                - logprobs (Optional[bool]): Whether to return log probabilities
                - max_tokens (Optional[int]): Maximum number of tokens to generate
                - presence_penalty (Optional[float]): Penalty for token presence
                - response_format (Optional[dict]): Desired format for the response
                - seed (Optional[int]): Random seed for generation
                - stop (Optional[int]): Stop sequence for generation
                - stream (Optional[bool]): Whether to stream the response
                - stream_options (Optional[dict]): Options for streaming
                - temperature (Optional[float]): Sampling temperature
                - top_logprobs (Optional[int]): Number of top log probabilities to return
                - top_p (Optional[float]): Nucleus sampling parameter
                - tool_choice (Optional[str]): Tool selection parameter
                - tool_prompt (Optional[str]): Prompt for tool usage
                - tools (Optional[List[dict]]): List of available tools
                - pref_params (Optional[[Dict[str, Any]]): Model-specific parameters

        Returns:
            ChatCompletionOutput | ChatCompletionStreamOutput:
                Response from model, which can be either direct outputs or
                stream generators depending on the stream parameter.

        Raises:
            Exception: If there's an error during the API request or response processing.
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

            if params.get("pref_params"):
                params.update(params["pref_params"])
                del params["pref_params"]

            body = {"model": model, "messages": messages, **params}
            res = self._make_api_request(endpoint=self.url, params=body)

            return res
        except Exception as e:
            raise Exception(f"Error calling model from OpenRouter {e}")
