import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from .base import LLMProvider


class HuggingFace(LLMProvider):
    """
    A provider class for interacting with HuggingFace's inference API.

    This class implements the LLMProvider interface to handle text generation requests
    through HuggingFace's models. It manages API authentication and provides methods
    for generating text completions.

    Attributes:
        api_key (str): The HuggingFace API key for authentication.
        client (InferenceClient): The HuggingFace inference client instance.

    Args:
        api_key (Optional[str]): The API key for HuggingFace. If not provided,
            attempts to read from HF_API_KEY environment variable.

    Raises:
        ValueError: If no API key is provided or found in environment variables.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not provided")

        self.client = InferenceClient(api_key=self.api_key)

    def generate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate text completions using specified HuggingFace model.

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
                - pref_params (Optional[Dict[str, Any]]): Model-specific parameters

        Returns:
            ChatCompletionOutput | ChatCOmpletionStreamOutput: Response from the model.

        Raises:
            Exception: If there's an error calling the HuggingFace API.
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

            if params.get("pref_params"):
                params.update(params["pref_params"])
                del params["pref_params"]

            res = self.client.chat_completion(model=model, messages=messages, **params)

            return res
        except Exception as e:
            raise Exception(f"Error calling model from HuggingFace: {e}")
