from typing import Any, Dict, List, Optional, Union

from openpo.internal.error import AuthenticationError, ProviderError
from openpo.internal.response import ChatCompletionOutput


class Completion:
    def __init__(self, client):
        self.client = client

    def generate(
        self,
        model: Union[str, List[str]],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[ChatCompletionOutput]:
        """Generate completions using the specified LLM provider.

        Args:
            model (str, List[str]): model identifier or list of model identifiers to use for generation. Follows <provider>/<model-identifier> format.
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
        if isinstance(model, str):
            try:
                provider = self.client._get_model_provider(model=model)
                model_id = self.client._get_model_id(model=model)
                llm = self.client._get_provider_instance(provider=provider)

                res = llm.generate(model=model_id, messages=messages, params=params)
                return res
            except Exception as e:
                raise ProviderError(
                    provider=provider,
                    message=f"Failed to execute chat completions: {str(e)}",
                )
        responses = []
        for m in model:
            try:
                provider = self.client._get_model_provider(model=m)
                model_id = self.client._get_model_id(model=m)
                llm = self.client._get_provider_instance(provider=provider)

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
