import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from anthropic import Anthropic as AnthropicClient
from pydantic import BaseModel

from openpo.internal import prompt as prompt_lib
from openpo.internal.error import AuthenticationError, ProviderError

from .base import LLMProvider


class AnnotateModel(BaseModel):
    rank: List[int]
    preferred_score: float
    rejected_score: float
    reason: str


class Response(BaseModel):
    evaluation: List[AnnotateModel]


class Anthropic(LLMProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise AuthenticationError("Anthropic")
        try:
            self.client = AnthropicClient(api_key=api_key)
        except Exception as e:
            raise AuthenticationError(
                "Anthropic", message=f"Failed to initialize Anthropic client: {str(e)}"
            )

    def generate(
        self,
        model: str,
        questions: List[str],
        responses: List[List],
        prompt: Optional[str] = None,
    ):
        tools = [
            {
                "name": "build_response_output",
                "description": "build response output with predefined structure",
                "input_schema": Response.model_json_schema(),
            }
        ]

        try:
            res = self.client.messages.create(
                model=model,
                system=prompt if prompt else prompt_lib.EVALUATION_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_lib.EVALUATION_QUERY.format(
                            questions, responses
                        ),
                    },
                ],
                max_tokens=4096,
                tools=tools,
                tool_choice={"type": "tool", "name": "build_response_output"},
            )

            return res

        except Exception as e:
            raise ProviderError(
                "Anthropic", message=f"Request to Anthropic model failed: {str(e)}"
            )
