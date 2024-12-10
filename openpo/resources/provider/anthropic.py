import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from anthropic import Anthropic as AnthropicClient
from pydantic import BaseModel

from openpo.internal import prompt

from .base import LLMProvider


class AnnotateModel(BaseModel):
    rank: List[int]
    preferred_score: float
    rejected_score: float
    reason: str


class Response(BaseModel):
    preference: List[AnnotateModel]


class Anthropic(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided")

        self.client = AnthropicClient(api_key=self.api_key)

    def generate(
        self,
        model: str,
        data: Union[List[Dict[str, Any]], pd.DataFrame],
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
                system=prompt if prompt else prompt.EVALUATION_PROMPT,
                messages=[
                    {"role": "user", "content": prompt.EVALUATION_QUERY.format(data)},
                ],
                max_tokens=4096,
                tools=tools,
                tool_choice={"type": "tool", "name": "build_response_output"},
            )

            return res
        except Exception as e:
            raise Exception(f"request to Anthropic model failed: {e}")
