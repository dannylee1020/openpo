import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from openai import OpenAI as OpenAIClient
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


class OpenAI(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided")

        self.client = OpenAIClient(api_key=self.api_key)

    def generate(
        self,
        model: str,
        data: Union[List[Dict[str, Any]], pd.DataFrame],
        prompt: Optional[str] = None,
    ):
        messages = [
            {
                "role": "system",
                "content": prompt if prompt else prompt.EVALUATION_PROMPT,
            },
            {"role": "user", "content": prompt.EVALUATION_QUERY.format(data)},
        ]

        try:
            res = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=Response,
            )

            return res
        except Exception as e:
            raise Exception(f"request to OpenAI model failed: {e}")
