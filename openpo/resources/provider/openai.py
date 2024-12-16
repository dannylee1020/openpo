import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from openai import OpenAI as OpenAIClient
from openai import OpenAIError
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


class OpenAI(LLMProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise AuthenticationError("OpenAI")
        try:
            self.client = OpenAIClient(api_key=api_key)
        except Exception as e:
            raise AuthenticationError(
                "OpenAI", message=f"Failed to initialize OpenAI client: {str(e)}"
            )

    def generate(
        self,
        model: str,
        questions: List[str],
        responses: List[List],
        prompt: Optional[str] = None,
    ):
        messages = [
            {
                "role": "system",
                "content": prompt if prompt else prompt_lib.EVALUATION_PROMPT,
            },
            {
                "role": "user",
                "content": prompt_lib.EVALUATION_QUERY.format(questions, responses),
            },
        ]

        try:
            res = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=Response,
            )

            return res
        except OpenAIError as e:
            if "authentication" in str(e).lower():
                raise AuthenticationError(
                    "OpenAI",
                    message=str(e),
                    status_code=e.status_code if hasattr(e, "status_code") else None,
                    response=e.response if hasattr(e, "response") else None,
                )
            raise ProviderError(
                "OpenAI",
                message=str(e),
                status_code=e.status_code if hasattr(e, "status_code") else None,
                response=e.response if hasattr(e, "response") else None,
            )
        except Exception as e:
            raise ProviderError(
                "OpenAI", message=f"Request to OpenAI model failed: {str(e)}"
            )
