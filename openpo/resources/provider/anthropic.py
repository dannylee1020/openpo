import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic as AnthropicClient
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from pydantic import BaseModel

from openpo.internal import prompt as prompt_lib
from openpo.internal.error import AuthenticationError, ProviderError

from .base import LLMProvider


class APIModel(BaseModel):
    q_index: int
    rank: List[int]
    preferred_score: float
    rejected_score: float
    reason: str


class APIResponse(BaseModel):
    evaluation: List[APIModel]


class BatchModel(BaseModel):
    rank: List[int]
    preferred_score: float
    rejected_score: float
    reason: str


class BatchResponse(BaseModel):
    evaluation: List[BatchModel]


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
                "input_schema": APIResponse.model_json_schema(),
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
                max_tokens=8192,
                tools=tools,
                tool_choice={"type": "tool", "name": "build_response_output"},
            )

            return res

        except Exception as e:
            raise ProviderError(
                "Anthropic", message=f"Request to Anthropic model failed: {str(e)}"
            )

    def generate_batch(
        self,
        model: str,
        questions: List[str],
        responses: List[str],
        prompt: str,
    ):
        tools = [
            {
                "name": "build_response_output",
                "description": "build response output with predefined structure",
                "input_schema": BatchResponse.model_json_schema(),
            }
        ]

        tasks = []
        for idx, (q, r) in enumerate(zip(questions, responses)):
            task = Request(
                custom_id=str(idx),
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    system=prompt if prompt else prompt_lib.EVALUATION_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_lib.EVALUATION_QUERY_BATCH.format(q, r),
                        },
                    ],
                    max_tokens=8192,
                    tools=tools,
                    tool_choice={"type": "tool", "name": "build_response_output"},
                ),
            )
            tasks.append(task)

        message_batch = self.client.beta.messages.batches.create(requests=tasks)

        return message_batch
