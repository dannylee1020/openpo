import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI as OpenAIClient
from openai import OpenAIError
from openai.lib._pydantic import to_strict_json_schema
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
        prompt: Optional[str],
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
                response_format=APIResponse,
                max_tokens=8192,
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

    def generate_batch(
        self,
        model: str,
        questions: List[str],
        responses: List[List],
        prompt: Optional[str],
    ):
        tasks = []
        for idx, (q, r) in enumerate(zip(questions, responses)):
            task = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                prompt if prompt else prompt_lib.EVALUATION_PROMPT
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt_lib.EVALUATION_QUERY_BATCH.format(q, r),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response_schema",
                            "description": "schema for response format",
                            "strict": True,
                            "schema": to_strict_json_schema(BatchResponse),
                        },
                    },
                },
            }

            tasks.append(task)

        # write data to jsonl
        data_dir = Path.home() / ".openpo"
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        filename = f"{data_dir}/evaluation_batch_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"
        with open(filename, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        # upload file to OpenAI
        batch_file = self.client.files.create(
            file=open(filename, "rb"),
            purpose="batch",
        )

        # create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_job
