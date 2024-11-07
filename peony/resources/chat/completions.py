import json
import random
from typing import Dict, Iterable, List, Optional, Union

import litellm
from pydantic import BaseModel
from typing_extensions import Literal

from peony.internal import prompt


class ExtractModel(BaseModel):
    main_answer: str
    key_points: str


class PreferenceModel(BaseModel):
    first_response: str
    second_response: str | None


class ResponseModel(BaseModel):
    first_response: str


class Completions:
    def create_preference(
        self,
        *,
        model: str,
        messages: list[str],
        response_format: Optional[dict] | None = None,
        temperature: Optional[float] | None = None,
        # openai optional params
        frequency_penalty: Optional[float] | None = None,
        presence_penalty: Optional[float] | None = None,
        # class specific param
        diff_frequency: Optional[float] = 0.0,
    ):
        if random.random() < diff_frequency:
            print("preference")

            core_response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.EXTRACT_PROMPT},
                    *messages[1:],
                ],
                response_format=ExtractModel,
                temperature=0.0,
            )

            core_content = json.loads(core_response.choices[0].message.content)

            pref_response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt.PREF_PROMPT.format(
                            core_content["main_answer"], core_content["key_points"]
                        ),
                    },
                    *messages[1:],
                ],
                response_format=PreferenceModel,
                temperature=1.0,
                presence_penalty=0.6,
                frequency_penalty=0.6,
                drop_params=True,
            )

            return pref_response

        completion = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format or ResponseModel,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            drop_params=True,
        )

        return completion

    def create(
        self,
        *,
        model: str,
        messages: list[str],
        response_format: Optional[dict] | None = None,
    ):
        completions = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format,
        )

        return completions
