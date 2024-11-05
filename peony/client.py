import os
import typing

import openai

from peony.resources.beta import beta
from peony.resources.chat import chat


class CustomClient:
    def __init__(self, *, api_key: str | None = None):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key is not provided.")

        openai.api_key = api_key

        self.beta = beta.Beta()
        self.chat = chat.Chat()
