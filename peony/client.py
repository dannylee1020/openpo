import os
import typing

import openai

from peony.resources.beta import beta
from peony.resources.chat import chat


class CustomClient:
    def __init__(self, *, api_key: str | None = None):
        self.beta = beta.Beta()
        self.chat = chat.Chat()
