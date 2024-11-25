import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from openpo.resources.chat import chat


class OpenPO:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key must be provided")

        self.base_url = base_url

        # Initialize client based on configuration
        if self.base_url:
            self.client = {
                "base_url": self.base_url,
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            }
        else:
            self.client = {
                "inference_client": InferenceClient(api_key=self.api_key),
                "api_key": self.api_key,
            }

        # Initialize chat resource
        self.chat = chat.Chat(self.client)
