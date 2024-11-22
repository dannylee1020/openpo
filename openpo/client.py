import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from openpo.adapters.base import StorageAdapter
from openpo.resources.chat import chat


class OpenPO:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        storage: Optional[StorageAdapter] = None,
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key must be provided")

        self.base_url = base_url
        self.storage = storage

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

    def save_feedback(self, dest: str, data: List[Dict[str, Any]]) -> bool:
        """Save feedback data to configured storage."""
        if not self.storage:
            raise ValueError("No storage adapter configured")
        return self.storage.save_feedback(dest, data)

    def get_feedback(self, dest: str, feedback_id: str) -> Dict[str, Any]:
        """Retrieve specific feedback by ID."""
        if not self.storage:
            raise ValueError("No storage adapter configured")
        return self.storage.get_feedback(dest, feedback_id)

    def get_all_feedback(self, dest: str) -> List[Dict[str, Any]]:
        """Retrieve all feedback from storage."""
        if not self.storage:
            raise ValueError("No storage adapter configured")
        return self.storage.get_feedback_all(dest)
