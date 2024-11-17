from typing import Any, Dict, List, Optional

from openpo.adapters.base import StorageAdapter
from openpo.resources.chat import chat


class OpenPO:
    def __init__(self, storage: Optional[StorageAdapter] = None):
        self.chat = chat.Chat()
        self.storage = storage

    def save_feedback(self, dest: str, data: List[Dict[str, Any]]) -> bool:
        if not self.storage:
            raise ValueError("No storage adapter configured")

        return self.storage.save_feedback(dest, data)

    def get_feedback(self, dest: str, feedback_id: str):
        if not self.storage:
            raise ValueError("No storage adapter configured")

        return self.storage.get_feedback(dest, feedback_id)

    def get_all_feedback(self, dest: str):
        if not self.storage:
            raise ValueError("No storage adapter configured")

        return self.storage.get_feedback_all(dest)
