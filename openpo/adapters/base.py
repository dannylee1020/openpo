from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


class StorageAdapter(ABC):
    @abstractmethod
    def save_feedback(self, feedback: Dict[str, Any]) -> bool:
        """save feedback data to a datastore"""
        pass

    @abstractmethod
    def get_feedback(self, feedback_id: str) -> Dict[str, Any]:
        """retrieve feedback data with id from a datastore"""
        pass

    @abstractmethod
    def get_feedback_all(self) -> Dict[str, Any]:
        """retrieve all feedback in the datastore"""
        pass
