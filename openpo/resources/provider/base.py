from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        kwargs: Optional[Dict[str, Any]],
    ):
        pass
