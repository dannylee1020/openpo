import os
from typing import Any, Dict, List, Optional

from .resources.provider.huggingface import HuggingFace
from .resources.provider.openrouter import OpenRouter

ENV_MAPPING = {
    "huggingface": "HF_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class OpenPO:
    def __init__(
        self,
        provider: Optional[str] = "huggingface",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{ENV_MAPPING[provider]}")
        if not self.api_key:
            raise ValueError("No API key is provided")

    def completions(
        self,
        models: List[str],
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
    ):
        if self.provider == "huggingface":
            llm = HuggingFace(api_key=self.api_key)
            res = llm.generate(models=models, messages=messages, params=params)

            return res
        else:
            llm = OpenRouter(api_key=self.api_key)
            res = llm.generate(models=models, messages=messages, params=params)

            return res
