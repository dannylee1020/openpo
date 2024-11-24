import json
import re
from typing import Any, Optional, Union

import numpy as np

from openpo.internal.error import InvalidJSONFormatError


def should_run(prob: float) -> bool:
    if prob == 0.0:
        return False
    if prob == 1.0:
        return True

    return np.random.random() < prob


def clean_text(text: str) -> str:
    """
    Cleans text by removing control characters, zero-width characters,
    and handling Unicode/JSON-specific issues.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # Handle Unicode escapes and JSON-specific cleaning
    text = text.encode("utf-8").decode("unicode-escape")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'(?<!\\)\\(?![\\"{}])', "", text)

    return text


def extract_json(text: str) -> Optional[Any]:
    """
    Extracts JSON from text, cleans it, and returns parsed JSON object.
    """
    try:
        # Find the first { and last } in the text
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return None

        json_str = clean_text(text[start : end + 1])
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise InvalidJSONFormatError(f"Failed to deserialize JSON: {e}")
