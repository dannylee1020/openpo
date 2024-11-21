import re
from typing import Union

import numpy as np


def should_run(prob: float) -> bool:
    if prob == 0.0:
        return False
    if prob == 1.0:
        return True

    return np.random.random() < prob


def clean_str(input_str: str) -> Union[str, None]:
    if input_str is None:
        return None
    if not isinstance(input_str, str):
        raise TypeError("Input must be a string")

    # Remove control characters except \n \r \t
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", input_str)

    # Remove zero-width characters
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)

    return cleaned
