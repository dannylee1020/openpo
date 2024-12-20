To use structured output, use Pydantic model.

```python
from pydantic import BaseModel
from openpo import OpenPO

openpo = OpenPO()

class ResponseModel(BaseModel):
    response: str


res = openpo.completions(
    models=["huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    params = {
        "response_format": ResponseModel,
    }
)
```

For more information on Hugging Face's implementation of structured output, refer [here](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_guidance#constrain-with-pydantic)

!!! Warning
    OpenRouter does not natively support structured output. This makes the response inconsistent and produces error.
    Some smaller models on Hugging Face are also prone to inconsistency when structured output is used.