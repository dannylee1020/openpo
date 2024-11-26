To use structured output, simply pass in pydantic model.

!!! Note
    Structure Output is not supported with streaming.

```python
from pydantic import BaseModel
from openpo.client import OpenPO

client = OpenPO(api_key="your-huggingface-api-key")

class ResponseModel(BaseModel):
    response: str


res = client.chat.completions.create_preference(
    model='mistralai/Mixtral-8x7B-Instruct-v0.1',
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5,
    response_format=ResponseModel,
)
```

For more information on Hugging Face's implementation of structured output, refer [here](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_guidance#constrain-with-pydantic)

!!! Warning
    OpenRouter does not natively support structured output. This makes the response inconsistent and produces error.
    Some smaller models on Hugging Face are also prone to inconsistency when structured output is used.