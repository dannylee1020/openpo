# OpenPO
Streamline LLM Preference Optimization through effortless human feedback collection

![Demo](./demo/demo.gif)

## What is OpenPO?
OpenPO is an open source library that simplifies the process of collecting, managing, and leveraging human feedback for LLM preference optimization. By automating the comparison of different LLM outputs and gathering human feedback, OpenPO helps developers build better, more fine-tuned language models with minimal effort.

## Key Features


- 🔌 **Multiple LLM Support**: Works with HuggingFace and OpenRouter out of the box

- 🤝 **OpenAI API Compatibility**: Seamlessly integrate with OpenAI-style client APIs

- 💾 **Flexible Storage:** Pluggable adapters for your preferred datastore

- 🎯 **Fine-tuning Ready**: Structured data output ready for immediate model fine-tuning

## Installation
OpenPO uses pip for installation. Run the following command to install OpenPO in your terminal:

```bash
pip install openpo
```


## Getting Started
By default, OpenPO client utilizes Huggingface's [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) to call models available on Huggingface Model Hub.

```python
import os
from openpo.client import OpenPO

client = OpenPO(api_key="your-huggingface-api-key")

response = client.chat.completions.create_preference(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5, # generate comparison responses 50% of the time
)

print(response.choices[0].message.content)
```

OpenPO also works with OpenRouter.

```python
# make request to OpenRouter
import os
from openpo.client import OpenPO

client = OpenPO(
    api_key='your-openrouter-api-key',
    base_url="https://openrouter.ai/api/v1/chat/completions"
)

response = client.chat.completions.create_preference(
    model="anthropic/claude-3.5-sonnet:beta",
    message=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ],
    diff_frequency=0.5
)

print(response.choices[0].message.content)
```

### With Storage Adapter
Client can be initialized with a storage adapter to save preference data in a datastore.

```python
import os
from openpo.client import OpenPO
from openpo.adapters import postgres as pg

pg_adapter = pg.PostgresAdapter(
    host=host,
    dbname=dbname,
    port=port,
    user=user,
    pw=pw
)

client = OpenPO(api_key="your-huggingface-token", storage=pg_adapter)

preference = {} # preference data needs to be in the format {"prompt": ..., "preferred": ..., "rejected": ...}
client.save_feedback(dest='destination', data=preference)
```

## Structured Outputs (JSON Mode)
OpenPO supports structured outputs using Pydantic model.

```python
from pydantic import BaseModel
from openpo.client import OpenPO

client = OpenPO(api_key="your-huggingface-api-key")

class SingleResponse(BaseModel):
    response: str

class PreferenceResponse(BaseModel):
    first_response:str
    second_response:str

res = client.chat.completions.create_preference(
    model='mistralai/Mixtral-8x7B-Instruct-v0.1',
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5,
    response_format=SingleResponse, # defines structure for a single response
    pref_response_format=PreferenceResponse, # defines structure for comparing responses.
)

print(res.choices[0].message.content)
```


## Try Out
`docker compose up --build` to run simple demo of how it works in the UI.
