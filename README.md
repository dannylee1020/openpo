# OpenPO 🐼
[![PyPI version](https://badge.fury.io/py/openpo.svg)](https://badge.fury.io/py/openpo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-docs.openpo.dev-blue)](https://docs.openpo.dev)
![Python](https://img.shields.io/badge/python->=3.10.1-blue.svg)


Streamline LLM Preference Optimization through effortless human feedback collection.

![Demo](./demo/demo.gif)


## What is OpenPO?
OpenPO is an open source library that simplifies the process of collecting, managing, and leveraging human feedback for LLM preference optimization. By automating the comparison of different LLM outputs and gathering human feedback, OpenPO helps developers build better, more fine-tuned language models with minimal effort.

## Key Features

- 🔌 **Multiple LLM Support**: Call any model from HuggingFace and OpenRouter, including popular models like GPT, Claude, Llama, and Mixtral

- 🤝 **OpenAI API Compatibility**: Seamlessly integrate with OpenAI-style client APIs for easy migration and familiar developer experience

- 💾 **Flexible Storage:** Pluggable adapters for your preferred datastore, supporting various data persistence options

- 🎯 **Fine-tuning Ready**: Structured data output ready for immediate model fine-tuning and preference optimization

## Installation
### Install from PyPI (recommended)
OpenPO uses pip for installation. Run the following command in the terminal to install OpenPO:

```bash
pip install openpo
```

### Install from source
Clone the repository first then run the follow command
```bash
cd openpo
poetry install
```

## Getting Started
By default, OpenPO client utilizes HuggingFace's [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) to call models available on HuggingFace Model Hub.

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

You can pass in a dictionary to `pref_params` argument to control the randomness of a second response when comparison logic is called. Currently supported parameters are: `temperature`, `frequency_penalty` and `presence_penalty`.

```python
response = client.chat.completions.create_preference(
    model="anthropic/claude-3.5-sonnet:beta",
    message=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ],
    diff_frequency=0.5,
    pref_params={"temperature": 1.5, "frequency_penalty": 0.5},
)
```

### Saving Data
Use providers to easily upload and download data.

```python
import os
from openpo.client import OpenPO
from openpo.providers.huggingface import HuggingFaceStorage

storage = HuggingFaceStorage(repo_id="my-dataset-repo", api_key="hf-token")
client = OpenPO(api_key="your-huggingface-token")

preference = {} # preference data needs to be in the format {"prompt": ..., "preferred": ..., "rejected": ...} for finetuning
storage.save_data(data=preference, key="my-data")
```

## Structured Outputs (JSON Mode)
OpenPO supports structured outputs using Pydantic model.

> [!NOTE]
> OpenRouter does not natively support structured outputs. This leads to inconsistent behavior from some models when structured output is used with OpenRouter.
>
> It is recommended to use HuggingFace models for structured output.


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

## Try Out
`docker compose up --build` to run simple demo of how it works in the UI.
