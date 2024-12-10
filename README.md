# OpenPO 🐼
[![PyPI version](https://img.shields.io/pypi/v/openpo.svg)](https://pypi.org/project/openpo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-docs.openpo.dev-blue)](https://docs.openpo.dev)
![Python](https://img.shields.io/badge/python->=3.10.1-blue.svg)


OpenPO simplifies building synthetic datasets for preference tuning from 200+ LLMs.

| Resources | Notebooks |
|----------|----------|
| Building dataset with OpenPO and PairRM  |📔 [Notebook](https://colab.research.google.com/drive/1G1T-vOTXjIXuRX3h9OlqgnE04-6IpwIf?usp=sharing) |


## What is OpenPO?
OpenPO is an open source library that simplifies the process of building synthetic datasets for LLM preference tuning. By collecting outputs from 200 + LLMs and evaluating them using research-proven methodologies, OpenPO helps developers build better, more fine-tuned language models with minimal effort.

## Key Features

- 🔌 **Multiple LLM Support**: Call 200+ models from HuggingFace and OpenRouter

- 🧪 **Research-Backed Methodologies**: Implementation of methodologies for data synthesis from latest research papers.

- 🤝 **OpenAI API Compatibility**: Support for OpenAI API format

- 💾 **Flexible Storage:** Out of the box storage providers for HuggingFace and S3.


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
set environment variable first
```bash
export HF_API_KEY=<your-api-key>
export OPENROUTER_API_KEY=<your-api-key>
```
To get started, simply pass in a list of model names of your choice

> [!NOTE]
> OpenPo requires provider name to be prepended to the model identifier.

```python
import os
from openpo.client import OpenPO

client = OpenPO()

response = client.completions(
    models = [
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
        "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
        "huggingface/microsoft/Phi-3.5-mini-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
)
```

You can also call models with OpenPO.

```python
# make request to OpenRouter
client = OpenPO()

response = client.completions(
    models = [
        "openrouter/qwen/qwen-2.5-coder-32b-instruct",
        "openrouter/mistralai/mistral-7b-instruct-v0.3",
        "openrouter/microsoft/phi-3.5-mini-128k-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],

)
```

OpenPO takes default model parameters as a dictionary. Take a look at the documentation for more detail.

```python
response = client.completions(
    models = [
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
        "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
        "huggingface/microsoft/Phi-3.5-mini-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    params={
        "max_tokens": 500,
        "temperature": 1.0,
    }
)

```

### Storing Data
Use out of the box storage class to easily upload and download data.

```python
from openpo.storage.huggingface import HuggingFaceStorage
hf_storage = HuggingFaceStorage(repo_id="my-dataset-repo")

# push data to repo
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
hf_storage.push_to_repo(data=preference)

# Load data from repo
data = hf_storage.load_from_repo()
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

client = OpenPO()

class ResponseModel(BaseModel):
    response: str


res = client.completions(
    models=["huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    params = {
        "response_format": ResponseFormat,
    }
)
```

## Contributing
Contributions are what makes open source amazingly special! Here's how you can help:

### Development Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/openpo.git
cd openpo
```

1. Install Poetry (dependency management tool)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

1. Install dependencies
```bash
poetry install
```

### Development Workflow
1. Create a new branch for your feature
```bash
git checkout -b feature-name
```

2. Submit a Pull Request
- Write a clear description of your changes
- Reference any related issues
