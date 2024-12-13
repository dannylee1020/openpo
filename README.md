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
OpenPO is an open source library that simplifies the process of building synthetic datasets for LLM preference tuning. By collecting outputs from 200 + LLMs and synthesizing them using research-proven methodologies, OpenPO helps developers build better, more fine-tuned language models with minimal effort.

## Key Features

- 🤖 **Multiple LLM Support**: Collect diverse set of outputs from 200+ LLMs

- 📊 **Research-Backed Evaluation Methods**: Support for state-of-art evaluation methods for data synthesis

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
set your environment variable first

```bash
# for completions
export HF_API_KEY=<your-api-key>
export OPENROUTER_API_KEY=<your-api-key>

# for evaluations
export OPENAI_API_KEY=<your-openai-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

### Completion
To get started with collecting LLM responses, simply pass in a list of model names of your choice

> [!NOTE]
> OpenPO requires provider name to be prepended to the model identifier.

```python
import os
from openpo import OpenPO

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

You can also call models with OpenRouter.

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

### Evaluation
OpenPO offers various ways to synthesize your dataset. To run evaluation, install extra dependencies by running

```bash
pip install openpo[eval]
```

#### LLM-as-a-Judge
To use single judge to evaluate your response data, use `eval_single`

```python
client = OpenPO()

res = openpo.eval_single(
    model='openai/gpt-4o',
    data=responses,
)
```

To use multi judge, use `eval_multi`

```python
res = openpo.eval_multi(
    models=["openai/gpt-4o", "anthropic/claude-sonnet-3-5-latest"],
    data=responses,
)
```

#### Pre-trained Models
You can use pre-trained open source evaluation models. OpenPo currently supports two types of models: `PairRM` and `Prometheus2`.

> [!NOTE]
> Appropriate hardware with GPU and memory is required to make inference with pre-trained models.

To use PairRM to rank responses:

```python
from openpo import PairRM

pairrm = PairRM()
res = pairrm.eval(prompts, responses)
```

To use Prometheus2:

```python
from openpo import Prometheus2
from openpo.resources.provider.vllm import VLLM

model = VLLM<(model="prometheus-eval/prometheus-7b-v2.0")
pm = Prometheus2(model=model)

feedback = pm.eval_relative(
    instructions=instructions,
    responses_A=response_A,
    responses_B=response_B,
    rubric='reasoning',
)
```



### Storing Data
Use out of the box storage class to easily upload and download data.

```python
from openpo.storage import HuggingFaceStorage
hf_storage = HuggingFaceStorage(repo_id="my-dataset-repo")

# push data to repo
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
hf_storage.push_to_repo(data=preference)

# Load data from repo
data = hf_storage.load_from_repo()
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
