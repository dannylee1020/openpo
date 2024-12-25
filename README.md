# OpenPO 🐼
[![PyPI version](https://img.shields.io/pypi/v/openpo.svg)](https://pypi.org/project/openpo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-docs.openpo.dev-blue)](https://docs.openpo.dev)

OpenPO simplifies building synthetic dataset with AI feedback and state-of-art evaluation methods.

| Resources | Notebooks |
|----------|----------|
| Building dataset with OpenPO and PairRM  |📔 [Notebook](https://colab.research.google.com/drive/1G1T-vOTXjIXuRX3h9OlqgnE04-6IpwIf?usp=sharing) |
| Using OpenPO with Prometheus 2 | 📔 [Notebook](https://colab.research.google.com/drive/1dro0jX1MOfSg0srfjA_DZyeWIWKOuJn2?usp=sharing) |
| Evaluating with LLM Judge| 📔 [Notebook](https://colab.research.google.com/drive/1_QrmejW2Ym8yzP5RLJbLpVNA_FsEt2ZG?usp=sharing) |


## Key Features

- 🤖 **Multiple LLM Support**: Collect diverse set of outputs from 200+ LLMs

- ⚡ **High Performance Inference**: Native vLLM support for optimized inference

- 🚀 **Scalable Processing**: Built-in batch processing capabilities for efficient large-scale data generation

- 📊 **Research-Backed Evaluation Methods**: Support for state-of-art evaluation methods for data synthesis

- 💾 **Flexible Storage:** Out of the box storage providers for HuggingFace and S3.


## Installation
### Install from PyPI (recommended)
OpenPO uses pip for installation. Run the following command in the terminal to install OpenPO:

```bash
pip install openpo

# to use vllm
pip install openpo[vllm]

# for running evaluation models
pip install openpo[eval]
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

response = client.completion.generate(
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

response = client.completion.generate(
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
response = client.completion.generate(
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
OpenPO offers various ways to synthesize your dataset.


#### LLM-as-a-Judge
To use single judge to evaluate your response data, use `evaluate.eval`

```python
client = OpenPO()

res = openpo.evaluate.eval(
    models=['openai/gpt-4o'],
    questions=questions,
    responses=responses,
)
```

To use multi judge, pass multiple judge models

```python
res_a, res_b = openpo.evaluate.eval(
    models=["openai/gpt-4o", "anthropic/claude-sonnet-3-5-latest"],
    questions=questions,
    responses=responses,
)

# get consensus for multi judge responses.
result = openpo.evaluate.get_consensus(
    eval_A=res_a,
    eval_B=res_b,
)
```
<br>

OpnePO supports batch processing for evaluating large dataset in a cost-effective way.

> [!NOTE]
> Batch processing is an asynchronous operation and could take up to 24 hours (usually completes much faster).

```python
info = openpo.batch.eval(
    models=["openai/gpt-4o", "anthropic/claude-sonnet-3-5-latest"],
    questions=questions,
    responses=responses,
)

# check status
status = openpo.batch.check_status(batch_id=info.id)
```

For multi-judge with batch processing:

```python
batch_a, batch_b = openpo.batch.eval(
    models=["openai/gpt-4o", "anthropic/claude-sonnet-3-5-latest"],
    questions=questions,
    responses=responses,
)

result = openpo.batch.get_consensus(
    batch_A=batch_a_result,
    batch_B=batch_b_result,
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

pm = Prometheus2(model="prometheus-eval/prometheus-7b-v2.0")

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
hf_storage = HuggingFaceStorage()

# push data to repo
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
hf_storage.push_to_repo(repo_id="my-hf-repo", data=preference)

# Load data from repo
data = hf_storage.load_from_repo(path="my-hf-repo")
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
