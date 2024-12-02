Start using OpenPO in your code within a minute.

## Set your API Key
Configure your API key of a model of your choice as an environment variable.

```bash
export HF_API_KEY=<your-huggingface-api-key>
export OPENROUTER_API_KEY=<your-openrouter-api-key>
```

## Basic Usage
OpenPO defaults to Hugging Face when provider argument is not set.

```python
import os
from openpo.client import OpenPO

client = OpenPO(api_key="your-huggingface-api-key") # no need to pass in the key if environment variable is already set.

response = client.completions(
    models = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
)
```

To use with OpenRouter, set the provider to `openrouter`

```python
# make request to OpenRouter
client = OpenPO(api_key="<your-openrouter-api-key", provider='openrouter')

response = client.completions(
    models = [
        "qwen/qwen-2.5-coder-32b-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
        "microsoft/phi-3.5-mini-128k-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
)
```

### Inference Endpoint
To call models deployed on Hugging Face Inference Endpoint, simply pass in endpoints to the `models` parameter

```python
response = client.completions(
    models = [
        'your-inference-endpoint-1',
        'your-inference-endpoint-2',
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
)
```