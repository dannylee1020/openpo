## Set your API Key
Configure your API key as an environment variable. You can also pass the key into the client.

```bash
# for completion
export HF_API_KEY=<your-huggingface-api-key>
export OPENROUTER_API_KEY=<your-openrouter-api-key>

# for evaluation
export OPENAI_API_KEY=<your-openai-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

## Basic Usage
`model` parameter accepts a model identifier or list of model identifiers.

```python
from openpo import OpenPO

client = OpenPO()

# use single model
response = client.completion.generate(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ]
)

# use multiple models
response = client.completion.generate(
    model=[
        "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
        "huggingface/microsoft/Phi-3.5-mini-instruct"
    ]
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ]
)
```

Use with OpenRouter models:
```python
client = OpenPO()

response = client.completion.generate(
    model=[
        "openrouter/mistralai/mistral-7b-instruct-v0.3",
        "openrouter/microsoft/phi-3.5-mini-128k-instruct",
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ],
)
```

### Inference Endpoint
Call models deployed on HuggingFace Inference Endpoint

```python
response = client.completion.generate(
    model=[
        'huggingface/your-inference-endpoint-1',
        'huggingface/your-inference-endpoint-2',
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": MESSAGE},
    ],
)
```