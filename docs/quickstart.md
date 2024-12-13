Start using OpenPO in your code within a minute.

## Set your API Key
Configure your API key as an environment variable.

```bash
# for completions
export HF_API_KEY=<your-huggingface-api-key>
export OPENROUTER_API_KEY=<your-openrouter-api-key>

# for evaluations
export OPENAI_API_KEY=<your-openai-api-key>
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

## Basic Usage
To call multiple models, simply add models of your choice as a list and pass it to the `completions` method.

!!! Note
    OpenPO requires provider name to be prepended to the model identifier. Refer to the [provider section of the documentation](provider.md)
    for more information.

```python
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

Use with OpenRouter models:
```python
# make request to OpenRouter
client = OpenPO(api_key="<your-openrouter-api-key", provider='openrouter')

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

### Inference Endpoint
To call models deployed on Hugging Face Inference Endpoint, simply pass in endpoints to the `models` parameter

```python
response = client.completions(
    models = [
        'huggingface/your-inference-endpoint-1',
        'huggingface/your-inference-endpoint-2',
    ],
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
)
```