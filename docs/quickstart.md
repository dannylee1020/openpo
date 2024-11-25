Start using OpenPO in your code within a minute.

## Set your API Key
Configure your API key of a model of your choice as an environment variable.

```bash
export HF_API_KEY=<your-huggingface-api-key>
export OPENROUTER_API_KEY=<your-openrouter-api-key>
```

## Using Hugging Face
By default, OpenPO client utilizes Hugging Face's [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) to call models available on Huggingface Model Hub.

### Inference API

```python
import os
from openpo.client import OpenPO

client = OpenPO(api_key=os.getenv("HF_API_KEY"))

response = client.chat.completions.create_preference(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5, # generate comparison responses 50% of the time
)

print(res.choices[0].message.content)
```

### Inference Endpoint
To call models that are deployed to HuggingFace Inference Endpoint, simply pass in the private url to the `model` parameter.

```python
client = OpenPO(api_key=os.getenv("HF_API_KEY"))

response = client.chat.completions.create_preference(
    model="<your-inference-endpoint-url",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5, # generate comparison responses 50% of the time
)
```

## Using OpenRouter
OpenPO supports OpenRouter by making requests to its endpoint

```python
client = OpenPO(
    base_url="https://openrouter.ai/api/v1/chat/completions",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client,.chat.completions.create_preference(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5,
)
```