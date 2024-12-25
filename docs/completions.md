Completion is the building block of synthetic data generation. It allows you to easily generate outputs from any LLM of your choice.

## Using API
Easiest way to generate outputs from LLMs is via API. `chat.generate` method provides OpenAI compatible interface to make request to various endpoints to gather outputs.

OpenPO supports various model parameters. You can pass them in as a dictionary to `params`

```python
response = client.completion.generate(
    model="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=messages,
    params = {
        "temperature": 1.0,
        "seed": 42,
        "max_tokens": 1000,
    }
)
```

List of all available model parameter is available in the [parameters section](parameters.md#optional-parameters)


## Using vLLM
For high performance local inference, OpenPO supports vLLM engine. To get started with vLLM, load the model using built-in vLLM class.

!!! Note
    vLLM requires appropriate hardware and GPU to load models and make inference locally.


```python
from openpo import VLLM

llm = VLLM(model="Qwen/Qwen2-0.5B-Instruct")
res = llm.generate(messages=messages)
```

You can configure VLLM instance as well as provide model parameters.

```python
llm = VLLM(
    model="Qwen/Qwen2-0.5B-Instruct",
    tokenizer=tokenizer,
    dtype='bfloat16',
    gpu_memory_utilization=0.95,
)

res = llm.generate(
    messages=messages,
    chat_template=chat_template,
    sampling_params={
        "temperature": 0.8,
        "max_tokens": 1000,
    }
)
```

For more information on parameters, refer to the [API reference](api.md)