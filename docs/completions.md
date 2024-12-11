`completions` method is responsible for collecting responses from multiple LLMs from HuggingFace and OpenRouter.

## Using Single Model

To use single model, simply pass in one model to the `models` argument

```python
response = client.completions(
    models=["huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages=messages
)
```
<br>
If you want to get multiple responses from the same model, pass in the same model multiple times

```python
response = client.completions(
    models=[
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    ],
    messages=messages,
    params = {
        "pref_params": [
            {
                "temperature": 1.3,
                "frequency_penalty": 0.3,
            },
        ],
    }
)
```
!!! TIP
    Adjusting `temperature` and `frequency_penalty` in `pref_params` argument will vary the response from the second model.

## Using Multiple Models

Using multiple models is as simple as passing in different model names

```python
response = client.completions(
    models=[
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
        "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
        "huggingface/microsoft/Phi-3.5-mini-instruct",
    ],
    messages=messages,
)
```


!!! NOTE
    `completions` is a synchronous operation. Request to multiple models will happen sequentially. To call models asynchronously, use OpenPO async client (coming soon!)

## Optional Model Parameters
`completions` method takes dictionary of optional model parameters. For the list available parameters, take a look at [parameters section](parameters.md#optional-parameters).

```python
response = client.completions(
    models=[ "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages=messages,
    params={
        "max_tokens": 1000,
        "temperature": 1.0,
    }
)
```