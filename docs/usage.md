## Using Single Model
To use single model, simply pass in one model to the `models` argument

```python
response = client.completions(
    models=["Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages=messages
)
```
<br>
If you want to get multiple responses from the same model, pass in the same model multiple times

```python
response = client.completions(
    models=[
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
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
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    messages=messages,
)
```


!!! NOTE
    `completions` is a synchronous operation. Request to multiple models will happen sequentially. To call models asynchronously, use OpenPO async client (coming soon!)
