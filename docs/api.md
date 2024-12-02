## Client
Simplest way to initialize client is with empty argument.
```python
client = OpenPO()
```

OpenPO takes two optional argument: `provider` and `api_key`. API Key can either be directly passed in to the client, or set as an environment variable:
```
export HF_API_KEY = <your-hf-api-key>
```

If no provider is specified, the client defaults to Hugging Face. To use other provider, the value needs to be passed into `provider` argument.


## Completions
`completions` method has two required arguments:

* `models: List[str]`
* `messages: List[dict]`

All other model parameters are passed into optional `params` argument:

* `params: Optional[Dict[str, Any]]`

<br>

```python
response = client.completions(
    models=["Qwen/Qwen2.5-Coder-32B-Instruct"],
    messages = messages,
    params = {
        "temperature": 0.8,
    }
)
```

