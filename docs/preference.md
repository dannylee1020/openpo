OpenPO conditionally returns two responses based on the value of `diff_frequency` parameter. These outputs can be shown to users to build preference dataset for preference optimization fine-tuning

!!! Note
    OpenPO makes two separate requests to the model, consolidates and returns the response. This will result in more tokens used per request and could incur additional API cost.

## Chat Completion
```python

# streaming is false
response = client.chat.completions.create_preference(
    model=model,
    messages=messages,
    stream=False,
    diff_frequency=0.5,
)

print(response.choices[0].message.content)


# streaming is true
response = client.chat.completions.create_preference(
    model=model,
    messages=messages,
    stream=True,
    diff_frequency=0.5,
)

response_1, response_2 = response
print(response_1.choices[0].message.content)
print(response_2.choices[0].message.content)

```
### Response Type
OpenPO returns `ChatCompletionOutput` or `ChatCompletionStreamOutput` response:

if `stream = False`:
```
type: Union[ChatCompletionOutput, List[ChatCompletionOutput]]
```

if `stream = True`:
```
type: Union[ChatCompletionStreamOutput, List[ChatCompletionStreamOutput]]
```

