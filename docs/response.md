# Response Types

OpenPO returns a custom response from completions that is compatible with OpenAI chat completion response.

## ChatCompletionOutput

```python
class ChatCompletionOutput:
    id: str,
    provider: str,
    model: str,
    object: 'chat.completion' | 'chat.completion.chunk',
    created: int,
    choices: ChatCompletionOutputComplete,
    usage: ChatCompletionOutputUsage
```

```python
class ChatCompletionOutputComplete
    finish_reason: str,
    index: int,
    message: ChatCompletionOutputMessage,
    lobprobs: bool,

```

```python
class ChatCompletionOutputUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

```

## ChatCompletionStreamOutput


```python
class ChatCompletionStreamOutput:
    id: str
    provider: str
    model: str
    object: str
    created: int
    choices: List[ChatCompletionStreamOutputChoice]
```

```python
class ChatCompletionStreamOutputChoice:
    index: int
    delta: ChatCompletionStreamOutputDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
```

```python
class ChatCompletionStreamOutputDelta:
    role: str
    content: str
```


