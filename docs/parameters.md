Following is available model parameters for completion.

## Completion using API

### Required Parameters

```title="model (str, list[str])"
* Required
```
Specifies the model to use for text generation. Can be a HuggingFace or OpenRouter model

<br>

```title="messages (list[dict])"
* Required
```
messages to send to model. OpenAI format required: `{"role", "content"}`

### Optional Parameters

```title="frequency_penalty (float)"
* Optional, -2.0 to 2.0
* Default = 0.0
```
Reduces repetitive text by applying a penalty to tokens based on how frequently they've appeared in the generated text.

<br>

```title="logit_bias (list[float])"
* Optional, -100 to 100
* Default = None
```
Modifies token generation probability by directly adjusting their logit scores. Useful for controlling specific token appearances.

<br>

```title="logprobs (bool)"
* Optional
* Default = False
```
Enables the return of log probabilities for generated tokens, useful for analyzing model confidence.

<br>

```title="max_tokens (int)"
* Optional
* Default = 20
```
Limits the length of the model's response by setting a maximum token count.

<br>

```title="presence_penalty (float)"
* Optional, -2.0 to 2.0
* Default = None
```
Influences topic diversity by penalizing tokens based on their presence in the text so far.

<br>

```title="response_format (PydanticModel)"
* Optional
```
Defines structural constraints for the output. Pydantic model is required.

<br>

```title="seed (int)"
* Optional
* Default = None
```
Enables reproducible outputs by setting a fixed random seed for generation.

<br>

```title="stop (str)"
* Optional
* Default = None
```
Defines up to 4 sequences that will cause the model to stop generating further tokens when encountered.


<br>

```title="temperature (float)"
* Optional, 0.0 to 2.0
* Default = 1.0
```
Controls response randomness - lower values produce more focused and deterministic outputs, while higher values increase creativity.

<br>

```title="top_logprobs (int)"
* Optional, 0 to 5
```
Returns the most probable token alternatives at each position with their probabilities when logprobs is enabled.

<br>

```title="top_p (float)"
* Optional, 0.0 to 1.0
* Default = 1.0
```
Controls response diversity by sampling from the most likely tokens that sum to the specified probability.

<br>

```title="tool_choice (str)"
* Optional
* Default = "auto"
```
Determines which tool the model should use for completion generation.

<br>

```title="tool_prompt (str)"
* Optional
* Only available for HuggingFace models
```
Provides additional context or instructions to be prepended before tool-related prompts.

<br>

```title="tools (list)"
* Optional
```
A list of functions that the model may use to generate structured outputs

<br>

```title="pref_params (list[dict[str, float]])"
* Optional
```
Extra parameters to control the model outputs from the model list except the first one . Currently supports `temperature` and `frequency_penalty`

<br>

## Completion using vLLM

Below is the list of parameters for vLLM configuration and its model.


### LLM Engine
[LLM configuration](https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html#vllm.LLM)

### Chat Parameter
[Chat](https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html#vllm.LLM.chat)

### Sampling Parameters
[Sampling Parameters](https://docs.vllm.ai/en/stable/dev/sampling_params.html#sampling-parameters)