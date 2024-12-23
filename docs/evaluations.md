Evaluation is the magic that synthesizes outputs data into finetuned ready dataset. To get started with evaluation, first install extra dependencies.

```bash
pip install openpo[eval]
```

## LLM-as-a-Judge
Since [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) from Anthropic established a groundwork for using AI feedback beyond just ensuring model safety, [Subsequent researches](paper.md) have demonstrated the effectiveness of LLM-based evaluation for synthetic data generation, potentially offering a scalable alternative to human annotation.

OpenPO adopts LLM-as-a-Judge methodology, supporting both single and multi-judge configurations to generate high-quality dataset.


### Using API
Simplest way to run evaluation is to make request to the models using `eval` method. To use a single model as a judge, pass a model into the models parameter.

!!! Note
    Evaluation currently suppports OpenAI and Anthropic models only.

```python
from openpo import OpenPO

openpo = OpenPO()

responses = [
    ["Lorem ipsum dolor sit amet", "consectetur adipiscing elit"],
    [" Aliquam pharetra neque", "ultricies elit imperdiet laoreet"],
]

res = openpo.evaluate.eval(
    models=['openai/gpt-4o'],
    questions=questions,
    responses=data,
)
```
<br>
For more robust method, OpenPO supports multi-judge, where more than one model is used as a judge to reach consensus. Simply pass in multiple models for this.

The responses that judges disagree will be discarded and only the responses that reach agreement will be returned.

```python
res = openpo.evaluate.eval(
    models=['openai/gpt-4o', 'anthropic/claude-sonnet-3-5-latest'],
    questions=questions,
    responses=responses,
)
```
<br>
If you want more control over the behavior of judge models, use custom prompt.

```python
res = openpo.eval_single(
    model=['openai/gpt-4o'],
    questions=questions,
    responses=responses,
    prompt=prompt,
)

```

For more details, take a look at the [API reference](api.md)

### Using Batch Processing
Using API to evaluate large volume of dataset can be costly and inefficient. Batch processing provides cost-effective way to handle tasks that do not require immediate responses.

Running batch processing is similar to how evaluation with API works. To use single model evaluation, pass the model identifier to the models parameter.

```python
openpo = OpenPO()

# run batch processing
batch_info = openpo.batch.eval(
    models=['openai/gpt-4o-mini'],
    questions=questions,
    responses=responses,
)

# check batch status
status = openpo.batch.check_status(batch_info[0].id)

# load batch result
result = openpo.batch.load_batch(
    filename=status.output_file_id,
    provider='openai',
)
```

Using multi-judge with batch processing is very similar:

```python
batch_a_info, batch_b_info = openpo.batch.eval(
    models=["openai/gpt-4o-mini", "anthropic/anthropic/claude-3-5-haiku-20241022"],
    questions=questions,
    responses=responses,
)

# load batch results once done
batch_a = openpo.batch.load_batch(filename=batch_a_info.output_file_id, provider='openai')
batch_b = openpo.batch.load_batch(filename=batch_b_info.id, provider'anthropic')

# get consensus
result = openpo.batch.get_consensus(
    batch_A=batch_a,
    batch_B=batch_b,
)

```

OpenAI and Anthropic provide UI for batch processing jobs. Visit their console to find more information such as status, metadata and more.


##  Evaluation Models
!!! Note
    Evaluation Models require to run on appropriate hardware with GPU and memory to make inference.

Fine-tuned evaluation models offer an efficient approach to synthesizing response datasets. These specialized models are trained on diverse data sources from both human and AI feedback, enabling accurate comparison and ranking of model responses.


To run inference with evaluation models, first install extra dependencies by running:


### PairRM
[Pairwise Reward Model for LLM (PairRM)](https://arxiv.org/abs/2306.02561) is an evaluation model specifically designed to assess and compare pairwise responses from LLMs. The model uses DeBERTa ([He et al., 2021](https://arxiv.org/abs/2006.03654)) as the base model, trained and evaluated on MixInstruct dataset. It is a very lightweight model with high accuracy on par with GPT-4.

| Model | Parameter | Size |
|:--- |:---- |:--- |
| [PairRM](https://huggingface.co/llm-blender/PairRM) | 0.4B | 1.7GB |

To run evaluation for pairwise ranking:

```python
from openpo import PairRM

pairrm = PairRM()
res = pairrm.eval(prompts, responses)
```

For full example of how to use PairRM, check out the [tutorial notebook](notebook.md)!

### Prometheus 2
[Prometheus 2](https://arxiv.org/abs/2405.01535) is an open source language model specialized in evaluations of other language models. It is aimed to solve shortcomings of existing proprietary models such as transparency, controllability and affordability.

The model uses Mistral-7B and Mixtral-8x7B as the base models, and uses two types of datasets: preference collection and feedback collection to train models on both direct assessment and pairwise ranking.

| Model | Parameter | Size  |
| :----------- | :------------- | :------------- |
| [prometheus-7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) | 7.2B | 13GB |
| [prometheus-8x7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0) | 46.7B | 90GB|


For pairwise ranking:

```python
from openpo import Prometheus2
from openpo.resources.provider import VLLM

model = VLLM<(model="prometheus-eval/prometheus-7b-v2.0")
pm = Prometheus2(model=model)

feedback = pm.eval_relative(
    instructions=instructions,
    responses_A=response_A,
    responses_B=response_B,
    rubric='reasoning',
)
```

For direct assessment:

```python
model = VLLM<(model="prometheus-eval/prometheus-7b-v2.0")
pm = Prometheus2(model=model)

feedback = pm.eval_absolute(
    instructions=instructions,
    responses=responses,
    rubric='reasoning',
)
```
You can choose different rubric depending on the evaluation goal. Prometheus2 supports [five rubrics](api.md/#prometheus-2) out of the box.

OpenPO also lets you use customized rubric:

```python
feedback = pm.eval_absolute(
    instructions=instructions,
    responses=responses,
    rubric='some-customized-rubric-template',
)
```

For full example of how to use Prometheus 2, check out the [tutorial notebook](notebook.md)!

For more information about the model, refer to their [github repository](https://github.com/prometheus-eval/prometheus-eval)









