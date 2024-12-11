Evaluation methods are what synthesizes data to create finetuned ready datasets.

## Using LLM-as-a-Judge
Since [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) from Anthropic established a groundwork for using AI feedback beyond just ensuring model safety, [Subsequent researches](resource.md/#research) have demonstrated the effectiveness of LLM-based evaluation for synthetic data generation, potentially offering a scalable alternative to human annotation.

OpenPO adopts what is called LLM-as-a-Judge methodology, supporting both single and multi-judge configurations to generate high-quality dataset.


### Usage
To use a single LLM as a judge, you can use `eval_single` method.

!!! Note
    Evaluation currently suppports OpenAI and Anthropic models only.

```python
client = OpenPO()

data = [
    ["Lorem ipsum dolor sit amet", "consectetur adipiscing elit"],
    [" Aliquam pharetra neque", "ultricies elit imperdiet laoreet"],
]

res = openpo.eval_single(
    model='openai/gpt-4o',
    data=data,
)
```
<br>
For more robust method, OpenPO supports multi-judge, where more than one model is used as a judge to reach consensus. You can use `eval_multi` method for this.

The responses that judges disagree will be discarded and only the responses that reach agreement by the judges will be returned.

```python
res = openpo.eval_multi(
    models=['openai/gpt-4o', 'anthropic/claude-sonnet-3-5-latest'],
    data=data
)
```
<br>
If you want more control over the behavior of judge models, use custom prompt.

```python
res = openpo.eval_single(
    model='openai/gpt-4o',
    data=data,
    prompt=prompt,
)

```

For more details, take a look at the [API reference](api.md)



## Using Finetuned Evaluation Models

Fine-tuned evaluation models offer an efficient approach to synthesizing response datasets. These specialized models are trained on diverse data sources from both human and AI feedback, enabling accurate comparison and ranking of model responses.

### PairRM (feature coming soon)
[Pairwise Reward Model for LLM (PairRM)](https://arxiv.org/abs/2306.02561) is an evaluation model specifically designed to assess and compare pairwise responses from LLMs. The model uses DeBERTa ([He et al., 2021](https://arxiv.org/abs/2006.03654)) as the base model, trained and evaluated on MixInstruct dataset.


### Prometheus 2 (feature coming soon)
[Prometheus 2](https://arxiv.org/abs/2405.01535) is an open source language model specialized in evaluations of other language models. It is aimed to solve shortcomings of existing proprietary models such as transparency, controllability and affordability.

The model uses Mistral-7B and Mixtral-8x7B as the base models, and uses two types of datasets: preference collection and feedback collection to train models on both direct assessment and pairwise ranking.







