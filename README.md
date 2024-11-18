# OpenPO
Streamline LLM Preference Optimization through effortless user feedback collection

![Demo](./demo/demo.gif)

## What is OpenPO?
OpenPO is a lightweight Python package that simplifies the process of collecting, managing, and leveraging user feedback for LLM preference optimization. By automating the comparison of different LLM outputs and gathering user feedback, OpenPO helps developers build better, more fine-tuned language models with minimal effort.

## Key Features

- 🤝 **API Compatibility**: Seamlessly integrate with OpenAI-style client APIs

- 🔌 **Multiple LLM Support**: Works with OpenAI and Anthropic models (more coming soon!)

- 💾 **Flexible Storage:** Extensible adapter system for your preferred datastore

- 🎯 **Fine-tuning Ready**: Structured data output ready for immediate model fine-tuning

## Installation
OpenPO uses pip for installation. Run the following command to install OpenPO in your terminal:

```bash
pip install openpo
```


## Getting Started
```python
import os
from openpo.client import OpenPO

# set api key as env variable
os.environ['OPENAI_API_KEY'] = "your-api-key"

client = OpenPO()

response = client.chat.completions.create_preference(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5, # generate comparison responses 50% of the time
)

print(res.choices[0].message.content)
```

### With Storage Adapter
```python
import os
from openpo.client import OpenPO
from openpo.adapters import postgres as pg

pg_adapter = pg.PostgresAdapter(
    host=host,
    dbname=dbname,
    port=port,
    user=user,
    pw=pw
)

client = OpenPO(storage=pg_adapter)

preference = {} # preference data needs to be in the format {"prompt": ..., "preferred": ..., "rejected": ...}
client.save_feedback(dest='destination', data=preference)
```

## Structured Outputs (JSON Mode)
OpenPO supports structured outputs using Pydantic model.
```python
from pydantic import BaseModel
from openpo.client import OpenPO

client = OpenPO()

class SingleResponse(BaseModel):
    response: str

class PreferenceResponse(BaseModel):
    first_response:str
    second_response:str

res = client.chat.completions.create_preference(
    model='gpt-4o',
    messages=[
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": MESSAGE},
    ],
    diff_frequency=0.5,
    response_format=SingleResponse, # defines structure for a single response
    pref_response_format=PreferenceResponse, # defines structure for comparing responses.
)

print(res.choices[0].message.content)
```


## Try Out
`docker compose up --build` to run simple demo of how it works in the UI.