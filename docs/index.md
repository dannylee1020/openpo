#👋🏻 Welcome
OpenPO simplifies collecting preference data by generating pairwise responses from 200+ LLMs.

## Key Features

- 🔌 **Multiple LLM Support**: Call any model from HuggingFace and OpenRouter

- 🤝 **OpenAI API Compatibility**: Seamlessly integrate with OpenAI-style client APIs

- 💾 **Flexible Storage:** Pluggable adapters for your preferred datastore

- 🎯 **Fine-tuning Ready**: Structured data output ready for immediate model fine-tuning


## How It Works
1. Makes request to any model on Hugging Face and OpenRouter.
2. Generates two different responses for user feedback, building preference dataset.
3. Preference dataset is used to finetune models.

## What is Preference Optimization?
Preference Optimization is a method to improve AI models based on human feedback about which outputs are better. Think of it as having a cooking show where judges taste two dishes and pick the better one - over time, the chef (AI model) learns what people prefer and gets better at cooking. By collecting preference data from humans and fine-tuning the model, models become better at capturing nuanced judgements and better align its output with what humans find desirable.

### Why It Matters
It helps models:

- Personalize to user's preference
- Improve overall performance of models
- Generate safer and more appropriate responses


OpenPO makes it easy to collect this valuable preference data - the first step in improving your own AI models.
