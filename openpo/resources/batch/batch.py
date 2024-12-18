import json
from typing import List, Optional

from anthropic import Anthropic as AnthropicClient
from openai import OpenAI as OpenAIClient

from openpo.internal.error import AuthenticationError, ProviderError
from openpo.resources.provider import Anthropic, OpenAI


class Batch:
    def __init__(self, client):
        self.client = client
        self.openai_client = OpenAIClient()
        self.anthropic_client = AnthropicClient()

    def eval(
        self,
        model: str,
        questions: List[str],
        responses: List[List[str]],
        prompt: Optional[str] = None,
    ):
        """Use single LLM-as-a-judge method to evaluate responses for building preference data.

        Args:
            model (str): Model identifier to use as a judge. Follows provider/model-identifier format.
            questions (List(str)): Questions for each response pair.
            responses (List[List[str]]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns (Dict): The evaluation data for responses with preferred, rejected, confidence_score and reason.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during evaluation.
            ValueError: If the model format is invalid or provider is not supported.
        """
        try:
            provider = self.client._get_model_provider(model)
            model_id = self.client._get_model_id(model)

            if provider not in ["openai", "anthropic"]:
                raise ProviderError(provider, "Provider not supported for evaluation")

            llm = self.client._get_provider_instance(provider=provider)
            res = llm.generate_batch(
                model=model_id,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )

            return res
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider=provider, message=f"Error during evaluation: {str(e)}"
            )

    def retrieve_status(self, batch_id: str):
        if batch_id.split("_")[0] == "batch":
            status = self.openai_client.batches.retrieve(batch_id)
        else:
            status = self.anthropic_client.beta.messages.batches.retrieve(batch_id)

        return status

    def load_batch(self, filename: str, provider: str):
        data = []
        if provider == "openai":
            res = self.openai_client.files.content(filename)

            for line in res.text.splitlines():
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))

            return data

        if provider == "anthropic":
            res = self.anthropic_client.beta.messages.batches.results(filename)
            for r in res:
                data.append(r)

            return data

    def get_consensus(
        self,
        batch_openai: List,
        batch_anthropic: List,
    ):
        """Get consensus between OpenAI and Anthropic batch results.

        Args:
            batch_openai (List): List of batch results from OpenAI
            batch_anthropic (List): List of batch results from Anthropic

        Returns:
            List: List of evaluation results where both providers agreed on the rank

        Raises:
            Exception: If there's an error processing the batch results
        """
        try:
            # uses dictionary to keep record of index and rank
            # only requires single pass on batch data to reach consensus.
            res = []
            check = {}
            for r in batch_openai:
                try:
                    custom_id = r["custom_id"]
                    content = json.loads(
                        r["response"]["body"]["choices"][0]["message"]["content"]
                    )
                    check[custom_id] = content["evaluation"][0]["rank"]
                except (KeyError, json.JSONDecodeError) as e:
                    continue  # Skip malformed entries

            for r in batch_anthropic:
                try:
                    custom_id = r.custom_id
                    content = r.result.message.content[0].input

                    # Check if same custom_id exists in OpenAI results and ranks match
                    if (
                        custom_id in check
                        and check[custom_id] == content["evaluation"][0]["rank"]
                    ):
                        record = {"q_index": custom_id} | content["evaluation"][0]
                        res.append(record)
                except (KeyError, AttributeError) as e:
                    continue

            return res
        except Exception as e:
            raise Exception(f"Error processing batch results: {str(e)}")
