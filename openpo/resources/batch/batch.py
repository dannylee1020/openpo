import json
from typing import Dict, List, Optional

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
        models: List[str],
        questions: List[str],
        responses: List[List[str]],
        prompt: Optional[str] = None,
    ):
        """Use input model as a judge to evaluate responses.

        Args:
            models (str): List of model identifiers to use as a judge. Follows provider/model-identifier format.
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
            result = []
            for m in models:
                provider = self.client._get_model_provider(m)
                model_id = self.client._get_model_id(m)

                if provider not in ["openai", "anthropic"]:
                    raise ProviderError(
                        provider, "Provider not supported for evaluation"
                    )

                llm = self.client._get_provider_instance(provider=provider)
                res = llm.generate_batch(
                    model=model_id,
                    questions=questions,
                    responses=responses,
                    prompt=prompt if prompt else None,
                )

                result.append(res)
            return result
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider=provider, message=f"Error during evaluation: {str(e)}"
            )

    def check_status(self, batch_id: str):
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
        batch_A: List,
        batch_B: List,
    ) -> List[Dict]:
        """Reach consensus between two batch results.

        Args:
            batch_A (List): List of batch results to compare
            batch_B (List): List of batch results to compare

        Returns:
            List[Dict]: List of evaluation results where both providers agree on

        Raises:
            Exception: If there's an error processing the batch results
        """
        try:
            # uses dictionary to keep record of index and rank
            # only requires single pass on batch data to reach consensus.
            res = []
            check = {}
            for r in batch_A:
                # check if batch is from openai
                if isinstance(r, dict):
                    custom_id = r["custom_id"]
                    content = json.loads(
                        r["response"]["body"]["choices"][0]["message"]["content"]
                    )
                else:
                    custom_id = r.custom_id
                    content = r.result.message.content[0].input

                check[custom_id] = content["evaluation"][0]["rank"]

            for r in batch_B:
                if isinstance(r, dict):
                    custom_id = r["custom_id"]
                    content = json.loads(
                        r["response"]["body"]["choices"][0]["message"]["content"]
                    )
                else:
                    custom_id = r.custom_id
                    content = r.result.message.content[0].input

                if (
                    custom_id in check
                    and check[custom_id] == content["evaluation"][0]["rank"]
                ):
                    record = {"q_index": custom_id} | content["evaluation"][0]
                    res.append(record)

            return res
        except Exception as e:
            raise Exception(f"Error processing batch results: {str(e)}")
