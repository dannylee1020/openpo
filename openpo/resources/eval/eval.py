import json
from typing import Dict, List, Optional, Union

from openpo.internal.error import AuthenticationError, ProviderError
from openpo.resources.provider import Anthropic, OpenAI


class Evaluation:
    def __init__(self, client):
        self.client = client

    def _get_model_consensus(
        self,
        res_a: List[Dict],
        res_b: List[Dict],
    ) -> List[int]:
        matching_indices = []
        for i, (a, b) in enumerate(zip(res_a, res_b)):
            if a.get("q_index") == b.get("q_index") and a["rank"] == b["rank"]:
                matching_indices.append(a.get("q_index", i))

        return matching_indices

    def _validate_provider(self, provider: str) -> None:
        if provider not in ["openai", "anthropic"]:
            raise ProviderError(provider, "Provider not supported for evaluation")

    def _parse_response(self, response) -> List[Dict]:
        try:
            if "chatcmpl" in response.id:
                return json.loads(response.choices[0].message.content)["evaluation"]
            return response.content[0].input["evaluation"]
        except Exception as e:
            raise Exception(f"Error parsing model responses: {e}")

    def eval(
        self,
        models: List[str],
        questions: List[str],
        responses: List[List[str]],
        prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Evaluate responses using either single or multiple LLMs as judges.

        Args:
            models (Union[str, List[str]]): List of  model identifier or list of models to use as judges. Follows provider/model-identifier format.
            questions (List[str]): Questions for each response pair.
            responses (List[List[str]]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns:
            List[Dict]: The evaluation data for responses. Response returns preferred, rejected, confidence_score and reason.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during evaluation.
            ValueError: If the model format is invalid or required models are missing.
        """
        try:
            eval_res = []
            for m in models:
                provider = self.client._get_model_provider(m)
                model_id = self.client._get_model_id(m)

                self._validate_provider(provider)

                llm = self.client._get_provider_instance(provider=provider)
                res = llm.generate(
                    model=model_id,
                    questions=questions,
                    responses=responses,
                    prompt=prompt if prompt else None,
                )
                eval_res.append(res)

            return eval_res

        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider="", message=f"Error during evaluation: {str(e)}"
            )

    def get_consensus(self, eval_A: List, eval_B: List):
        """Reach consensus between two evaluation results

        Args:
            eval_A (List): List of batch results to compare
            eval_B (List): List of batch results to compare

        Returns:
            List: List of evaluation results where both providers agreed on the rank

        Raises:
            Exception: If there's an error processing the batch results
        """
        try:
            parsed_a = self._parse_response(
                response=eval_A,
            )
            parsed_b = self._parse_response(
                response=eval_B,
            )

            res = []
            check = {}

            for e in parsed_a:
                q_index = e["q_index"]
                check[q_index] = e["rank"]

            for e in parsed_b:
                q_index = e["q_index"]
                if q_index in check and check[q_index] == e["rank"]:
                    res.append(e)
            return res
        except Exception as e:
            raise Exception(f"Error processing responses for consensus: {str(e)}")
