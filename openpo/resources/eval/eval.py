import json
from typing import Dict, List, Optional

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

    def eval_single(
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
        b     AuthenticationError: If required API keys are missing or invalid.
             ProviderError: For provider-specific errors during evaluation.
             ValueError: If the model format is invalid or provider is not supported.
        """
        try:
            provider = self.client._get_model_provider(model)
            model_id = self.client._get_model_id(model)

            if provider not in ["openai", "anthropic"]:
                raise ProviderError(provider, "Provider not supported for evaluation")

            llm = self.client._get_provider_instance(provider=provider)
            res = llm.generate(
                model=model_id,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )

            if provider == "anthropic":
                result = res.content[0].input['"evaluation']
            result = json.loads(res.choices[0].message.content)["evaluation"]

            return {"evaluation": result}
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider=provider, message=f"Error during evaluation: {str(e)}"
            )

    def eval_multi(
        self,
        models: List[str],
        questions: List[str],
        responses: List[List],
        prompt: Optional[str] = None,
    ):
        """Use multiple LLMs as a judge for model consensus to evaluate responses for building preference data.

        Args:
            models (List): List of models to use as a judge. Follows provider/model-identifier format.
            questions (List(str)): Questions for each response pair.
            responses (List[List[str]]): Pairwise responses to evaluate.
            prompt (str): Optional custom prompt for judge model to follow.

        Returns (Dict): The evaluation data for responses that all models agree on.

            - preference: Evaluation data on the input responses.
            - q_index: Index of questions that reached consensus by the models.

        Raises:
            AuthenticationError: If required API keys are missing or invalid.
            ProviderError: For provider-specific errors during evaluation.
            ValueError: If the model format is invalid or required models are missing.
        """
        try:
            judge_a = self.client._get_provider_instance("anthropic")
            judge_o = self.client_get_provider_instance("openai")

            a_model = ""
            o_model = ""

            for m in models:
                provider = self.client._get_model_provider(m)
                if provider == "anthropic":
                    a_model = self.client._get_model_id(m)
                elif provider == "openai":
                    o_model = self.client._get_model_id(m)
                else:
                    raise ProviderError(
                        provider, "Provider not supported for evaluation"
                    )

            if not a_model or not o_model:
                raise ValueError("Both Anthropic and OpenAI models must be provided")

            res_a = judge_a.generate(
                model=a_model,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )
            parsed_res_a = res_a.content[0].input["evaluation"]

            res_o = judge_o.generate(
                model=o_model,
                questions=questions,
                responses=responses,
                prompt=prompt if prompt else None,
            )
            parsed_res_o = json.loads(res_o.choices[0].message.content)["evaluation"]

            idx = self._get_model_consensus(
                parsed_res_a,
                parsed_res_o,
            )

            return {
                "evaluation": [parsed_res_o[i] for i in idx],
                "q_index": idx,
            }
        except (AuthenticationError, ValueError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                provider="eval-multi",
                message=f"Error during multi-model evaluation: {str(e)}",
            )
