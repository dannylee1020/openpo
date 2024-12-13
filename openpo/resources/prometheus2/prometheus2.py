from typing import List, Tuple

from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import (
    ABSOLUTE_PROMPT_WO_REF,
    FACTUAL_VALIDITY_RUBRIC,
    HARMLESSNESS_RUBRIC,
    HELPFULNESS_RUBRIC,
    HONESTY_RUBRIC,
    REASONING_RUBRIC,
    RELATIVE_PROMPT_WO_REF,
)


class Prometheus2:
    """
    A class that implements the Prometheus2 evaluation model for assessing LLM responses.

    This class provides methods for both relative and absolute evaluation of LLM responses
    using different rubrics such as factual validity, helpfulness, honesty, and reasoning.

    """

    def __init__(self, model):
        self.model = model
        self.rubric_mapping = {
            "factual-validity": FACTUAL_VALIDITY_RUBRIC,
            "helpfulness": HELPFULNESS_RUBRIC,
            "honesty": HONESTY_RUBRIC,
            "harmlessness": HARMLESSNESS_RUBRIC,
            "reasoning": REASONING_RUBRIC,
        }

    def _format_absolute(
        self,
        scores: List[int],
        instructions: List[str],
        responses: List[str],
        feedbacks: List[str],
    ) -> List[dict]:

        dataset = []
        for i in range(len(instructions)):
            dataset.append(
                {
                    "prompt": instructions[i],
                    "response": responses[i],
                    "score": scores[i],
                    "feedback": feedbacks[i],
                }
            )

        return dataset

    def _format_relative(
        self,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        feedbacks: List[str],
        scores: List[str],
    ) -> List[dict]:

        dataset = []

        for i in range(len(instructions)):
            dataset.append(
                {
                    "prompt": instructions[i],
                    "preferred": responses_A[i] if scores[i] == "A" else responses_B[i],
                    "rejected": responses_A[i] if scores[i] == "B" else responses_B[i],
                    "feedback": feedbacks[i],
                }
            )

        return dataset

    def eval_relative(
        self,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        rubric: str,
    ) -> List[dict]:
        """
        Performs relative evaluation comparing pairs of responses.

        Args:
            instructions (List[str]): List of instruction prompts.
            responses_A (List[str]): First set of responses to compare.
            responses_B (List[str]): Second set of responses to compare.
            rubric (str): The evaluation rubric to use. Supported rubrics:

                - factual-validity
                - helpfulness
                - honesty
                - harmlessness
                - reasoning

        Returns:
            List[dict]: A formatted list of evaluation results containing preferences and feedback.

        Raises:
            Exception: If there's an error during the evaluation process.
        """
        try:
            judge = PrometheusEval(
                model=self.model,
                relative_grade_template=RELATIVE_PROMPT_WO_REF,
            )
            feedbacks, scores = judge.relative_grade(
                instructions=instructions,
                responses_A=responses_A,
                responses_B=responses_B,
                rubric=self.rubric_mapping[rubric],
            )

            return self._format_relative(
                instructions=instructions,
                responses_A=responses_A,
                responses_B=responses_B,
                feedbacks=feedbacks,
                scores=scores,
            )

        except Exception as e:
            raise Exception(f"Error while evaluating with Prometheus2: {e}")

    def eval_absolute(
        self,
        instructions: List[str],
        responses: List[str],
        rubric: str,
    ) -> List[dict]:
        """
        Performs absolute evaluation of individual responses.

        Args:
            instructions (List[str]): List of instruction prompts.
            responses (List[str]): List of responses to evaluate.
            rubric (str): The evaluation rubric to use. Supported rubrics:

                - factual-validity
                - helpfulness
                - honesty
                - harmlessness
                - reasoning

        Returns:
            List[dict]: A formatted list of evaluation results containing scores and feedback.

        Raises:
            Exception: If there's an error during the evaluation process.
        """
        try:
            judge = PrometheusEval(
                model=self.model,
                absolute_grade_template=ABSOLUTE_PROMPT_WO_REF,
            )
            feedbacks, scores = judge.absolute_grade(
                instructions=instructions,
                responses=responses,
                rubric=self.rubric_mapping[rubric],
            )

            return self._format_absolute(
                scores=scores,
                instructions=instructions,
                responses=responses,
                feedbacks=feedbacks,
            )
        except Exception as e:
            raise Exception(f"Error while evaluating with Prometheus2: {e}")
