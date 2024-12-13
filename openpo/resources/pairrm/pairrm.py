from typing import List

import llm_blender
import numpy as np


class PairRM:
    """
    A class that implements the Pairwise Rewards Model (PairRM) for evaluating and ranking LLM responses.

    This class uses the llm-blender package to load and utilize the PairRM model, which can rank
    multiple responses for a given prompt based on their quality.

    """

    def __init__(self):
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

    def _format_preference(
        self,
        ranks: List,
        prompts: List,
        responses: List,
    ) -> List[dict]:
        dataset = []

        for i in range(len(prompts)):
            try:
                dataset.append(
                    {
                        "prompt": prompts[i],
                        "preferred": responses[i][np.where(ranks[i] == 1)[0][0]],
                        "rejected": responses[i][
                            np.where(ranks[i] == max(ranks[i]))[0][
                                0
                            ]  # Fixed: ranks[0] to ranks[i]
                        ],
                    }
                )
            except (ValueError, IndexError):
                print(f"Skipping index {i} due to ranking issues.")
                continue

        return dataset

    def eval(
        self,
        prompts: List,
        responses: List,
    ) -> List[dict]:
        """
        Evaluates and ranks multiple responses for given prompts.

        Args:
            prompts (List): List of input prompts to evaluate.
            responses (List): List of response sets to be ranked.
                Each response set should contain multiple responses for the corresponding prompt.

        Returns:
            List[dict]: A formatted list of preference data containing the ranking results.
                See _format_preference method for the structure of the returned data.
        """
        ranks = self.blender.rank(prompts, responses)
        return self._format_preference(ranks, prompts, responses)
