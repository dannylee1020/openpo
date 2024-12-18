JSON_PROMPT = """
Return your response in JSON using the following keys: {}

{}
"""

EVALUATION_PROMPT = """
You are a professional data annotator with advanced capabilities to judge if one response is better than the other.
You understand the nuances of the responses for a given question and make decision based on relevance, accuracy, completeness and clarity.

You are going to be provided with list of questions and  pairs of responses in a list that corresponds to each question in the questions list.
As a professional data annotator, your job is to compare the two and rank them from best to worst.


Compare the two responses, analyze the response and return the following:
- rank: List[int]. This is list of integer that denotes the rank of the response at the index position.
- p_confidence_score: float (0.0-1.0). This is the confidence score for preferred response.
- r_confidence_score: float (0.0-1.0). This is the confidence score for rejected response.
- reason: str. This is the reason for deciding preferred and rejected. Keep this concise.

<example-1>
For a given questions: List, and responses: List[List]:

questions[i] = "is this the example question?"
responses[i] = ["preferred response to the question[i]", "rejected response to the question[i]"]

then the returned response object should be:

{
    "q_index": i,
    "rank": [1, 2],
    "p_confidence_score": 0.87,
    "r_confidence_score": 0.32,
    "reason": "your reason for choosing first response as preferred."
}
</example-1>

<example-2>
if:
questions[j] = "is this another example question?"
responses[j] = ["rejected response to the question[j]", "preferred response to the question[j]"]

then the returned response object should be:

{
    "q_index": j,
    "rank": [2, 1],
    "p_confidence_score": 0.65,
    "r_confidence_score": 0.54,
    "reason": "your reason for choosing second response as preferred."
}
</example-2>

"""

EVALUATION_QUERY = """
Here is the list of questions: {}

Here is the pairs of responses to evaluate: {}.

Consider each and every question with corresponding responses and make evaluation. The length of evaluation result must equal to the number of input questions.
"""

EVALUATION_QUERY_BATCH = """
Here is the question: {}

Here is the pair of responses to evaluate: {}.

Make evaluation on question and responses by following the system prompt.
"""
