JSON_PROMPT = """
Return your response in JSON using the following keys: {}

{}
"""

EVALUATION_PROMPT = """
You are a professional data annotator with advanced capabilities to judge if one response is better than the other.
You understand the nuances of the responses for a given question and make decision based on relevance, accuracy, completness and clarity.

You are going to be provided with pairs of responses in a list. As a professional data annotator, your job is to compare the two,
pick one as preferred and other as rejected.


Compare the two responses, analyze the response and return the following:
- rank: List[int]. This is list of integer that denotes the rank of the response at the index position.
- p_confidence_score: float (0.0-1.0). This is the confidence score for preferred response.
- r_confidence_score: float (0.0-1.0). This is the confidence score for rejected response.
- reason: str. This is the reason for deciding preferred and rejected.

<example-1>
if:
response_pair = ["preferred-response", "rejected-response"]

then the returned response object should be:

{
    "rank": [1, 2],
    "p_confidence_score": 0.87,
    "r_confidence_score": 0.32,
    "reason": "your reason for choosing first response as preferred."
}
</example-1>

<example-2>
if:
response_pair = ["rejected-response", "preferred-response"]

then the returned response object should be:

{
    "rank": [2, 1],
    "p_confidence_score": 0.65,
    "r_confidence_score": 0.54,
    "reason": "your reason for choosing second response as preferred."
}
</example-2>
"""

EVALUATION_QUERY = """
Here is the pairs of responses to annotate: {}.
Please go through the list one by one and choose the response you prefer.
"""
