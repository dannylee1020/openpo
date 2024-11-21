EXTRACT_PROMPT = """
Extract key facts and main point from the answer to the question Make both concise and to the point.
"""

EXTRACT_PROMPT_JSON = (
    EXTRACT_PROMPT
    + """ Return your response in JSON format using the following keys: {}"""
)


PREF_PROMPT = """
{}

Provide exactly two different responses.
Answer in your own style.
vary your language and tone, but do not contradict or add to these core facts.
Main answer: {}, Key points: {}.
"""

PREF_PROMPT_JSON = """{}
Return your response in JSON format using the following keys: {}"""


SINGLE_PROMPT = """
Provide exactly one response.
{}
"""

SINGLE_PROMPT_JSON = """
Provide exactly one response.
{}
Return your response in JSON format using the following keys: {}
"""
