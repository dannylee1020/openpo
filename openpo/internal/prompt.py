EXTRACT_PROMPT = """
Extract key facts and answers from the response of this query from user.
"""

PREF_PROMPT = """
{}

Provide exactly two different responses.
Answer in your own style.
vary your language and tone, but do not contradict or add to these core facts.
Main answer: {}, Key points: {}.
"""

SINGLE_PROMPT = """
{}

Provide exactly one response.
"""