from typing import Union, Dict
from huggingface_hub import InferenceClient
from openpo.resources.chat import completions


class Chat:
    def __init__(self, client: Union[InferenceClient, Dict]):
        self.completions = completions.Completions(client)
