from resources.beta.chat import completions


class Chat:
    def __init__(self):
        self.completions = completions.Completions()
