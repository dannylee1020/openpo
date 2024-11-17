from openpo.resources.chat import completions


class Chat:
    def __init__(self):
        self.completions = completions.Completions()
