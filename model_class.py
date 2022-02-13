class ModelClass:
    def __init__(self, **kwargs) -> None:
        self.model = self._generate_model(**kwargs)
        pass

    def _setup(self):
        return

    @staticmethod
    def _generate_model():
        return

    def train(self):
        return

    def get_summary(self):
        return self.model.summary()

