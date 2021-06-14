

class TrainedModel:
    def __init__(self, **kwargs):
        self.title = kwargs.get('title')
        self.train_time = 0.0
        self.model = None

    def set_train_time(self, how_long):
        self.train_time = how_long

    def set_model(self, mdl):
        self.model = mdl