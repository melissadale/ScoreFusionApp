

class Experiment:
    def __init__(self, **kwargs):
        self.results = kwargs.get('results')
        self.models = kwargs.get('models')



class TrainedModel:
    def __init__(self, **kwargs):
        self.title = kwargs.get('title')
        self.train_time = kwargs.get('train_time')
        self.model = kwargs.get('model')

