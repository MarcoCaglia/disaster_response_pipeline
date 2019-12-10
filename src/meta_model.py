import xgboost

class MetaModel():
    def __init__(self, sub_models=None, load_model=None):
        self.sub_models = sub_models

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
