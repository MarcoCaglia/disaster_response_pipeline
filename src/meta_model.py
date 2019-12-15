import xgboost as xgb
import numpy as np


class MetaModel():
    def __init__(self, sub_models, train_data):
        X_train, X_test, y_train, y_test = train_data
        self.X_meta = [model.predict(np.vstack([X_train, X_test])) for model in
                       sub_models]
        self.y_meta = np.vstack([y_train, y_test])
        self.xgboost = xgb.XGBoostClassifier()  

    def fit(self):
        self.xgboost.fit(self.X_meta, self.y_meta)

    def predict(self):
        meta_prediction = self.xgboost.predict(self.X_meta)

        return meta_prediction
