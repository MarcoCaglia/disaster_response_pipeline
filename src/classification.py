from .meta_model import MetaModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import pickle
import numpy as np
import os
from tqdm import tqdm

class MessageCategorizer():
    def __init__(self, train_data=None, model=None):
        self._validate_parameters(train_data, model)
        if isinstance(model, str):
            with open(model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.models, self.train_f1 = self._define_model(train_data)
            self.meta_prediction = self._get_meta_predictions(train_data)

        return self.meta_prediction, self.train_f1_score

    def _validate_parameters(self, train_data, model):
        pass

    def _define_model(self, train_data):
        X_train, X_test, y_train, y_test = train_data
        logreg = MultiOutputClassifier(LogisticRegression(), n_jobs=-1)
        # svc = MultiOutputClassifier(SVC())
        adaboost = MultiOutputClassifier(AdaBoostClassifier(), n_jobs=-1)
        gradboost = MultiOutputClassifier(GradientBoostingClassifier(),
                                          n_jobs=-1)

        models = {
            'LogisticRegression': logreg,
            # 'SVC': svc,
            'AdaBoost': adaboost,
            'GradBoost': gradboost
        }
        models_fit = []
        predictions = []
        for model in tqdm(models):
            model_cv = GridSearchCV(
                estimator=models[model],
                param_grid=self._get_param_grid(model),
                error_score=0.0,
                scoring=self._scoring,
                n_jobs=-1,
                cv=5,
                refit=False
            )
            model_cv.fit(X_train, y_train)
            predictions.append(model_cv.predict(X_test))
            models_fit.append(model[model].set_params(**model_cv.best_params_))
        
        return models_fit, np.mean([self._scoring(y_test, prediction) for
                                    prediction in predictions])

    def _get_meta_predictions(self, train_data):
        meta_model = MetaModel(sub_models=self.models, train_data=train_data)
        meta_model.fit()
        meta_prediction = meta_model.predict()

        return meta_prediction

    def _get_param_grid(self, label):
        param_grid = {
            'LogisticRegression': {
                'penalty': ['l2', 'l1', 'elasticnet'],
                'C': [0.1, 0.25, 0.5, 0.75, 0.9],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            },
            # 'SVC': {},
            'AdaBoost': {
                'base_estimator': [DecisionTreeClassifier(max_depth=50),
                                   RandomForestClassifier(n_estimators=200),
                ],
                'n_estimators': [25, 50, 75, 100],
            },
            'GradBoost': {
                'n_estimators': [50, 75, 100, 125, 150]
            }
        }
        
        return param_grid[label]

    def _scoring(self, y_true, y_pred):
        avg_f1 = np.mean([f1_score(y_true.values[:, i], y_pred[:, i],
                          average='weighted') for i in range(y_true.shape[1])])
        return avg_f1

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)