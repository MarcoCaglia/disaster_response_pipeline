from .meta_model import MetaModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, make_scorer
import pickle
import numpy as np
import os
from tqdm import tqdm

class MessageCategorizer():
    def __init__(self, train_data=None, model=None):
        self._validate_parameters(train_data, model)
        self.scorer = self._make_scorer()
        if isinstance(model, str):
            with open(model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.models, self.train_f1 = self._define_model(train_data)
            self.meta_model = self._get_meta_predictions(train_data)

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
                scoring=self.scorer,
                n_jobs=-1,
                cv=5,
                refit=True
            )
            model_cv.fit(X_train, y_train)
            predictions.append(model_cv.predict(X_test))
            models_fit.append(models[model].set_params(**model_cv.best_params_))
        
        return models_fit, np.mean([self.scorer(y_test, prediction) for
                                    prediction in predictions])

    def _get_meta_predictions(self, train_data):
        meta_model = MetaModel(sub_models=self.models, train_data=train_data)
        meta_model.fit()
        
        return meta_model

    def _get_param_grid(self, label):
        param_grid = {
            'LogisticRegression': {
                'estimator__penalty': ['l2', 'l1', 'elasticnet'],
                'estimator__C': [0.1, 0.25, 0.5, 0.75, 0.9],
                'estimator__solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            },
            # 'SVC': {},
            'AdaBoost': {
                'estimator__base_estimator': [DecisionTreeClassifier(max_depth=50),
                                   RandomForestClassifier(n_estimators=200),
                ],
                'estimator__n_estimators': [25, 50, 75, 100],
            },
            'GradBoost': {
                'estimator__n_estimators': [50, 75, 100, 125, 150]
            }
        }
        
        return param_grid[label]

    # def _scoring(self, y_true, y_pred):
    #     avg_f1 = np.mean([f1_score(y_true.values[:, i], y_pred[:, i],
    #                       average='weighted') for i in range(y_true.shape[1])])
    #     return avg_f1

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.meta_model, f)

    def _make_scorer(self):
        scorer = lambda y_true, y_pred: np.mean([f1_score(y_true.values[:, i], y_pred[:, i],
                                        average='weighted') for i in range(y_true.shape[1])])

        return make_scorer(scorer)
