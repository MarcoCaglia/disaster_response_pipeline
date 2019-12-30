from .etl_pipeline import load_messages_to_db
from .vectorization_union import Vectorizer
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
import xgboost as xgb


class ParameterNotStringError(Exception):
    pass


class UnknownMethodError(Exception):
    pass


class ModelNotImplementedError(Exception):
    pass


class DisasterResponsePipeline:
    def __init__(self, message_path, categories_path, folder):
        for parameter in [message_path, categories_path, folder]:
            try:
                assert isinstance(parameter, str)
            except AssertionError:
                raise ParameterNotStringError
        self.folder = folder
        self._load_messages(message_path, categories_path, folder)

        self.report = {
            'vectorized': False,
            'LogReg_score': None,
            'SVC_score': None,
            'RandForest_score': None,
            'AdaBoost_score': None,
            'GradBoost_score': None,
            'XGBoost_score': None,
            'MetaModel_score': None,
        }

        self.scorer = make_scorer(scorer)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = None
        self.models = {
            'LogReg': MultiOutputClassifier(LogisticRegression()),
            'SVC': MultiOutputClassifier(SVC()),
            'AdaBoost': MultiOutputClassifier(AdaBoostClassifier()),
            'GradBoost': MultiOutputClassifier(GradientBoostingClassifier()),
            'RandForest': MultiOutputClassifier(RandomForestClassifier()),
            'XGBoost': MultiOutputClassifier(xgb.XGBClassifier()),
            'MetaModel': MultiOutputClassifier(xgb.XGBClassifier())
        }

    def _load_messages(self, message_path, categories_path, folder):
        load_messages_to_db(message_path, categories_path, folder)

    def vectorize(self, method, **kwargs):
        conn = sqlite3.connect(self.folder)
        query_y = 'SELECT * FROM data'
        y = pd.read_sql(query_y, conn).drop('message', axis=1)
        self.labels = y.columns
        y = y.values
        if method is 'doc2vec':
            vectorizer = Vectorizer(self.folder)
            X = vectorizer.transform(**kwargs)
        elif method is 'tfidf':
            query = 'SELECT message FROM data'
            X = pd.read_sql(query, conn).values
            tfidf = TfidfVectorizer(min_df=1,
                                    stop_words='english',
                                    **kwargs)
            X = tfidf.fit_transform(X)
        else:
            raise UnknownMethodError

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, stratify=y)

        self.report['vectorized'] = method

    def fit_model(self, model, optimize=False, n_jobs=1, **kwargs):
        if model not in self.models:
            raise ModelNotImplementedError
        if optimize:
            model_cv = GridSearchCV(self.models[model],
                                    param_grid=self._get_param_grid,
                                    scoring=self.scorer,
                                    n_jobs=n_jobs,
                                    **kwargs
                                    )
            self.report[model + '_score'] = self.scorer(model_cv,
                                                        self.X_test,
                                                        self.y_test)
            self.models[model] = model_cv
        else:
            self.models[model].n_jobs = n_jobs
            self.models[model].fit(self.X_train, self.y_train)
            self.report[model + '_score'] = self.scorer(self.models[model],
                                                        self.X_test,
                                                        self.y_test)

    def _get_param_grid(self, label):
        param_grid = {
            'LogisticRegression': {
                'estimator__penalty': ['l2', 'l1', 'elasticnet'],
                'estimator__C': [0.1, 0.25, 0.5, 0.75, 0.9],
                'estimator__solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            },
            # 'SVC': {},
            'AdaBoost': {
                'estimator__base_estimator': [
                    DecisionTreeClassifier(max_depth=50),
                    RandomForestClassifier(n_estimators=200),
                ],
                'estimator__n_estimators': [25, 50, 75, 100],
            },
            'GradBoost': {
                'estimator__n_estimators': [50, 75, 100, 125, 150]
            }
        }

        return param_grid[label]


def scorer(y, y_pred):
    score = np.mean(
        [f1_score(y[:, i], y_pred[:, i], average='weighted') for i in
         range(y.shape[0])]
    )
    return score
