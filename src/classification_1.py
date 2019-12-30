from .etl_pipeline import load_messages_to_db
from .vectorization_union import Vectorizer
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from multiprocessing import cpu_count
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
            'LogRegModel_score': None,
            'SVCModel_score': None,
            'RandForest_score': None,
            'AdaBoost_score': None,
            'GradBoost_score': None,
            'XGBoost_score': None,
            'meta_model': None
            'best_avg_f1_score': 0,
            'best_model': 'None'
        }
        scorer = lambda x, y: np.mean([f1_score(x[:, i], y[:, i],
                                       average='weighted')
                                       for i in range(x.shape[1])])
        self.scorer = make_scorer(scorer)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = None
        self.logreg = None
        self.svc = None
        self.adaboost = None
        self.gradboost = None
        self.xgboost = None

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

        self.X_train,self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, stratify=y)

        self.report['vectorized'] = method

    def fit_logreg(self, optimize=False, max_jobs=1, **kwargs):
        logreg = MultiOutputClassifier(
                LogisticRegression(**kwargs),
                n_jobs=max_jobs
                )
        if not optimize:
            self.logreg, score = self._fit_model(logreg)
            self.report['LogRegModel_score'] == score
        elif optimize:
            self.logreg, score = self._fit_grid_search(logreg, 'logreg')

    def _fit_model(self, model):
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        score = self.scorer(self.y_test, y_hat)

        return model, score

    def _fit_grid_search(self, model):
        param_grids = {
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
        model_cv = GridSearchCV(model, )

            