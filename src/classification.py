import vectorization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import pickle

class MessageCategorizer():
    def __init__(self, train_data=None, new_data=None, model=None):
        self._validate_parameters(train_data, new_data, model)
        if isinstance(model, str):
            with open(model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = self._define_model(train_data)

    def _validate_parameters(self, train_data, new_data, model):
        pass

    def _define_model(self, train_data):
        X_train, X_test, y_train, y_test = train_data
        logreg = LogisticRegression()
        svc = SVC()
        adaboost = AdaBoostClassifier()
        gradboost = GradientBoostingClassifier()

        models = {
            'LogisticRegression': LogisticRegression(),
            'SVC': SVC(),
            'AdaBoost': AdaBoostClassifier(),
            'GradBoost': GradientBoostingClassifier()
        }

        for model in models:
            model_cv = GridSearchCV(
                estimator=models[model],
                param_grid = self.get ##### HERE

            )
            model.fit(X_train, y_train)