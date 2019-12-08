import vectorization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn

class MessageCategorizer():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.logreg = 