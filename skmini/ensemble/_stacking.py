import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin
from ..linear_model import LogisticRegression, RidgeCV
# from ..model_selection import KFold


class BaseStacking:
    '''Base class for voting'''
    def __init__(self, estimators=[], final_estimator=None, cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

        # folds = list(KFold(self.cv).split(X, y))

        stack_preds = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            stack_preds.append(pred)
        stack_preds = np.hstack(stack_preds)
        self.final_estimator.fit(stack_preds, y)

    def predict(self, X):
        preds = np.array([estim.predict(X) for estim in self.estimators])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self.final_estimator.predict(pred) for pred in preds])


class StackingClassifier(ClassifierMixin, BaseStacking):
    def __init__(self, estimators=[], final_estimator=LogisticRegression(),
                 cv=5):
        super().__init__(estimators=estimators,
                         final_estimator=final_estimator, cv=cv)


class StackingRegressor(RegressorMixin, BaseStacking):
    def __init__(self, estimators=[], final_estimator=RidgeCV(),
                 cv=5):
        super().__init__(estimators=estimators,
                         final_estimator=final_estimator, cv=cv)
