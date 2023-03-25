import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin
from ..linear_model import LogisticRegression
from ..model_selection import KFold


class BaseStacking:
    '''Base class for voting'''
    def __init__(self, estimators=[], final_estimator=LogisticRegression,
                 cv=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

        folds = list(KFold(self.cv).split(X, y))

        stack_preds = []
        for estimator in self.estimators:
            pred = cross_val_predict(estimator, X, y, self.cv)
            stack_preds.append(pred)
        stack_preds = np.hstack(stack_preds)
        self.final_estimator.fit(stack_preds, y)

    def predict(self, X):
        preds = np.array([estim.predict(X) for estim in self.estimators])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self._get_vote(pred) for pred in preds])


class StackingClassifier(ClassifierMixin, BaseStacking):
    def _get_vote(self, y):
        return mode(y)


class StackingRegressor(RegressorMixin, BaseStacking):
    def _get_vote(self, y):
        return y.mean()
