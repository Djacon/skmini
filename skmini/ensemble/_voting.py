import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin


class BaseVoting:
    '''Base class for voting'''
    def __init__(self, estimators=[], voting='hard'):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        if self.voting == 'hard':
            preds = np.array([estim.predict(X) for estim in self.estimators])
            preds = np.swapaxes(preds, 0, 1)
            return np.array([self._get_vote(pred) for pred in preds])
        else:
            print('Not implemented, yet')


class VotingClassifier(ClassifierMixin, BaseVoting):
    def _get_vote(self, y):
        return mode(y)


class VotingRegressor(RegressorMixin, BaseVoting):
    def _get_vote(self, y):
        return y.mean()
