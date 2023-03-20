import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin


class DummyClassifier(ClassifierMixin):
    '''DummyClassifier makes predictions that ignore the input features'''
    def __init__(self, strategy='prior', random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def fit(self, _, y):
        self.y_train = np.array(y)

    def predict(self, X):
        n_samples = X.shape[0]

        if self.strategy in ('most_frequent', 'prior'):
            constant = mode(self.y_train)
            return np.array([constant for _ in range(n_samples)])

        elif self.strategy == 'constant':
            constant = self.constant
            return np.array([constant for _ in range(n_samples)])

        elif self.strategy == 'stratified':
            pass

        elif self.strategy == 'uniform':
            values = np.unique(self.y_train)
            rng = np.random.RandomState(self.random_state)
            return rng.choice(values, n_samples)


class DummyRegressor(RegressorMixin):
    '''Regressor that makes predictions using simple rules'''
    def __init__(self, strategy='mean', constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def fit(self, _, y):
        self.y_train = np.array(y)

    def predict(self, X):
        n_samples = X.shape[0]

        if self.strategy == 'mean':
            constant = self.y_train.mean()

        elif self.strategy == 'median':
            constant = self.y_train.median()

        elif self.strategy == 'quantile':
            constant = self.y_train.quantile(self.quantile)

        elif self.strategy == 'constant':
            constant = self.constant

        return np.array([constant for _ in range(n_samples)])
