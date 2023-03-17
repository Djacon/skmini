import numpy as np
from statistics import mode

from ..base import BaseClassifier, BaseRegressor
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseBagging():
    '''Base Bootstrap aggregating model'''
    def __init__(self, estimator=None, n_estimators=10, max_features=None,
                 bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)

        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = clone(self.estimator)Bagging

            if self.bootstrap:
                idxs = rng.choice(n_samples, n_samples)
                estimator.fit(X[idxs], y[idxs])
            else:
                estimator.fit(X, y)

            self.estimators.append(estimator)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self._get_vote(pred) for pred in preds])


class BaggingClassifier(BaseClassifier, _BaseBagging):
    def __init__(self, estimator=DecisionTreeClassifier(), n_estimators=10,
                 max_features=None, bootstrap=True, random_state=None):
        super().__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         random_state=random_state,
                         max_features=max_features, bootstrap=bootstrap)

    def _get_vote(self, y):
        return mode(y)


class BaggingRegressor(BaseRegressor, _BaseBagging):
    def __init__(self, estimator=DecisionTreeRegressor(), n_estimators=10,
                 max_features=None, bootstrap=True, random_state=None):
        super().__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         random_state=random_state,
                         max_features=max_features, bootstrap=bootstrap)

    def _get_vote(self, y):
        return y.mean()
