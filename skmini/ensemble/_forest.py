import numpy as np
from statistics import mode

from ..base import BaseClassifier, BaseRegressor
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseForest():
    '''Base Random Forest model'''
    def __init__(self, estimator=None, n_estimators=100, criterion='gini',
                 max_depth=100, min_samples_split=2, max_features=None,
                 bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)

        self.trees = []
        for _ in range(self.n_estimators):
            tree = self.estimator(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features, random_state=self.random_state)

            if self.bootstrap:
                idxs = rng.choice(n_samples, n_samples, replace=True)
                tree.fit(X[idxs], y[idxs])
            else:
                tree.fit(X, y)

            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self._get_vote(pred) for pred in preds])


class RandomForestClassifier(BaseClassifier, _BaseForest):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=100,
                 min_samples_split=2, max_features=None, bootstrap=True,
                 random_state=None):
        super().__init__(estimator=DecisionTreeClassifier,
                         n_estimators=n_estimators, criterion=criterion,
                         min_samples_split=min_samples_split,
                         random_state=random_state, max_depth=max_depth,
                         max_features=max_features, bootstrap=bootstrap)

    def _get_vote(self, y):
        return mode(y)


class RandomForestRegressor(BaseRegressor, _BaseForest):
    def __init__(self, n_estimators=100, criterion='squared_error',
                 max_depth=100, min_samples_split=2, max_features=None,
                 bootstrap=True, random_state=None):
        super().__init__(estimator=DecisionTreeRegressor,
                         n_estimators=n_estimators, criterion=criterion,
                         min_samples_split=min_samples_split,
                         random_state=random_state, max_depth=max_depth,
                         max_features=max_features, bootstrap=bootstrap)

    def _get_vote(self, y):
        return y.mean()
