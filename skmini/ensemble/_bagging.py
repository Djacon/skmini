import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaseBagging:
    '''Base Bootstrap aggregating model'''
    def __init__(self, estimator=None, n_estimators=10, max_features=None,
                 bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # # Check max_features variable
        # if self.max_features:
        #     self.max_features = min(self.max_features, X.shape[1])
        # else:
        #     self.max_features = X.shape[1]

        rng = np.random.RandomState(self.random_state)

        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = self.estimator.clone()

            if self.bootstrap:
                idxs = rng.choice(n_samples, n_samples)
                estimator.fit(X[idxs], y[idxs])
            else:
                estimator.fit(X, y)

            self.estimators.append(estimator)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.estimators])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self._get_vote(pred) for pred in preds])


class BaggingClassifier(ClassifierMixin, BaseBagging):
    def __init__(self, estimator=DecisionTreeClassifier(), n_estimators=10,
                 max_features=None, bootstrap=True, random_state=None):
        super().__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         max_features=max_features,
                         bootstrap=bootstrap, random_state=random_state)

    def _get_vote(self, y):
        return mode(y)


class BaggingRegressor(RegressorMixin, BaseBagging):
    def __init__(self, estimator=DecisionTreeRegressor(), n_estimators=10,
                 max_features=None, bootstrap=True, random_state=None):
        super().__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         max_features=max_features,
                         bootstrap=bootstrap, random_state=random_state)

    def _get_vote(self, y):
        return y.mean()
