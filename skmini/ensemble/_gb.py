import numpy as np

from ..metrics import MSE
from ..tree import DecisionTreeRegressor
from ..base import ClassifierMixin, RegressorMixin


class BaseGradientBoosting:
    '''Abstract base class for Gradient Boosting'''
    def __init__(self, loss=MSE(), learning_rate=.1, n_estimators=100,
                 criterion='', min_samples_split=2, max_depth=3,
                 max_features=None, random_state=None):
        self.grad = loss.grad
        self.lr = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.F0 = y.mean()
        y_pred = np.full_like(y, self.F0, dtype=np.float64)

        self.trees = []
        for _ in range(self.n_estimators):
            grad = -self.grad(y, y_pred)

            tree = DecisionTreeRegressor(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state)

            tree.fit(X, grad)
            self.trees.append(tree)

            y_pred += self.lr * tree.predict(X)


class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    '''Gradient Boosting Classifier model'''
    def __init__(self, loss=MSE(), learning_rate=.1, n_estimators=100,
                 criterion='squared_error', min_samples_split=2, max_depth=3,
                 max_features=None, random_state=None):
        super().__init__(loss=loss, learning_rate=learning_rate,
                         n_estimators=n_estimators, criterion=criterion,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth, max_features=max_features,
                         random_state=random_state)

    def predict(self, X):
        preds = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return np.round(self.F0 + self.lr * preds)


class GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting):
    '''Gradient Boosting Regressor model'''
    def __init__(self, loss=MSE(), learning_rate=.1, n_estimators=100,
                 criterion='squared_error', min_samples_split=2, max_depth=3,
                 max_features=None, random_state=None):
        super().__init__(loss=loss, learning_rate=learning_rate,
                         n_estimators=n_estimators, criterion=criterion,
                         min_samples_split=min_samples_split,
                         max_depth=max_depth, max_features=max_features,
                         random_state=random_state)

    def predict(self, X):
        preds = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return self.F0 + self.lr * preds
