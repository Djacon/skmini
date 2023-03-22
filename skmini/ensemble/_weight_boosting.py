import numpy as np

from ..base import ClassifierMixin, RegressorMixin
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor


class AdaBoostClassifier(ClassifierMixin):
    '''Adaptive Boosting Classifier'''
    def __init__(self, estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50, learning_rate=1., random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimator_weights = np.zeros(n_estimators)
        # self.estimator_errors = np.zeros(n_estimators)
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_estimators):
            estimator = self.estimator.clone()
            estimator.random_state = self.random_state
            estimator.fit(X, y, sample_weight=sample_weights)

            y_pred = estimator.predict(X)
            error = np.sum(sample_weights * np.abs(y_pred - y))
            alpha = self.learning_rate * np.log((1 - error) / error)

            sample_weights *= np.exp(alpha * np.abs(y_pred - y))
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights[i] = alpha
            # self.estimator_errors[i] = error

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.estimators):
            preds += self.estimator_weights[i] * estimator.predict(X)


class AdaBoostRegressor(RegressorMixin):
    '''Adaptive Boosting Classifier'''
    def __init__(self, estimator=DecisionTreeRegressor(max_depth=3),
                 n_estimators=50, learning_rate=1., random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimator_weights = np.zeros(n_estimators)
        # self.estimator_errors = np.zeros(n_estimators)
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_estimators):
            estimator = self.estimator.clone()
            estimator.random_state = self.random_state
            estimator.fit(X, y, sample_weight=sample_weights)

            y_pred = estimator.predict(X)
            error = np.sum(sample_weights * np.abs(y_pred - y))
            alpha = self.learning_rate * np.log((1 - error) / error)

            sample_weights *= np.exp(alpha * np.abs(y_pred - y))
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights[i] = alpha
            # self.estimator_errors[i] = error

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.estimators):
            preds += self.estimator_weights[i] * estimator.predict(X)
        return preds
