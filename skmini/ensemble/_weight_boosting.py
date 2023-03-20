import numpy as np

from ..base import ClassifierMixin, RegressorMixin


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoostClassifier(ClassifierMixin):
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Init weights
        w = np.full(n_samples, 1 / n_samples)

        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = DecisionStump()

            min_err = float('inf')
            for feature_idx in range(n_features):
                X_column = X[:, feature_idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    y_preds = np.ones(n_samples)
                    y_preds[X_column < threshold] = -1

                    misclassified = w[y != y_preds]
                    error = sum(misclassified)

                    if error >= 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_err:
                        min_err = error
                        estimator.polarity = p
                        estimator.threshold = threshold
                        estimator.feature_idx = feature_idx

            estimator.alpha = .5 * np.log((1 - min_err) / (min_err + 1e-10))

            preds = estimator.predict(X)

            w *= np.exp(-estimator.alpha * y * preds)
            w /= np.sum(w)

            self.estimators.append(estimator)

    def predict(self, X):
        pred = [estim.alpha * estim.predict(X) for estim in self.estimators]
        return np.sign(np.sum(pred, axis=0))
