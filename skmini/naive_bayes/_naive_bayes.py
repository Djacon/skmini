import numpy as np

from ..base import ClassifierMixin


class _BaseNB(ClassifierMixin):
    '''Abstract base class for naive Bayes estimators'''

    def predict(self, X):
        return np.array([self._predict(x) for x in X])


class GaussianNB(_BaseNB):
    '''Gaussian Naive Bayes model'''
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._priors = np.zeros(n_classes)

        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))

        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._priors[i] = X_c.shape[0] / n_samples

            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0) + self.var_smoothing

    def _predict(self, x):
        posteriors = []

        priors = np.log(self._priors)
        for i in range(len(self._classes)):
            likelihood = np.sum(np.log(self._pdf(i, x)))
            posteriors.append(likelihood + priors[i])
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        return np.exp(-(x-mean)**2/(2*var)) / np.sqrt(2 * np.pi * var)
