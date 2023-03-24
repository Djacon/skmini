import numpy as np


'''Classification Metrics'''


class Softmax:
    '''Softmax CrossEntropy with MSE loss'''
    def loss(self, y, y_pred_hot):
        y_hot = self._one_hot_encoder(y, y_pred_hot)
        return ((y_hot - y_pred_hot) ** 2).sum(axis=1)

    def grad(self, y, y_pred_hot):
        y_hot = self._one_hot_encoder(y, y_pred_hot)
        return 2 * (y_pred_hot - y_hot)

    def _one_hot_encoder(self, y, y_pred_hot):
        y_hot = np.zeros_like(y_pred_hot)
        y_hot[np.arange(len(y)), y] = 1
        return y_hot


class Hinge:
    '''Hinge loss for binary classification tasks with y in {-1,1}'''
    def __init__(self, threshold=1.):
        self.threshold = threshold

    def loss(self, y, y_pred):
        M = self.threshold - y * y_pred
        return np.where(M <= 0, 0, M)

    def grad(self, y, y_pred):
        M = self.threshold - y * y_pred
        return np.where(M <= 0, 0, -y)


class Squared_hinge:
    '''Squared Hinge loss for binary classification tasks with y in {-1,1}'''
    def __init__(self, threshold=1.):
        self.threshold = threshold

    def loss(self, y, y_pred):
        M = self.threshold - y * y_pred
        return np.where(M <= 0, 0, M) ** 2

    def grad(self, y, y_pred):
        M = self.threshold - y * y_pred
        return 2 * y * np.where(M <= 0, 0, -M)
