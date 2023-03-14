import numpy as np


'''Regression Metrics'''


class MSE:
    '''Mean Squared Error'''
    def loss(self, y, y_pred):
        return (y - y_pred) ** 2

    def grad(self, y, y_pred):
        return 2 * (y_pred - y)


class MAE:
    '''Mean Absolute Error'''
    def loss(self, y, y_pred):
        return np.abs(y - y_pred)

    def grad(self, y, y_pred):
        return np.sign(y_pred - y)


class MAPE:
    '''Mean Absolute Percentage Error'''
    def loss(self, y, y_pred):
        return np.abs(1 - y_pred / y)  # y != 0

    def grad(self, y, y_pred):
        return np.sign(y_pred - y) / np.abs(y)


class Huber:
    '''Huber Loss'''
    def __init__(self, eps=1.35):
        self.eps = eps
        self.floss = np.vectorize(lambda z: z * z / 2 if z <= self.eps
                                  else self.eps * (z - self.eps / 2))
        self.fgrad = np.vectorize(lambda z: z if abs(z) <= self.eps
                                  else self.eps * np.sign(z))

    def loss(self, y, y_pred):
        z = np.abs(y - y_pred)
        return self.floss(z)

    def grad(self, y, y_pred):
        z = y_pred - y
        return self.fgrad(z)
