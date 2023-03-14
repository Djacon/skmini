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


class HUBER:
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


'''Classification Metrics'''


class BCE:
    '''Binary CrossEntropy (LogLoss)'''
    def loss(self, y, y_pred):
        return -np.where(y == 1, np.log(y_pred), np.log(1-y_pred))

    def grad(self, y, y_pred):
        return y_pred - y
#         return np.where(y == 1, -1 / y_pred, 1 / (1 - y_pred))


class HINGE:
    '''Hinge loss'''
    def loss(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, M)

    def grad(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, -y)


class Squared_Hinge:
    '''Squared Hinge loss'''
    def loss(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, M) ** 2

    def grad(self, y, y_pred):
        M = 1 - y * y_pred
        return 2 * y * np.where(M <= 0, 0, -M)
