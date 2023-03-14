import numpy as np


'''Classification Metrics'''


class BCE:
    '''Binary CrossEntropy (LogLoss)'''
    def loss(self, y, y_pred):
        return -np.where(y == 1, np.log(y_pred), np.log(1-y_pred))

    def grad(self, y, y_pred):
        return y_pred - y
#         return np.where(y == 1, -1 / y_pred, 1 / (1 - y_pred))


class Hinge:
    '''Hinge loss'''
    def loss(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, M)

    def grad(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, -y)


class Squared_hinge:
    '''Squared Hinge loss'''
    def loss(self, y, y_pred):
        M = 1 - y * y_pred
        return np.where(M <= 0, 0, M) ** 2

    def grad(self, y, y_pred):
        M = 1 - y * y_pred
        return 2 * y * np.where(M <= 0, 0, -M)
