import numpy as np


def euclidean(x1, x2):
    return ((x1 - x2)**2).sum() ** .5


def manhattan(x1, x2):
    return np.abs(x1 - x2).sum()


METRICS = {
    'euclidean': euclidean,
    'manhattan': manhattan
}


class NeighborsBase:
    '''Base class for nearest neighbors estimators'''
    def __init__(self, n_neighbors=5, distance='euclidean'):
        self.k = n_neighbors
        self.distance = METRICS[distance]

    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)

    def predict(self, X):
        y_pred = []
        for x in X:
            dist = [self.distance(x, x_train) for x_train in self.X_train]
            labels = self.y_train[np.argsort(dist)[:self.k]]
            y_pred.append(self.kfunc(labels))
        return np.array(y_pred)
