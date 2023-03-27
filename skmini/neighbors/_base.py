import numpy as np

from ..metrics.pairwise import euclidean_distances, manhattan_distances


METRICS = {
    'euclidean': euclidean_distances,
    'manhattan': manhattan_distances
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
