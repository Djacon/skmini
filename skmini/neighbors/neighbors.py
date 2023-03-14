import numpy as np

from ._base import Neighbors

from ..base import BaseRegressor


class KNeighborsClassifier(Neighbors):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)


class KNeighborsRegressor(BaseRegressor, Neighbors):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)
        self.kfunc = lambda x: x.mean()
