import numpy as np

from ._base import _Neighbors

from ..base import BaseRegressor


class KNeighborsClassifier(_Neighbors):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)


class KNeighborsRegressor(BaseRegressor, _Neighbors):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)

    def kfunc(x):
        return x.mean()
