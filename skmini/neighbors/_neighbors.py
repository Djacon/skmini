from statistics import mode

from ._base import NeighborsBase

from ..base import RegressorMixin, ClassifierMixin


class KNeighborsClassifier(ClassifierMixin, NeighborsBase):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)

    def kfunc(self, x):
        return mode(x)


class KNeighborsRegressor(RegressorMixin, NeighborsBase):
    def __init__(self, n_neighbors=5, distance='euclidean'):
        super().__init__(n_neighbors, distance)

    def kfunc(self, x):
        return x.mean()
