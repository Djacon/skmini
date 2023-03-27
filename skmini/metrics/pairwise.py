import numpy as np

__all__ = [
    'euclidean_distances',
    'manhattan_distances',
]


def euclidean_distances(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum())


def manhattan_distances(x1, x2):
    return np.abs(x1 - x2).sum()
