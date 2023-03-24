import numpy as np

from ..base import TransformerMixin


class MinMaxScaler(TransformerMixin):
    '''Transform features by scaling each feature to a given range'''
    def __init__(self, feature_range=(0, 1)):
        self.min, self.max = feature_range

    def fit(self, X):
        X = np.array(X)

        self.data_max_ = np.nanmax(X, axis=0)
        self.data_min_ = np.nanmin(X, axis=0)

        self.scale_ = (self.max - self.min) / (self.data_max_ - self.data_min_)
        self.min_ = self.min - self.data_min_ * self.scale_

    def transform(self, X):
        return X * self.scale_ + self.min_

    def inverse_transform(self, X):
        return (X - self.min_) / self.scale_


class StandardScaler(TransformerMixin):
    '''Standardize features by removing mean and scaling to unit variance'''
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_ = 0
        self.std_ = 1

    def fit(self, X):
        X = np.array(X)

        if self.with_mean:
            self.mean_ = X.mean(axis=0)

        if self.with_std:
            self.std_ = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


class Normalizer(TransformerMixin):
    '''Transform features by scaling each feature to a given range'''
    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X):
        pass

    def transform(self, X):
        X = np.array(X)

        if self.norm == 'l2':
            X_norm = (X ** 2).sum(axis=1, keepdims=True) ** .5
        elif self.norm == 'l1':
            X_norm = np.abs(X).sum(axis=1, keepdims=True)
        elif self.norm == 'max':
            X_norm = np.abs(X).max(axis=1, keepdims=True)
        return X / X_norm

    def inverse_transform(self, X):
        return (X - self.min_) / self.scale_
