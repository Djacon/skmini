import numpy as np


class MinMaxScaler:
    '''Transform features by scaling each feature to a given range'''
    def __init__(self, feature_range=(0, 1)):
        self.min, self.max = feature_range

    def fit(self, X):
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_min_ = np.nanmin(X, axis=0)

        self.scale_ = (self.max - self.min) / (self.data_max_ - self.data_min_)
        self.min_ = self.min - self.data_min_ * self.scale_

    def transform(self, X):
        return X * self.scale_ + self.min_

    def inverse_transform(self, X):
        return (X - self.min_) / self.scale_
