import numpy as np

from ..base import TransformerMixin


class LabelEncoder(TransformerMixin):
    '''Encode target labels with value between 0 and n_classes-1'''
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.classes_ = []

    def fit(self, values):
        self.classes_ = np.unique(values)

        for index, value in enumerate(self.classes_):
            self.label_to_index[str(value)] = index
            self.index_to_label[index] = str(value)

    def transform(self, values):
        return np.array([self.label_to_index[str(value)] for value in values])

    def inverse_transform(self, indices):
        return np.array([self.index_to_label[index] for index in indices])
