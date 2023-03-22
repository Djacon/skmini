import numpy as np


class BaseKFold:
    '''Base class for KFold, GroupKFold, and StratifiedKFold'''
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        '''Split the data into folders'''

        # Get number of samples
        n_samples = len(X)

        if n_samples < self.n_splits:
            raise ValueError('n_splits must be less than number of samples!')

        # Init array of indices
        indices = np.arange(n_samples)

        # If it's True, shuffle the indices
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(indices)

        # Calculate size of every fold and number of residuals
        fold_size = n_samples // self.n_splits
        res = n_samples % self.n_splits

        # Create generator to get train-test indices
        curr = 0
        for i in range(self.n_splits):
            start, stop = curr, curr + fold_size + (i < res)

            train_idx = np.concatenate([indices[:start], indices[stop:]])
            test_idx = indices[start:stop]
            yield train_idx, test_idx

            curr = stop


class KFold(BaseKFold):
    pass


class LeaveOneOut(BaseKFold):
    def split(self, X):
        self.n_splits = X.shape[0]
        return super().split(X)
