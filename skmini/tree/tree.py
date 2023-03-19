import numpy as np
from statistics import mode

from ..base import ClassifierMixin, RegressorMixin


'''Criterions'''


def entropy(y):
    ps = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    ps = np.bincount(y) / len(y)
    return 1 - (ps**2).sum()


CRITERIA_CLF = {
    'entropy': entropy,
    'gini': gini,
}

CRITERIA_REG = {
    'squared_error': lambda y: ((y - y.mean()) ** 2).mean(),  # or y.var()
    'absolute_error': lambda y: np.abs(y - np.median(y)).mean()
}


'''Base Decision Tree model'''


class Node:
    '''Node of Decision Tree'''
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_note(self):
        return self.value is not None

    def __repr__(self):
        return f'Node(feature={self.feature}, threshold={self.threshold},\
            value={self.value})'


class BaseDecisionTree:
    '''Base class for decision trees'''
    def __init__(self, criterion=None, max_depth=100, min_samples_split=2,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        self.random_state = random_state

    def fit(self, X, y):
        # Check max_features variable
        if self.max_features:
            self.max_features = min(self.max_features, X.shape[1])
        else:
            self.max_features = X.shape[1]

        # Check criterion variable
        if self._estimator_type == 'classifier':
            self._criterion = CRITERIA_CLF[self.criterion]
        else:
            self._criterion = CRITERIA_REG[self.criterion]

        # Specify seed and "grow" tree
        self._rng = np.random.RandomState(self.random_state)
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def get_params(self):
        return {'criterion': self.criterion,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'criterion': self.criterion,
                'random_state': self.random_state}

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or
           n_samples < self.min_samples_split):
            leaf_value = self._get_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = self._rng.choice(n_features, self.max_features,
                                     replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = self._get_thresholds(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_thresh = feat_idx, threshold

        return split_idx, split_thresh

    def _get_thresholds(self, X):
        thresh = np.unique(X)
        thresh.sort()

        averages = [round((thresh[i] + thresh[i+1]) / 2, 6)
                    for i in range(len(thresh)-1)]
        return np.concatenate([averages, thresh])

    def _information_gain(self, X, y, threshold):
        parent = self.criterion(y)

        left_idxs, right_idxs = self._split(X, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        childs = len(left_idxs) * self._criterion(y[left_idxs]) +\
            len(right_idxs) * self._criterion(y[right_idxs])
        return parent - childs / len(y)

    def _split(self, X, threshold):
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_note():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    '''Decision Tree Classifier model'''
    def __init__(self, criterion='gini', max_depth=100, min_samples_split=2,
                 max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         max_features, random_state)

    def _get_leaf_value(self, y):
        return mode(y)


class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    '''Decision Tree Regressor model'''
    def __init__(self, criterion='squared_error', max_depth=100,
                 min_samples_split=2, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split,
                         max_features, random_state)

    def _get_leaf_value(self, y):
        return y.mean()
