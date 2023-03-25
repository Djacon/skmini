import numpy as np
from ._split import KFold


def cross_val_predict(estimator, X, y, cv=5):
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in KFold(cv).split(X):
        estimator.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = estimator.predict(X[test_idx])
    return y_pred
