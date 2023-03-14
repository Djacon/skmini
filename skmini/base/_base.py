import numpy as np


class BaseRegressor:
    '''Base Regressor model'''

    # R2-score
    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        RSS = ((y - y_pred) ** 2).sum()
        TSS = ((y - y.mean()) ** 2).sum()
        return round(1 - RSS/TSS, 4)


class BaseClassifier:
    '''Base Classifier model'''

    # Accuracy score
    def score(self, X, y):
        y_pred = self.predict(X).round()
        return round((y == y_pred).mean(), 4)
