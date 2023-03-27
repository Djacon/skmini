import numpy as np


class BaseEstimator:
    def clone(self):
        return type(self)(**self.get_params())

    # def get_params(self):
    #     raise NotImplementedError('Subclasses must implement get_params()')


class RegressorMixin(BaseEstimator):
    '''Mixin class for all regressors in scikit-learn'''

    _estimator_type = 'regressor'

    def score(self, X, y):
        '''R2-score'''
        y = np.array(y)
        y_pred = self.predict(X)
        RSS = ((y - y_pred) ** 2).sum()
        TSS = ((y - y.mean()) ** 2).sum()
        return round(1 - RSS/TSS, 4)


class ClassifierMixin(BaseEstimator):
    '''Mixin class for all classifiers in scikit-learn'''

    _estimator_type = 'classifier'

    def score(self, X, y):
        '''Accuracy score'''
        y_pred = self.predict(X).round()
        return round((y == y_pred).mean(), 4)


class TransformerMixin(BaseEstimator):
    '''Mixin class for all transformers in scikit-learn'''

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


class ClusterMixin(BaseEstimator):
    """Mixin class for all cluster estimators in scikit-learn."""

    _estimator_type = 'clusterer'
