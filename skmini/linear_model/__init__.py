from ._regression import LinearRegression
from ._regression import Ridge, RidgeCV
from ._regression import Lasso, LassoCV
from ._regression import ElasticNet, ElasticNetCV
from ._regression import SGDRegressor
from ._regression import HuberRegressor

from ._classification import LinearSVC
from ._classification import Perceptron
from ._classification import SGDClassifier
from ._classification import LogisticRegression

__all__ = [
    'LinearRegression',
    'Ridge',
    'RidgeCV',
    'Lasso',
    'LassoCV',
    'ElasticNet',
    'ElasticNetCV',
    'SGDRegressor',
    'HuberRegressor',

    'LinearSVC',
    'Perceptron',
    'SGDClassifier',
    'LogisticRegression',
]
