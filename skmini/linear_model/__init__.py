from ._regression import LinearRegression
from ._regression import Ridge
from ._regression import Lasso
from ._regression import ElasticNet
from ._regression import SGDRegressor
from ._regression import HuberRegressor

from ._classification import LinearSVC
from ._classification import Perceptron
from ._classification import SGDClassifier
from ._classification import LogisticRegression

__all__ = [
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'SGDRegressor',
    'HuberRegressor',

    'LinearSVC',
    'Perceptron',
    'SGDClassifier',
    'LogisticRegression',
]
