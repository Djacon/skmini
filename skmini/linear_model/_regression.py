from ._base import LinearModel

from ..metrics import MSE, Huber
from ..optimizers import Adam, SGD
from ..base import RegressorMixin


'''Regression Linear models'''


class LinearRegression(RegressorMixin, LinearModel):
    '''Simple Linear Regression model'''
    def __init__(self, max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class Lasso(RegressorMixin, LinearModel):
    '''Lasso model'''
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l1', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class Ridge(RegressorMixin, LinearModel):
    '''Ridge model'''
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class ElasticNet(RegressorMixin, LinearModel):
    '''ElasticNet model'''
    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=1000, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='elasticnet', alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class SGDRegressor(RegressorMixin, LinearModel):
    '''SGD Regressor model'''
    def __init__(self, eval_metric=MSE(), penalty='l2', alpha=1e-4, lr=1e-4,
                 max_iter=1000, l1_ratio=.15, batch_size=10, verbose=1000):
        super().__init__(eval_metric=eval_metric, penalty=penalty, alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter,
                         optim=SGD(lr=lr), batch_size=batch_size,
                         verbose=verbose)


class HuberRegressor(RegressorMixin, LinearModel):
    '''Huber Regressor model'''
    def __init__(self, epsilon=1.35, max_iter=100, alpha=1e-4, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=Huber(epsilon), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)
