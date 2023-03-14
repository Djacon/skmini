import numpy as np

from ._base import Linear

from ..metrics import MSE, Huber, BCE, Squared_hinge
from ..optimizers import Adam, SGD
from ..base import BaseClassifier


'''Regression models'''


class LinearRegression(Linear):
    '''Simple Linear Regression model'''
    def __init__(self, max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class Lasso(Linear):
    '''Lasso model'''
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l1', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class Ridge(Linear):
    '''Ridge model'''
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class ElasticNet(Linear):
    '''ElasticNet model'''
    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=1000, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='elasticnet', alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class SGDRegressor(Linear):
    '''SGD Regressor model'''
    def __init__(self, eval_metric=MSE(), penalty='l2', alpha=1e-4, lr=1e-4,
                 max_iter=1000, l1_ratio=.15, batch_size=10, verbose=1000):
        super().__init__(eval_metric=eval_metric, penalty=penalty, alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter,
                         optim=SGD(lr=lr), batch_size=batch_size,
                         verbose=verbose)


class HuberRegressor(Linear):
    '''Huber Regressor model'''
    def __init__(self, epsilon=1.35, max_iter=100, alpha=1e-4, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=Huber(epsilon), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


'''Classification models'''


class LogisticRegression(BaseClassifier, Linear):
    '''Logistic Regression model'''
    def __init__(self, penalty='l2', C=1., max_iter=100, l1_ratio=.15,
                 optim=Adam(), batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=BCE(), C=C,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)

    def predict(self, Xs):
        y = Xs @ self.W + self.b
        return 1 / (1 + np.exp(-y))


class LinearSVC(BaseClassifier, Linear):
    '''Linear SVC model'''
    def __init__(self, eval_metric=Squared_hinge(), penalty='l2', C=1.,
                 max_iter=1000, optim=SGD(), batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)
