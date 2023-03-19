import numpy as np

from ._base import LinearModel

from ..metrics import BCE, Squared_hinge, Hinge
from ..optimizers import Adam, SGD
from ..base import ClassifierMixin


'''Classification Linear models'''


class LogisticRegression(ClassifierMixin, LinearModel):
    '''Logistic Regression model'''
    def __init__(self, penalty='l2', C=1., max_iter=100, l1_ratio=.15,
                 optim=Adam(), batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=BCE(), C=C,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)

    def predict(self, Xs):
        y = Xs @ self.W + self.b
        return 1 / (1 + np.exp(-y))


class LinearSVC(ClassifierMixin, LinearModel):
    '''Linear SVC model'''
    def __init__(self, eval_metric=Squared_hinge(), penalty='l2', C=1.,
                 max_iter=1000, optim=SGD(), batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)


class SGDClassifier(ClassifierMixin, LinearModel):
    '''SGD Classifier model'''
    def __init__(self, eval_metric=Hinge(), penalty='l2', C=1.,
                 max_iter=1000, optim=SGD(), batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)


class Perceptron(ClassifierMixin, LinearModel):
    '''Perceptron Classifier model'''
    def __init__(self, penalty='l2', C=1., max_iter=1000, optim=SGD(),
                 batch_size=10, verbose=100):
        super().__init__(penalty=penalty, eval_metric=Hinge(0), C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)
