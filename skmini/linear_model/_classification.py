import numpy as np

from ._base import LinearModel

from ..metrics import Softmax, Squared_hinge, Hinge
from ..optimizers import Adam, SGD
from ..base import ClassifierMixin


'''Classification Linear models'''


class LogisticRegression(ClassifierMixin, LinearModel):
    '''Logistic Regression model'''
    def __init__(self, penalty='l2', C=1., max_iter=1000, l1_ratio=.15,
                 optim=Adam(), batch_size=10, random_state=None, verbose=0):
        super().__init__(penalty=penalty, eval_metric=Softmax(), C=C,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, random_state=random_state,
                         verbose=verbose)

    def _init_weights(self, Xs, ys):
        n_features = Xs.shape[1]

        self.classes = np.unique(ys)
        self.n_classes = len(self.classes)

        if self.n_classes > 50:
            raise ValueError('Too many classes!')

        self.W = np.zeros((n_features, self.n_classes))
        self.b = np.zeros((self.n_classes,))

    def predict_proba(self, Xs):
        y = np.exp(Xs @ self.W + self.b)
        return y / y.sum(axis=1, keepdims=True)

    def _predict(self, Xs):
        return self.predict_proba(Xs)

    def predict(self, Xs):
        y_proba = self.predict_proba(Xs)
        return np.argmax(y_proba, axis=1)


class LinearSVC(ClassifierMixin, LinearModel):
    '''Linear SVC model'''
    def __init__(self, eval_metric=Squared_hinge(), penalty='l2',
                 C=1., max_iter=1000, optim=SGD(), batch_size=10,
                 random_state=None, verbose=0):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         random_state=random_state, verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)


class SGDClassifier(ClassifierMixin, LinearModel):
    '''SGD Classifier model'''
    def __init__(self, eval_metric=Hinge(), penalty='l2', C=1.,
                 max_iter=1000, optim=SGD(), batch_size=10,
                 random_state=None, verbose=0):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         random_state=random_state, verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)


class Perceptron(ClassifierMixin, LinearModel):
    '''Perceptron Classifier model'''
    def __init__(self, penalty='l2', C=1., max_iter=1000, optim=SGD(),
                 batch_size=10, random_state=None, verbose=0):
        super().__init__(penalty=penalty, eval_metric=Hinge(0), C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         random_state=random_state, verbose=verbose)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)
