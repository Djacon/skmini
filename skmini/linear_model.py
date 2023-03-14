import numpy as np
from random import choices

from .metrics import MSE, HUBER, BCE, Squared_Hinge
from .optimizers import Adam, SGD
from .base_model import BaseRegressor, BaseClassifier


def info(epoch, loss, lr):
    return f'Epoch: {epoch}, Loss: {loss:.6f}, eta: {lr:.6f}'


'''Regression models'''


class Linear(BaseRegressor):
    '''Base Linear Regressor'''
    def __init__(self, eval_metric=MSE(), penalty='', alpha=1e-4, l1_ratio=.15,
                 C=0, max_iter=10_000, optim=Adam(), batch_size=10,
                 verbose=1000):
        # Weights & biases
        self.W = []
        self.b = 0

        # Calculating loss and gradient of the model
        self.loss = eval_metric.loss
        self.grad = eval_metric.grad

        # Alpha hyperparam (when penalty != 'None')
        if C:
            self.alpha = 1 / C
        else:
            self.alpha = alpha

        # L1-ratio (when penalty = 'elasticnet')
        self.l1_ratio = l1_ratio

        # The maximum number of passes over the training data (epochs)
        self.max_iter = max_iter

        # Number of batches
        self.batch_size = batch_size

        # Custom gradient function for each case
        if penalty == 'l1':
            self._get_grad = self._grad_l1
        elif penalty == 'l2':
            self._get_grad = self._grad_l2
        elif penalty == 'elasticnet':
            self._get_grad = self._grad_elastic
        else:
            self._get_grad = self._grad

        # Selected solver (ex: SGD or Adam)
        self.optim = optim

        # When set to True, use solution of the previous call to fit as init
        self.warm_start = False

        # The logging level to output to stdout
        self.verbose = verbose

    # Xs: matrix [NxM], ys: vector [Nx1]
    def fit(self, Xs, ys):
        if not self.warm_start:
            self.warm_start = True
            self.W = np.zeros(len(Xs[0]))
            self.b = 0

            self.optim.init(self.W)

        # Convert to numpy arrays
        Xs = np.array(Xs)
        ys = np.array(ys)

        # Start Iterative learning
        for epoch in range(self.max_iter):
            batches = choices(range(len(ys)), k=self.batch_size)
            Xs_batch = Xs[batches]
            ys_batch = ys[batches]

            y_preds = self.predict(Xs_batch)
            Y_pred = self.grad(ys_batch, y_preds)

            loss = self.loss(ys_batch, y_preds).mean()

            dW, db = self._get_grad(Y_pred, Xs_batch)
            dW, db = self.optim.update(dW, db, epoch + 1)

            self.W -= dW
            self.b -= db

            if self.verbose and epoch % self.verbose == 0:
                print(info(epoch, loss, self.optim.lr))

        print('Final ' + info(epoch, loss, self.optim.lr))

    # Gradients for weights and bias
    def _grad(self, Y_pred, Xs):
        dW = (Y_pred @ Xs) / Y_pred.shape[0]
        db = Y_pred.mean()
        return dW, db

    # L2-regularization
    def _grad_l2(self, Y_pred, Xs):
        dW, db = self._grad(Y_pred, Xs)
        dW += 2 * self.alpha * self.W
        return dW, db

    # L1-regularization
    def _grad_l1(self, Y_pred, Xs):
        dW, db = self._grad(Y_pred, Xs)
        dW += self.alpha * np.sign(self.W)
        return dW, db

    # Elastic-net
    def _grad_elastic(self, Y_pred, Xs):
        dW, db = self._grad(Y_pred, Xs)
        dW += self.alpha * (self.l1_ratio * np.sign(self.W) +
                            (1 - self.l1_ratio) * self.W)
        return dW, db

    def predict(self, Xs):
        return Xs @ self.W + self.b


class LinearRegression(Linear):
    def __init__(self, max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class Lasso(Linear):
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l1', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class Ridge(Linear):
    def __init__(self, alpha=1., max_iter=1000, optim=Adam(), batch_size=10,
                 verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


class ElasticNet(Linear):
    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=1000, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=MSE(), penalty='elasticnet', alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size, verbose=verbose)


class SGDRegressor(Linear):
    def __init__(self, eval_metric=MSE(), penalty='l2', alpha=1e-4, lr=1e-4,
                 max_iter=1000, l1_ratio=.15, batch_size=10, verbose=1000):
        super().__init__(eval_metric=eval_metric, penalty=penalty, alpha=alpha,
                         l1_ratio=l1_ratio, max_iter=max_iter,
                         optim=SGD(lr=lr), batch_size=batch_size,
                         verbose=verbose)


class HuberRegressor(Linear):
    def __init__(self, epsilon=1.35, max_iter=100, alpha=1e-4, optim=Adam(),
                 batch_size=10, verbose=1000):
        super().__init__(eval_metric=HUBER(epsilon), penalty='l2', alpha=alpha,
                         max_iter=max_iter, optim=optim, batch_size=batch_size,
                         verbose=verbose)


'''Classification models'''


class LogisticRegression(BaseClassifier, Linear):
    '''Logistic Regression model'''
    def __init__(self, penalty='l2', C=1., max_iter=100, l1_ratio=.15,
                 optim=Adam(), batch_size=10):
        super().__init__(penalty=penalty, eval_metric=BCE(), C=C,
                         l1_ratio=l1_ratio, max_iter=max_iter, optim=optim,
                         batch_size=batch_size)

    def predict(self, Xs):
        y = Xs @ self.W + self.b
        return 1 / (1 + np.exp(-y))


class LinearSVC(BaseClassifier, Linear):
    def __init__(self, eval_metric=Squared_Hinge(), penalty='l2', C=1.,
                 max_iter=1000, optim=SGD(), batch_size=10):
        super().__init__(penalty=penalty, eval_metric=eval_metric, C=C,
                         max_iter=max_iter, optim=optim, batch_size=batch_size)

    def predict(self, Xs):
        return np.sign(Xs @ self.W + self.b)
