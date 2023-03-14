import numpy as np
from random import choices

from ..metrics import MSE
from ..optimizers import Adam


'''Base models'''


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
        return (y == y_pred).mean()


def info(epoch, loss, lr):
    return f'Epoch: {epoch}, Loss: {loss:.6f}, eta: {lr:.6f}'


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
