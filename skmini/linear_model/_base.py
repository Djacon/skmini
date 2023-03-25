import numpy as np

from ..metrics import MSE
from ..optimizers import Adam
from ..model_selection import KFold


def info(epoch, loss, lr):
    return f'Epoch: {epoch}, Loss: {loss:.6f}, eta: {lr:.6f}'


class LinearModel:
    '''Base class for Linear Models'''
    def __init__(self, eval_metric=MSE(), penalty='', alpha=1e-4, l1_ratio=.15,
                 C=0, max_iter=10_000, optim=Adam(), batch_size=10,
                 random_state=None, verbose=0):
        # Weights & biases
        self.W = []
        self.b = 0

        # Calculating loss and gradient of the model
        self.loss = eval_metric.loss
        self.grad = eval_metric.grad

        # Alpha hyperparam (when penalty != 'None')
        if C:
            self.alpha = 1 / (2*C)
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

        # Random State (for generating samples)
        self.random_state = random_state

        # The logging level to output to stdout
        self.verbose = verbose

    # Xs: matrix [NxM], ys: vector [NxK]
    def fit(self, Xs, ys):
        # Random State (for generating samples)
        self._rng = np.random.RandomState(self.random_state)

        # Convert to numpy arrays
        Xs = np.array(Xs)
        ys = np.array(ys)

        # Init weights and optimizer
        self._init_weights(Xs, ys)
        self.optim.init(self.W)

        # Start Iterative learning
        for epoch in range(self.max_iter):
            batches = self._rng.choice(len(ys), self.batch_size)
            # batches = choices(range(len(ys)), k=self.batch_size)
            Xs_batch = Xs[batches]
            ys_batch = ys[batches]

            y_preds = self._predict(Xs_batch)
            Y_pred = self.grad(ys_batch, y_preds)

            loss = self.loss(ys_batch, y_preds).mean()

            dW, db = self._get_grad(Y_pred, Xs_batch)
            dW, db = self.optim.update(dW, db, epoch + 1)

            self.W -= dW
            self.b -= db

            if self.verbose and epoch % self.verbose == 0:
                print(info(epoch, loss, self.optim.lr))

        if self.verbose:
            print('Final ' + info(epoch, loss, self.optim.lr))

    # Initialize new weights
    def _init_weights(self, Xs, ys):
        n_features = Xs.shape[1]
        self.W = np.zeros(n_features)
        self.b = np.zeros((1,))

    # Gradients for weights and bias
    def _grad(self, Y_pred, Xs):
        dW = (Xs.T @ Y_pred) / Y_pred.shape[0]
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

    def _predict(self, Xs):
        return self.predict(Xs)

    def predict(self, Xs):
        return Xs @ self.W + self.b


class LinearModelCV:
    '''Base Linear model with cross-validation'''
    def __init__(self, estimator=None, alphas=[1.], cv=5):
        self.estimator = estimator
        self.alphas = alphas
        self.cv = cv

    def fit(self, X, y):
        best_alpha = 0
        best_score = float('-inf')

        # Cross-validator
        kf = KFold(self.cv, shuffle=True,
                   random_state=self.estimator.random_state)
        folds = list(kf.split(X, y))

        # Make cross-validation to find out best alpha
        for alpha in self.alphas:
            self.estimator.alpha = alpha  # my small glitch :)

            scores = []
            for train_idx, test_idx in folds:
                self.estimator.fit(X[train_idx], y[train_idx])
                score = self.estimator.score(X[test_idx], y[test_idx])
                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Set best alpha hyperparam
        self.estimator.alpha = best_alpha
        self.estimator.fit(X, y)

        self.alpha_ = best_alpha

    def predict(self, X):
        return self.estimator.predict(X)
