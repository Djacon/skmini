import numpy as np

from . import Tensor

from ..base import ClassifierMixin, RegressorMixin
from ..optimizers import Adam


def info(epoch, loss, lr):
    return f'Epoch: {epoch}, Loss: {loss:.6f}, eta: {lr:.6f}'


class Linear:
    '''Simple implementation of Linear layer'''
    def __init__(self, in_features, out_features):
        # Number of input features
        self.in_features = in_features

        # Number of output features
        self.out_features = out_features

        # Initialize new weights from [-k^.5, k^.5]
        sqrt_k = (1 / in_features) ** .5
        self.W = np.random.uniform(-sqrt_k, sqrt_k, (in_features,
                                                     out_features))
        self.b = np.random.uniform(-sqrt_k, sqrt_k, out_features)

        # Transform weights to autograd tensor
        self.W = Tensor(self.W)
        self.b = Tensor(self.b)

    def __call__(self, X):
        '''Return a tensor of output data'''
        return X @ self.W + self.b

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, ' +\
                f'out_features={self.out_features})'


class Sequential:
    '''Simple implementation of Sequential container'''
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, xs):
        out = xs
        for layer in self.layers:
            out = layer(out)
        return out

    def add(self, layer):
        self.layers.append(layer)

    def __repr__(self):
        repr = ''.join(f'  {layer}\n' for layer in self.layers)
        return f'Sequential(\n{repr})'


class BaseMultilayerPerceptron:
    '''Base class for MLP classification and regression'''
    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver=Adam(), alpha=1e-4, batch_size='auto',
                 max_iter=200, random_state=None, verbose=0):
        # Number of neurons in the ith hidden layer
        self.hidden_layer_sizes = hidden_layer_sizes

        # Custom Activation function
        self.activation = activation

        # Selected solver (ex: SGD or Adam)
        self.solver = solver

        # Alpha hyperparam (for l2-regularization)
        self.alpha = alpha

        # Number of batches
        self.batch_size = batch_size

        # The maximum number of passes over the training data (epochs)
        self.max_iter = max_iter

        # Random State (for generating samples)
        self.random_state = random_state

        # The logging level to output to stdout
        self.verbose = verbose

    def fit(self, X, y):
        # Random State (for generating samples)
        self._rng = np.random.RandomState(self.random_state)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Init weights and optimizer
        self._init_weights(X, y)
        self.optim.init(self.W)

        if self.batch_size == 'auto':
            self.batch_size = min(200, X.shape[0])

    # Initialize new weights
    def _init_weights(self, Xs, ys):
        n_features = Xs.shape[1]
        self.W = np.zeros(n_features)
        self.b = np.zeros((1,))
