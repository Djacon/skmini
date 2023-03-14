import numpy as np


'''Optimizers'''


class SGD:
    '''Stochastic Gradient Descent'''
    def __init__(self, lr=1e-4):
        self.lr = lr

    def init(self, _):
        pass

    def update(self, dW, db, _):
        return self.lr * dW, self.lr * db


class RMSProp:
    '''Root Mean Squared Propagation'''
    def __init__(self, lr=1e-4, beta=0.99, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.lr = lr

    def init(self, W):
        self.sw = np.zeros_like(W)
        self.sb = 0

    def update(self, dW, db, _):
        self.sw = self.beta * self.sw + (1 - self.beta) * dW * dW
        self.sb = self.beta * self.sb + (1 - self.beta) * db * db

        dW_new = dW / (np.sqrt(self.sw) + self.eps)
        db_new = db / (np.sqrt(self.sb) + self.eps)

        return self.lr * dW_new, self.lr * db_new


class Adam:
    '''ADAptive Moment estimator'''
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr

    def init(self, W):
        self.vw = np.zeros_like(W)
        self.vb = 0

        self.sw = np.zeros_like(W)
        self.sb = 0

    def update(self, dW, db, t):
        self.vw = self.beta1 * self.vw + (1 - self.beta1) * dW
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * db

        vw_norm = self.vw / (1 - self.beta1 ** t)
        vb_norm = self.vb / (1 - self.beta1 ** t)

        self.sw = self.beta2 * self.sw + (1 - self.beta2) * dW * dW
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * db * db

        sw_norm = self.sw / (1 - self.beta2 ** t)
        sb_norm = self.sb / (1 - self.beta2 ** t)

        dW_new = vw_norm / (np.sqrt(sw_norm) + self.eps)
        db_new = vb_norm / (np.sqrt(sb_norm) + self.eps)

        return self.lr * dW_new, self.lr * db_new
