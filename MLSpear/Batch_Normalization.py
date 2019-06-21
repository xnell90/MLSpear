from MLSpear.MLF import *
import numpy as np

class Batch_Normalization:

    def __init__(self):
        self.mu    = 0
        self.sigma = 1

    def weight_initialize(self, scale_weights = True):
        self.Gamma = 1
        self.Beta  = 0

    def weight_update(self, lr, A_norm, D):
        self.Gamma = self.Gamma - lr * sumToRow(A_norm.T.dot(D))
        self.Beta = self.Beta - lr * sumToRow(D)

    def train_forward(self, A, mtype, mu, e = 1e-9, alpha = 0.9):
        self.A  = A
        mu_b    = np.mean(A, axis = 0)
        mu_b    = mu_b.reshape((1, mu_b.shape[0]))
        sigma_b = np.std(A, axis = 0)
        sigma_b = sigma_b.reshape((1, sigma_b.shape[0]))
        A_norm  = (A - mu_b) / np.sqrt((sigma_b ** 2) + e)

        self.mu = alpha * self.mu + (1 - alpha) * mu_b
        self.sigma = np.sqrt(alpha * (self.sigma ** 2)
                             + (1 - alpha) * (sigma_b ** 2))

        return self.Gamma * (A_norm) + self.Beta

    def forward(self, A, mtype, mu, e = 1e-9):
        A_norm  = (A - self.mu) / np.sqrt((self.sigma ** 2) + e)

        return self.Gamma * (A_norm) + self.Beta

    def backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        A_norm  = (self.A - self.mu) / np.sqrt((self.sigma ** 2) + e)
        self.weight_update(lr, A_norm, D)

        return D * self.Gamma * (1 / np.sqrt((self.sigma ** 2) + e))

    def ada_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)

    def rmsprop_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)

    def adam_backward(self, D, lr, mtype, mu, t, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)
