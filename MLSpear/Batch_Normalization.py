from MLSpear.MLF import *
import numpy as np

class Batch_Normalization:

    def __init__(self):
        self.mu    = 0
        self.sigma = 1

    #Returns:
    # None but initializes Gamma and Beta
    def weight_initialize(self, scale_weights = True):
        self.Gamma = 1
        self.Beta  = 0

    #Parameters:
    # lr     = learning rate
    # A_norm = normalized input matrix
    # D      = numpy backpropagated matrix

    #Returns:
    # None but updates Gamma and Beta
    def weight_update(self, lr, A_norm, D):
        self.Gamma = self.Gamma - lr * sumToRow(A_norm.T.dot(D))
        self.Beta  = self.Beta  - lr * sumToRow(D)

    #Parameters:
    # A     = input numpy matrix
    # mtype = momentum type ('conventional' or 'nesterov')
    # mu    = momentum value
    # e     = small positive real number to prevent division by zero.
    # alpha =

    #Returns:
    # a numpy matrix for forward propagation.
    def train_forward(self, A, mtype, mu, e = 1e-9, alpha = 0.9):
        self.A  = A

        mu_b = np.mean(A, axis = 0)
        mu_b = mu_b.reshape((1, mu_b.shape[0]))

        sigma_b = np.std(A, axis = 0)
        sigma_b = sigma_b.reshape((1, sigma_b.shape[0]))

        A_norm  = (A - mu_b) / np.sqrt((sigma_b ** 2) + e)

        self.mu = alpha * self.mu + (1 - alpha) * mu_b
        self.sigma = np.sqrt(alpha * (self.sigma ** 2) + (1 - alpha) * (sigma_b ** 2))

        return self.Gamma * (A_norm) + self.Beta

    #Parameters:
    # A     = input numpy matrix
    # mtype = momentum type ('conventional' or 'nesterov')
    # mu    = momentum value
    # e     = small positive real number to prevent division by zero.

    #Returns:
    # a numpy matrix for forward propagation.
    def forward(self, A, mtype, mu, e = 1e-9):
        A_norm  = (A - self.mu) / np.sqrt((self.sigma ** 2) + e)

        return self.Gamma * (A_norm) + self.Beta

    #Paramters:
    # D  = a numpy matrix that is produced by backpropagtion from the layer
    #      ahead.
    # lr = learning rate
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)
    # l1 = represents the l1 regularization parameter. (Lasso Regression)
    #      (non-negative real number)
    # l2 = represents the l2 regularization parameter. (Ridge Regression)
    #      (non-negative real number)

    #Returns:
    # a backpropagated numpy matrix.
    def backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        A_norm  = (self.A - self.mu) / np.sqrt((self.sigma ** 2) + e)
        self.weight_update(lr, A_norm, D)

        return D * self.Gamma * (1 / np.sqrt((self.sigma ** 2) + e))

    #Paramters:
    # D  = a numpy matrix that is produced by backpropagtion from the layer
    #      ahead.
    # lr = learning rate
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)
    # l1 = represents the l1 regularization parameter. (Lasso Regression)
    #      (non-negative real number)
    # l2 = represents the l2 regularization parameter. (Ridge Regression)
    #      (non-negative real number)

    #Returns:
    # a backpropagated numpy matrix.
    def ada_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)

    #Paramters:
    # D  = a numpy matrix that is produced by backpropagtion from the layer
    #      ahead.
    # lr = learning rate
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)
    # l1 = represents the l1 regularization parameter. (Lasso Regression)
    #      (non-negative real number)
    # l2 = represents the l2 regularization parameter. (Ridge Regression)
    #      (non-negative real number)

    #Returns:
    # a backpropagated numpy matrix.
    def rmsprop_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)

    #Paramters:
    # D  = a numpy matrix that is produced by backpropagtion from the layer
    #      ahead.
    # lr = learning rate
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)
    # l1 = represents the l1 regularization parameter. (Lasso Regression)
    #      (non-negative real number)
    # l2 = represents the l2 regularization parameter. (Ridge Regression)
    #      (non-negative real number)

    #Returns:
    # a backpropagated numpy matrix.
    def adam_backward(self, D, lr, mtype, mu, t, l1, l2, e = 1e-9):
        return self.backward(D, lr, mtype, mu, l1, l2, e = 1e-9)
