import numpy as np

from mlspear.mlf import *

class Regression:
    #Parameters:
    # indims  = the number of input dimensions (positive integer)
    # outdims = the number of output dimensions (positive integer)

    #Returns:
    # None, but once Regression layer is initialized, weights and biases are automatically
    # initialized and scaled.
    def __init__(self, indims, outdims):
        self.output  = (lambda z: z)

        self.indims  = indims
        self.outdims = outdims

        self.p = 1

        self.weight_initialize()

    #Parameters:
    # scale_parameters = a boolean that determines whether or not you want to
    # scale your weights and biases. All parameters are scaled by 'He initializing
    # method'

    #Returns:
    # None
    def weight_initialize(self, scale_parameters = True):
        self.W = np.random.randn(self.indims, self.outdims)
        self.B = np.random.randn(1, self.outdims)

        self.Del_W, self.Del_B                   = 0, 0
        self.G_W,   self.G_B                     = 1, 1
        self.M_W,   self.V_W, self.M_B, self.V_B = 0, 0, 0, 0

        if scale_parameters:
            scale = np.sqrt(2 / (self.indims + self.outdims))
            self.W = scale * self.W
            self.B = scale * self.B

    #Parameters:
    # l1 = represents the l1 regularization parameter. (Lasso Regression)
    #      (non-negative real number)
    # l2 = represents the l2 regularization parameter. (Ridge Regression)
    #      (non-negative real number)
    # (Note: if l1 and l2 are non zero positive real, you would use Elastic Net
    #  Regularization)

    #Returns:
    # None, but weights and biases are updated via gradient descent.
    def weight_update(self, l1, l2):
        self.W = self.W + self.Del_W - (l1 * np.sign(self.W)) - (l2 * self.W)
        self.B = self.B + self.Del_B - (l1 * np.sign(self.B)) - (l2 * self.B)

    #Parameters:
    # A = a numpy input matrix from the previous or input layer.
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)

    #Returns:
    # Z = a numpy output matrix after forward propagation.
    def forward(self, A, mtype, mu):
        if mtype == 'nesterov':
            self.W = self.W + mu * self.Del_W
            self.B = self.B + mu * self.Del_B

            self.A = A
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.output(self.H)

            self.W = self.W - mu * self.Del_W
            self.B = self.B - mu * self.Del_B

            return self.Z

        if mtype == 'conventional':
            self.A = A
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.output(self.H)

            return self.Z

    #Parameters:
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
    # D.dot((self.W).T) a numpy matrix backpropagated to the previous layer
    def backward(self, D, lr, mtype, mu, l1, l2):
        self.Del_W = mu * self.Del_W + (-lr * (self.A).T).dot(D)
        self.Del_B = mu * self.Del_B + (-lr * row_sum(D))

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    #Parameters:
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
    # e = small parameter to avoid division by zero. Default value is 1e-9

    #Returns:
    # D.dot((self.W).T) a numpy matrix backpropagated to the previous layer
    def ada_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        self.G_W = self.G_W + ((((self.A).T).dot(D)) ** 2)
        self.G_B = self.G_B + (row_sum(D) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D)
        self.Del_B = mu * self.Del_B - lr_B * row_sum(D)

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    #Parameters:
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
    # e = small parameter to avoid division by zero. Default value is 1e-9
    # g = hyper parameter used for computing moving average of the square of the
    #     gradients. Default value is 0.9

    #Returns:
    # D.dot((self.W).T) a numpy matrix backpropagated to the previous layer
    def rmsprop_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9, g = 0.9):
        self.G_W = g * self.G_W + (1 - g) * ((((self.A).T).dot(D)) ** 2)
        self.G_B = g * self.G_B + (1 - g) * (row_sum(D) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D)
        self.Del_B = mu * self.Del_B - lr_B * row_sum(D)

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    #Parameters:
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
    # b, d = hyper parameter used for computing moving average of the square of the
    #        gradients. Default value is 0.9
    # e = small parameter to avoid division by zero. Default value is 1e-9

    #Returns:
    # D.dot((self.W).T) a numpy matrix backpropagated to the previous layer
    def adam_backward(self, D, lr, mtype, mu, t, l1, l2, b = 0.9, d = 0.9, e = 1e-9):
        self.M_W = b * self.M_W + (1 - b) * ((self.A).T).dot(D)
        self.V_W = d * self.V_W + (1 - d) * ((((self.A).T).dot(D)) ** 2)

        M_W_hat = self.M_W / (1 + (b ** t))
        V_W_hat = self.V_W / (1 + (d ** t))

        Eta_W = lr / np.sqrt(V_W_hat + e)

        self.M_B = b * self.M_B + (1 - b) * (row_sum(D))
        self.V_B = d * self.V_B + (1 - d) * (row_sum(D) ** 2)

        M_B_hat = self.M_B / (1 + (b ** t))
        V_B_hat = self.V_B / (1 + (d ** t))

        Eta_B = lr / np.sqrt(V_B_hat + e)

        self.Del_W = mu * self.Del_W - Eta_W * M_W_hat
        self.Del_B = mu * self.Del_B - Eta_B * M_B_hat

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    #Returns:
    # a python dictionary that contains all parameters in this layer.
    def params(self):
        params = {}
        params['output']     = self.output
        params['W']          = self.W
        params['B']          = self.B
        params['indims']     = self.indims
        params['outdims']    = self.outdims
        params['p']          = self.p

        return params
