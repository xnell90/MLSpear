from MLSpear.MLF import *
import numpy as np

class PReLU:
    #Parameters:
    # indims  = the number of input dimensions (positive integer)
    # outdims = the number of output dimensions (positive integer)
    # p       = the drop out rate. Default value is 1.

    #Returns:
    # None, but once PReLU layer is initialized, weights and biases are automatically
    # initialized and scaled.
    def __init__(self, indims, outdims, p = 1):
        self.activation = PReLu
        self.derivative_a = PReLu_derivative_a
        self.derivative_p = PReLu_derivative_p

        self.indims = indims
        self.outdims = outdims

        self.p = p

        self.weight_initialize()

    #Parameters:
    # scale_parameters = a boolean that determines whether or not you want to
    # scale your weights and biases. All parameters are scaled by 'He initializing
    # method'

    #Returns:
    # None but initializes the weights and biases
    def weight_initialize(self, scale_parameters = True):
        self.W = np.random.randn(self.indims, self.outdims)
        self.B = np.random.randn(1, self.outdims)
        self.P = np.random.randn(1, self.outdims)

        self.Del_W, self.Del_B, self.Del_P = 0, 0, 0
        self.G_W,   self.G_B,   self.G_P   = 1, 1, 1
        self.M_W,   self.M_B,   self.M_P   = 0, 0, 0
        self.V_W,   self.V_B,   self.V_P   = 0, 0, 0

        if scale_parameters:
            scale = np.sqrt(2 / (self.indims + self.outdims))
            self.W = scale * self.W
            self.B = scale * self.B
            self.P = scale * self.P

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
        self.P = self.P + self.Del_P - (l2 * np.sign(self.P)) - (l2 * self.P)

    #Parameters:
    # A = a numpy input matrix from the previous or input layer.
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu = the amount of momentum added to the gradient descent algorithm.
    #      (non-negative real number)

    #Returns:
    # Z = a numpy output matrix after forward propagation.
    def dropout_forward(self, A, mtype, mu):
        if mtype == 'nesterov':
            self.W = self.W + mu * self.Del_W
            self.B = self.B + mu * self.Del_B
            self.P = self.P + mu * self.Del_P

            M = np.random.rand(*A.shape) < self.p
            self.A = A * M
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.activation(self.P, self.H)

            self.W = self.W - mu * self.Del_W
            self.B = self.B - mu * self.Del_B
            self.P = self.P - mu * self.Del_P

            return self.Z

        if mtype == 'conventional':
            M = np.random.rand(*A.shape) < self.p
            self.A = A * M
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.activation(self.P, self.H)

            return self.Z

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
            self.P = self.P + mu * self.Del_P

            self.A = A
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.activation(self.P, self.H)

            self.W = self.W - mu * self.Del_W
            self.B = self.B - mu * self.Del_B
            self.P = self.P - mu * self.Del_P

            return self.p * self.Z

        if mtype == 'conventional':
            self.A = A
            self.H = (self.A).dot(self.W) + self.B
            self.Z = self.activation(self.P, self.H)

            return self.p * self.Z

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
        self.Del_W = mu * self.Del_W + (-lr * (self.A).T).dot(D * self.derivative_a(self.P, self.H))
        self.Del_B = mu * self.Del_B + (-lr * sumToRow(D * self.derivative_a(self.P, self.H)))
        self.Del_P = mu * self.Del_P + (-lr * sumToRow(D * self.derivative_p(self.H)))

        self.weight_update(l1, l2)

        return (D * (self.derivative_a(self.P, self.H))).dot((self.W).T)

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
        self.G_W = self.G_W + ((((self.A).T).dot(D * self.derivative_a(self.P, self.H))) ** 2)
        self.G_B = self.G_B + (sumToRow(D * self.derivative_a(self.P, self.H)) ** 2)
        self.G_P = self.G_P + (sumToRow(D * self.derivative_p(self.H)) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)
        lr_P = lr / np.sqrt(self.G_P + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D * self.derivative_a(self.P, self.H))
        self.Del_B = mu * self.Del_B - lr_B * (sumToRow(D * self.derivative_a(self.P, self.H)))
        self.Del_P = mu * self.Del_P - lr_P * (sumToRow(D * self.derivative_p(self.H)))

        self.weight_update(l1, l2)

        return (D * (self.derivative_a(self.P, self.H))).dot((self.W).T)

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
        self.G_W = g * self.G_W + (1 - g) * ((((self.A).T).dot(D * self.derivative_a(self.P, self.H))) ** 2)
        self.G_B = g * self.G_B + (1 - g) * (sumToRow(D * self.derivative_a(self.P, self.H)) ** 2)
        self.G_P = g * self.G_P + (1 - g) * (sumToRow(D * self.derivative_p(self.H)) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)
        lr_P = lr / np.sqrt(self.G_P + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D * self.derivative_a(self.P, self.H))
        self.Del_B = mu * self.Del_B - lr_B * (sumToRow(D * self.derivative_a(self.P, self.H)))
        self.Del_P = mu * self.Del_P - lr_P * (sumToRow(D * self.derivative_p(self.H)))

        self.weight_update(l1, l2)

        return (D * (self.derivative_a(self.P, self.H))).dot((self.W).T)

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
        self.M_W = b * self.M_W + (1 - b) * ((self.A).T).dot(D * self.derivative_a(self.P, self.H))
        self.V_W = d * self.V_W + (1 - d) * ((((self.A).T).dot(D * self.derivative_a(self.P, self.H))) ** 2)

        M_W_hat = self.M_W / (1 + (b ** t))
        V_W_hat = self.V_W / (1 + (d ** t))

        Eta_W = lr / np.sqrt(V_W_hat + e)

        self.M_B = b * self.M_B + (1 - b) * (sumToRow(D * self.derivative_a(self.P, self.H)))
        self.V_B = d * self.V_B + (1 - d) * ((sumToRow(D * self.derivative_a(self.P, self.H))) ** 2)

        M_B_hat = self.M_B / (1 + (b ** t))
        V_B_hat = self.V_B / (1 + (d ** t))

        Eta_B = lr / np.sqrt(V_B_hat + e)

        self.M_P = b * self.M_P + (1 - b) * (sumToRow(D * self.derivative_p(self.H)))
        self.V_P = d * self.V_P + (1 - d) * ((sumToRow(D * self.derivative_p(self.H))) ** 2)

        M_P_hat = self.M_P / (1 + (b ** t))
        V_P_hat = self.V_P / (1 + (d ** t))

        Eta_P = lr / np.sqrt(V_P_hat + e)

        self.Del_W = mu * self.Del_W - Eta_W * M_W_hat
        self.Del_B = mu * self.Del_B - Eta_B * M_B_hat
        self.Del_P = mu * self.Del_P - Eta_P * M_P_hat

        self.weight_update(l1, l2)

        return (D * (self.derivative_a(self.P, self.H))).dot((self.W).T)

    #Returns:
    # a python dictionary that contains all parameters in this layer.
    def params(self):
        params = {}
        params['activation']   = self.activation
        params['derivative_a'] = self.derivative_a
        params['derivative_p'] = self.derivative_p
        params['W']            = self.W
        params['B']            = self.B
        params['P']            = self.P
        params['indims']       = self.indims
        params['outdims']      = self.outdims
        params['p']            = self.p

        return params
