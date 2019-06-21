from MLSpear.MLF import *
import numpy as np

class Regression:

    def __init__(self, indims, outdims):
        self.output  = identity

        self.indims  = indims
        self.outdims = outdims

        self.p = 1

        self.weight_initialize()

    def weight_initialize(self, scale_parameters = True):
        self.W = np.random.randn(self.indims, self.outdims)
        self.B = np.random.randn(1, self.outdims)

        self.Del_W = 0
        self.Del_B = 0

        self.G_W = 1
        self.G_B = 1

        self.M_W = 0
        self.V_W = 0
        self.M_B = 0
        self.V_B = 0

        if scale_parameters:
            scale = np.sqrt(2 / (self.indims + self.outdims))
            self.W = scale * self.W
            self.B = scale * self.B

    def weight_update(self, l1, l2):
        self.W = self.W + self.Del_W - (l1 * np.sign(self.W)) - (l2 * self.W)
        self.B = self.B + self.Del_B - (l1 * np.sign(self.B)) - (l2 * self.B)

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

    def backward(self, D, lr, mtype, mu, l1, l2):
        self.Del_W = mu * self.Del_W + (-lr * (self.A).T).dot(D)
        self.Del_B = mu * self.Del_B + (-lr * sumToRow(D))

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    def ada_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9):
        self.G_W = self.G_W + ((((self.A).T).dot(D)) ** 2)
        self.G_B = self.G_B + (sumToRow(D) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D)
        self.Del_B = mu * self.Del_B - lr_B * sumToRow(D)

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    def rmsprop_backward(self, D, lr, mtype, mu, l1, l2, e = 1e-9, g = 0.9):
        self.G_W = g * self.G_W + (1 - g) * ((((self.A).T).dot(D)) ** 2)
        self.G_B = g * self.G_B + (1 - g) * (sumToRow(D) ** 2)

        lr_W = lr / np.sqrt(self.G_W + e)
        lr_B = lr / np.sqrt(self.G_B + e)

        self.Del_W = mu * self.Del_W - lr_W * ((self.A).T).dot(D)
        self.Del_B = mu * self.Del_B - lr_B * sumToRow(D)

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    def adam_backward(self, D, lr, mtype, mu, t, l1, l2, b = 0.9, d = 0.9, e = 1e-9):
        self.M_W = b * self.M_W + (1 - b) * ((self.A).T).dot(D)
        self.V_W = d * self.V_W + (1 - d) * ((((self.A).T).dot(D)) ** 2)

        M_W_hat = self.M_W / (1 + (b ** t))
        V_W_hat = self.V_W / (1 + (d ** t))

        Eta_W = lr / np.sqrt(V_W_hat + e)

        self.M_B = b * self.M_B + (1 - b) * (sumToRow(D))
        self.V_B = d * self.V_B + (1 - d) * (sumToRow(D) ** 2)

        M_B_hat = self.M_B / (1 + (b ** t))
        V_B_hat = self.V_B / (1 + (d ** t))

        Eta_B = lr / np.sqrt(V_B_hat + e)

        self.Del_W = mu * self.Del_W - Eta_W * M_W_hat
        self.Del_B = mu * self.Del_B - Eta_B * M_B_hat

        self.weight_update(l1, l2)

        return D.dot((self.W).T)

    def params(self):
        params = {}
        params['output']     = self.output
        params['W']          = self.W
        params['B']          = self.B
        params['indims']     = self.indims
        params['outdims']    = self.outdims

        return params
