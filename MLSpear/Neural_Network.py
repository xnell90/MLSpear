from MLSpear.MLF import *
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:
    #Parameters:
    # layers      = an array of activation layers.
    # scale_WB    = boolean that determines whether you scale weights and biases
    # print_error = boolean that determines whether you print the error curve.

    #Returns:
    # None, but initialized, weights and biases are automatically initiazlized
    # with the option of scaling.
    def __init__(self, layers, scale_WB = True, print_error = True):
        self.layers      = layers
        self.num_layers  = len(self.layers)
        self.scale_WB    = scale_WB
        self.print_error = print_error

        if self.layers[-1].output == softmax:
            self.error_function = cost_entropy
            self.error_metric   = "Cost Entropy"
        else:
            self.error_function = sum_squared_error
            self.error_metric   = "Sum Squared Error"

        self.initialize_weights()

    # Initializes weights and biases.
    def initialize_weights(self):
        for layer in self.layers:
            layer.weight_initialize(self.scale_WB)

    #Parameters:
    # X     = numpy matrix where each column represents a predictor variable.
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu    = momentum value

    #Returns:
    # a numpy matrix used to forward propagate.
    def dropout_predict(self, X, mtype = 'conventional', mu = 0):
        P = X
        for layer in self.layers:
            if type(layer).__name__ == "Batch_Normalization":
                P = layer.train_forward(P, mtype, mu)
            elif layer.p < 1 and layer.p > 0:
                P = layer.dropout_forward(P, mtype, mu)
            else:
                P = layer.forward(P, mtype, mu)

        return P

    #Parameters:
    # X     = numpy matrix where each column represents a predictor variable.
    # mtype = momentum type used to optimize the gradient descent algorithm.
    #         ('conventional' or 'nesterov').
    # mu    = momentum value

    #Returns:
    # a numpy matrix where each value represents a prediction
    def predict(self, X, mtype = 'conventional', mu = 0):
        P = X
        for layer in self.layers:
            P = layer.forward(P, mtype, mu)

        return P

    #Parameters:
    # errors = an array of non_negative real numbers where each number
    #          represents error.
    # error_metric = name of the error metric. (string)

    #Returns:
    # None, but displays the error graph.
    def display_errors(self, errors, error_metric):
        x = [i + 1 for i in range(len(errors))]
        plt.plot(x, errors, color = 'blue')
        plt.xlabel("Epochs")
        plt.ylabel(error_metric)
        plt.show()

    #Parameters:
    # X          = numpy matrix where each column represents a predictor variable.
    # Y          = numpy column vector that corresponds to a response variable.
    # cycles     = the number of times the training algorithm goes over the
    #              entire dataset (also known as epochs) (non-negative integer)
    # lr         = learning rate
    # batch_size = the number of rows that constitute a batch, normally used for
    #              batch gradient descent
    # mtype      = momentum type used to optimize the gradient descent algorithm.
    #              ('conventional' or 'nesterov').
    # mu         = momentum value
    # l1         = represents the l1 regularization parameter. (Lasso Regression)
    #              (non-negative real number)
    # l2         = represents the l2 regularization parameter. (Ridge Regression)
    #              (non-negative real number)
    # optimzer   = graident descent optimizer.
    #              (takes only 'vanilla', 'rmsprop', 'adam', 'ada')

    #Returns:
    # None but prints out the error if self.print_error is set to True

    # Remark: To train the model stochastically, set batch_size to 1. For the
    # full gradient descent, set batch size to X.shape[0]
    def train(self, X, Y, cycles, lr, batch_size = 20, mtype = 'conventional', mu = 0, l1 = 0, l2 = 0, optimizer = 'vanilla'):
        self.initialize_weights()

        Data = np.hstack((X, Y))
        np.random.shuffle(Data)

        X_ = Data[:, 0:X.shape[1]]
        Y_ = Data[:, X.shape[1]:]

        errors = []
        for i in range(cycles):
            for j in range(X_.shape[0]):
                if j + batch_size > X_.shape[0]:
                    break

                X_batch = X_[j:(j + batch_size), :]
                P_batch = self.dropout_predict(X_batch, mtype, mu)
                last_index = self.num_layers - 1

                Y_batch = Y_[j:(j + batch_size), :]
                error = self.error_function(Y_batch, P_batch)
                errors.append(error)

                D = P_batch - Y_batch
                for i in range(last_index, -1, -1):
                    if optimizer == 'vanilla':
                        D = self.layers[i].backward(D, lr, mtype, mu, l1, l2)
                    elif optimizer == 'rmsprop':
                        D = self.layers[i].rmsprop_backward(D, lr, mtype, mu, l1, l2)
                    elif optimizer == 'adam':
                        D = self.layers[i].adam_backward(D, lr, mtype, mu, i + 1, l1, l2)
                    elif optimizer == 'ada':
                        D = self.layers[i].ada_backward(D, lr, mtype, mu, l1, l2)
                    else:
                        print("Error: optimizer " + optimizer + " does not exists!")
                        return

        if self.print_error:
            self.display_errors(errors, self.error_metric)

    #Returns:
    # a python dictionary that contains all parameters in this layer.
    def params(self):
        params = {}
        params['layers']               = self.layers
        params['num_layers']           = self.num_layers
        params['scale_WB']             = self.scale_WB
        params['print_error']          = self.print_error

        return params
