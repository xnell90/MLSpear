import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Pre-processing Functions
def min_max(X):
    mins = list(X.min(axis = 0))
    maxs = list(X.max(axis = 0))

    return (mins, maxs)

def mean_std(X):
    means = list(X.mean(axis = 0))
    stds  = list(X.std(axis = 0))

    return(means, stds)

def min_max_scaling(X, mins, maxs):
    for i in range(0, X.shape[1]):
        minimum = mins[i]
        maximum = maxs[i]

        spread = maximum - minimum
        X[:, i] =  (X[:, i] - minimum) / spread

def z_score_scaling(X, means, stds):
    for i in range(0, X.shape[1]):
        mean = means[i]
        std  = stds[i]

        X[:, i] =  (X[:, i] - mean) / std

def one_hot_encode(X):
    if type(X) == np.ndarray: category_labels = list(set(X.flatten()))
    else: category_labels = list(set(X))

    row_num = len(X)
    col_num = len(category_labels)

    ohe = np.zeros((row_num, col_num))
    for i in range(row_num):
        ohe[i, category_labels.index(X[i])] = 1

    return ohe

def train_validate_test(X):
    i = (X.shape[0] * 7) // 10
    j = i + ((X.shape[0] * 15) // 100)

    X_train = X[0:i, :]
    X_valid = X[i:j, :]
    X_test  = X[j:, :]

    return (X_train, X_valid, X_test)

#  Softmax and Sigmoid combined
def softmax(X):
    if X.shape[1] == 1:
        return np.where(X >= 0, 1 / (1 + np.exp(- X)), np.exp(X) / (1 + np.exp(X)))
    else:
        P = np.exp(X)
        col_sum = np.apply_along_axis(np.sum, 1, P)
        col_sum = col_sum.reshape((col_sum.shape[0], 1))
        P = P / col_sum

        return P

# Error Metrics
def accuracy(P, Y):
    diff = np.abs(__round(P) - Y)
    return 1 - np.sum(diff) / (Y.shape[0] * Y.shape[1])

def recall(P, Y):
    return np.sum(__round(P) * Y) / np.sum(Y)

# $$$$
def __round(P):
    if P.shape[1] == 1: return np.round(P)
    else:
        result  = np.zeros(P.shape)
        max_col = P.argmax(axis = 1)

        for row in range(0, result.shape[0]):
            result[row, max_col[row]] = 1

        return result

def sum_squared_error(Y, P):
    return 0.5 * np.sum((Y - P) ** 2)

def cost_entropy(Y, P):
    if P.shape[1] == 1 and Y.shape[1] == 1:
        column = np.log(P ** Y) + np.log((1 - P) ** (1 - Y))
        result = - np.sum(column)
    else:
        result = - np.sum(np.log(P ** Y))

    return result

# ROC plots the ROC curve and prints out the AUC
def roc(P, Y):
    x_fpr = []
    y_tpr = []

    p_thresholds = [1 - i/999 for i in range(0, 1000)]

    for p_threshold in p_thresholds:
        PT = __round_by(p_threshold, P)
        (fpr, tpr) = __tpr_fpr(PT, Y)
        x_fpr.append(fpr)
        y_tpr.append(tpr)

    plt.plot(x_fpr, y_tpr)
    plt.title("ROC Curve From the Model", fontweight = 'bold')
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    auc = 0
    for i in range(0, 999):
        auc += 0.5 * (y_tpr[i + 1] + y_tpr[i]) * (x_fpr[i + 1] - x_fpr[i])

    print("AUC for the ROC Curve: " + str(auc))
    plt.show()

# Hidden functions
def __round_by(p_threshold, P):
    return [int(P[i] >= p_threshold) for i in range(0, len(P))]

def __tpr_fpr(PT, Y):
    (tp, fn) = (0, 0)
    (fp, tn) = (0, 0)

    for i in range(0, len(PT)):
        if PT[i] == Y[i]:
            if PT[i] == 1: tp += 1
            else: tn += 1
        else:
            if PT[i] == 1: fp += 1
            else: fn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    return (fpr, tpr)

# ---------------------------------------------------------------------------
# Other Functions

def row_sum(M):
    return np.sum(M, axis = 0).reshape(1, M.shape[1])
