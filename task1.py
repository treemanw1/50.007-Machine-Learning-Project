import math
import pandas as pd
import numpy as np
from tqdm import tqdm


def sigmoid(z):
    return (1/(1+math.e**(-z)))


def loss(y, yhat):
    # y and yhat are np arrays
    m = len(y)
    return -(1/m)*np.sum(y*np.log(yhat)+(1-y)*(np.log(1-yhat)))


def gradients(X, y, yhat):
    m = X.shape[0]
    dw = (1/m)*(yhat-y).T@X
    db = (1/m)*np.sum(yhat-y)
    return dw, db  # (batch size)


def normalize(X):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def min_max_scale(X):
    return (X - np.amin(X))/(np.amax(X) - np.amin(X))


def train(X, y, bs, epochs, lr):
    m, n = X.shape
    w = np.zeros((n, 1))
    print('w shape:', w.shape)
    b = 0
    y = y.reshape(m, 1)
    # X = normalize(X)
    X = min_max_scale(X)
    losses = []
    for j in tqdm(range(epochs)):
        for i in range(m//bs):
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            # print('b:', b)
            # print('xbw:', xb@w)
            y_hat = sigmoid(xb@w + b)
            dw, db = gradients(xb, yb, y_hat)
            w -= lr * dw.T
            b -= lr * db.T
            # print('w avg:', np.mean(w))
            # print('b:', b)
        l = loss(y, sigmoid(np.dot(X, w) + b))  # y, yhat
        losses.append(l)
    return w, b, losses


def predict(X, w, b):
    # X = normalize(X)
    X = min_max_scale(X)
    preds = sigmoid(np.dot(X, w) + b)
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    return np.array(pred_class)
