import math
import pandas as pd
import numpy as np


def sigmoid(z):
    return 1/(1+math.e**(-z))


def loss(y, yhat):
    # y and yhat are np arrays
    m = len(y)
    return -(1/m)*np.sum(y*np.log(yhat)+(1-y)*(np.log(1-yhat)))


def gradients(X, y, yhat):
    m = X.shape[0]
    dw = (1/m)*(yhat-y)@X
    db = (1/m)*(yhat-y)
    return dw, db


def train(X, y, bs, epochs, lr):
    m, n = X.shape #
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(m, 1)
    # Normalizing the inputs.
    # x = normalize(X)
    losses = []
    for j in range(epochs):
        for i in range(m//bs):
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            y_hat = sigmoid(xb@w + b)
            dw, db = gradients(xb, yb, y_hat)
            w -= lr * dw
            b -= lr * db
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
    return w, b, losses


# def predict(X):




df = pd.read_csv('csvs/train.csv')
data = df.to_numpy()
print(len(data))
print(df)
print(data)