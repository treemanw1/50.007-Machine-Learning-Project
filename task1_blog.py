import pandas as pd
import numpy as np
from tqdm import tqdm


def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss


def gradients(X, y, y_hat):
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).

    # m-> number of training examples.
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1 / m) * np.dot(X.T, (y_hat - y))

    # Gradient of loss w.r.t bias.
    db = (1 / m) * np.sum((y_hat - y))

    return dw, db


def normalize(X):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def train(X, y, bs, epochs, lr):
    m, n = X.shape

    # Initializing weights and bias to zeros.
    w = np.zeros((n, 1))
    b = 0

    # Reshaping y.
    y = y.reshape(m, 1)

    # Normalizing the inputs.
    X = normalize(X)
    print('normalized')
    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in tqdm(range(epochs)):
        for i in range((m - 1) // bs + 1):
            # Defining batches. SGD.
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, w) + b)

            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters.
            w -= lr * dw
            b -= lr * db

        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)

    # returning weights, bias and losses(List).
    return w, b, losses


def predict(X):
    # X --> Input.

    # Normalizing the inputs.
    x = normalize(X)

    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)

    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]

    return np.array(pred_class)


def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy
