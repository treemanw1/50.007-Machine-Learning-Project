import math
import numpy as np
import pandas as pd


def normalize(X):
    # return X/np.linalg.norm(X)
    X = X.astype(float)
    m, n = X.shape
    for i in range(n):
        X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])
    return X


def normalize_blog(X):
    # return X/np.linalg.norm(X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


# y = pd.read_csv("csvs/train.csv")
# print(y['label'])
# X = pd.read_csv("csvs/train_tfidf_features.csv")
# X = X.drop(columns=['id', 'label'])
# print(X.head())
# X_test = pd.read_csv("csvs/test_tfidf_features.csv")
# print(X_test.shape)

# ls = [[1,2,3], [4,5,6], [5,341,2], [3,2,1]]
# X = np.array(ls)
# print('BEFORE:', X)
# X_me = normalize(X)
# X_blog = normalize_blog(X)
# print("me:", X_me)
# print("blog:", X_blog)
