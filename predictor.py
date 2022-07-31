# Task 1b
from tqdm import tqdm
import task1 as lr
import pandas as pd
import numpy as np

X_test = pd.read_csv('csvs/test_tfidf_features.csv')
w = pd.read_csv('csvs/submission/w.csv')
b = -0.6905338153010798

test_ids = X_test['id']
print(test_ids)
X_test = X_test.drop(columns=['id'])
# # remove zero std features from X_test
# print('X shape:', X_test.shape)
# zero_std_cols = list(X_test.loc[:, X_test.std() == 0].columns)
# print('zero std cols:', len(zero_std_cols))
# X = X_test.loc[:, X_test.std() != 0]
# print('X shape:', X_test.shape)
# exit()

X_test = X_test.drop(['1134', '2468'], axis=1)
X_test = X_test.to_numpy()
preds = lr.predict(X_test, w, b)
preds = pd.DataFrame(preds, index=test_ids)
print('predictions:', preds)
preds.to_csv("csvs/submission/preds.csv")