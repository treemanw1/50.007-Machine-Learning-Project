# Task 1b
from tqdm import tqdm
import task1 as lr
import pandas as pd
import numpy as np

X = pd.read_csv("csvs/train_tfidf_features.csv")
X = X.drop(columns=['id', 'label'])
# remove zero std features from X
print('X shape:', X.shape)
zero_std_cols = list(X.loc[:, X.std() == 0].columns)
print('zero std cols:', zero_std_cols)
X = X.loc[:, X.std() != 0]
print('X shape:', X.shape)
X = X.to_numpy()

y = pd.read_csv("csvs/train.csv")
y = y['label']
y = y.to_numpy()

w, b, losses = lr.train(X, y, bs=100, epochs=1000, lr=0.01)
print("w: ", w)
print("b: ", b)
# was NaN before, seems to work now
w_df = pd.DataFrame(w)
losses_df = pd.DataFrame(losses)
w_df.to_csv('csvs/submission/w.csv', index=False)
losses_df.to_csv('csvs/submission/losses.csv')
print('weights saved')
