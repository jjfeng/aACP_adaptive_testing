import numpy as np

from sklearn.linear_model import LogisticRegression

class LogisticRegressionMod(LogisticRegression):
    def fit(self, X, y):
        all_idxs = np.arange(y.shape[0])
        selected_idxs = np.isin(all_idxs % 6, [0,3,5])
        super().fit(X[selected_idxs], y[selected_idxs])

class LogisticRegressionOdd(LogisticRegression):
    def fit(self, X, y):
        all_idxs = np.arange(y.shape[0])
        odd_idxs = all_idxs % 2 == 1
        super().fit(X[odd_idxs], y[odd_idxs])

class LogisticRegressionEven(LogisticRegression):
    def fit(self, X, y):
        all_idxs = np.arange(y.shape[0])
        even_idxs = all_idxs % 2 == 0
        super().fit(X[even_idxs], y[even_idxs])

