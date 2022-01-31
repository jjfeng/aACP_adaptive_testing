"""
ML algorithms considered by the model developer
"""
import numpy as np

from sklearn.linear_model import LogisticRegression

class MyLogisticRegression(LogisticRegression):
    def get_decision(self, X):
        return np.ones(X.shape[0])

class SelectiveLogisticRegression(LogisticRegression):
    def __init__(self, penalty: str = "none", target_acc: float=1, subset_frac: float = 0.7):
        self.target_acc = target_acc
        self.subset_frac = subset_frac
        super().__init__(penalty=penalty)

    def fit(self, X, y):
        self.prob_thres = 0
        ntrain = int(self.subset_frac * X.shape[0])
        super().fit(X[:ntrain], y[:ntrain])

        X_calib = X[ntrain:]
        y_calib = y[ntrain:]
        fitted_class = self.predict(X_calib)
        fitted_probs = self.predict_proba(X_calib)[:,1]

        acc = fitted_class == y_calib
        for thres in np.arange(0, 0.5, step=0.01):
            keep_mask = np.abs(fitted_probs - 0.5) > thres
            subset_acc = np.mean(acc[keep_mask])
            #print("searching...", subset_acc, "THRES", thres, self.target_acc, keep_mask.mean())
            if subset_acc > self.target_acc:
                print("SUCCESS", subset_acc, "THRES", thres, self.target_acc, keep_mask.mean())
                self.prob_thres = thres
                return
        print("WARNING: no threshold found")
        self.prob_thres = 0.5

    def predict(self, X):
        pred_class = super().predict(X)
        pred_probs = self.predict_proba(X)[:,1]
        keep_mask = np.abs(pred_probs - 0.5) > self.prob_thres
        pred_class[np.logical_not(keep_mask)] = -100
        return pred_class

    def get_decision(self, X):
        pred_probs = self.predict_proba(X)[:,1]
        return np.abs(pred_probs - 0.5) > self.prob_thres

