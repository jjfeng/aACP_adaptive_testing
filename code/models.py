"""
ML algorithms considered by the model developer
"""
import numpy as np

from sklearn.linear_model import LogisticRegression

class MyLogisticRegression(LogisticRegression):
    def __init__(self, penalty: str = "none"):
        super().__init__(penalty=penalty)

    def get_decision(self, X):
        return np.ones(X.shape[0])

class RevisedLogisticRegression(LogisticRegression):
    def __init__(self, penalty: str = "none", subset_frac: float = 0.7, spec_buffer: float = 0.05):
        self.subset_frac = subset_frac
        self.spec_buffer = spec_buffer
        super().__init__(penalty=penalty)

    def fit_orig_mdl(self, X, y):
        self.orig_mdl = MyLogisticRegression(penalty=self.penalty)
        ntrain = int(self.subset_frac * X.shape[0])
        self.orig_mdl.fit(X[:ntrain], y[:ntrain])
        print("SDJFKLSJFKLSJDKLFDSF")
        print(self.orig_mdl)
        return self.orig_mdl

    def fit(self, X, y, orig_mdl):
        print("SUP", orig_mdl)
        ntrain = int(self.subset_frac * X.shape[0])
        super().fit(X[:ntrain], y[:ntrain])

        X_calib = X[ntrain:]
        y_calib = y[ntrain:]
        print("NTRAIN", ntrain, X_calib.shape[0])
        fitted_probs = self.predict_proba(X_calib)[:,1]
        orig_pred_class = orig_mdl.predict(X_calib)

        for thres in np.arange(0, 1, step=0.01):
            fitted_class = (fitted_probs > thres).astype(int)
            acc_diff = (fitted_class == y_calib).astype(int) - (orig_pred_class == y_calib).astype(int)
            spec_diff_est = np.mean(acc_diff[y_calib == 0])
            if spec_diff_est > self.spec_buffer:
                print("SUCCESS", spec_diff_est, "THRES", thres)
                self.prob_thres = thres
                return
        print("WARNING: no threshold found. using default")
        self.prob_thres = 0.5

    def predict(self, X):
        pred_probs = self.predict_proba(X)[:,1]
        return (pred_probs > self.prob_thres).astype(int)

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

