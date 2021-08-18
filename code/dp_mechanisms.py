import numpy as np
from scipy.stats import norm

class NoDP:
    name = "no_dp"
    def __init__(self):
        return

    def get_test_eval(self, test_y, pred_y):
        """
        @return test perf without any DP, return NLL
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        return -np.mean(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))

class BinaryThresholdDP:
    name = "binary_thres"
    def __init__(self, base_threshold):
        self.base_threshold = base_threshold

    def get_test_eval(self, test_y, pred_y):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nll = -np.mean(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        return int(test_nll < self.base_threshold)

class BonferroniThresholdDP(BinaryThresholdDP):
    name = "bonferroni_thres"
    def __init__(self, base_threshold, alpha):
        self.base_threshold = base_threshold
        self.alpha = alpha

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.correction_factor = np.power(2, num_adapt_queries)
        print(num_adapt_queries, self.correction_factor)

    def get_test_eval(self, test_y, pred_y):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.alpha/self.correction_factor)
        return int(upper_ci < self.base_threshold)

