import numpy as np

class NoDP:
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
    def __init__(self, base_threshold):
        self.base_threshold = base_threshold

    def get_test_eval(self, test_y, pred_y):
        """
        @return test perf without any DP, return NLL
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nll = -np.mean(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        return int(test_nll > self.base_threshold)

