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

