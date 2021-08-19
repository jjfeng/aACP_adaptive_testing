import numpy as np
from scipy.stats import norm, multivariate_normal
import scipy.optimize

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


class Node:
    def __init__(self, alpha, subfam_root=None):
        """
        @param subfam_root: which node is the subfamily's root node. if none, this is the root
        """
        self.success = None
        self.failure = None
        self.alpha = alpha
        self.subfam_root = subfam_root if subfam_root is not None else self

    def observe_losses(self, test_losses):
        self.test_losses = test_losses

    def set_test_thres(self, thres):
        self.test_thres = thres

    def earn(self, alpha_earn):
        self.alpha += alpha_earn

class GraphicalBonfDP(BinaryThresholdDP):
    name = "graphical_bonf_thres"
    def __init__(self, base_threshold, alpha, success_weight):
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight

    def _create_tree(self, node, tree_depth):
        if tree_depth == 0:
            return

        node.success = Node(0, subfam_root=None)
        node.success_edge = self.success_weight
        node.failure = Node(0, subfam_root=node.subfam_root)
        node.failure_edge = 1 - self.success_weight
        node.failure.parent = node
        self._create_tree(node.success, tree_depth - 1)
        self._create_tree(node.failure, tree_depth - 1)

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(self.alpha, subfam_root=None)
        self._create_tree(self.test_tree, num_adapt_queries)

        # reset num queries
        self.num_queries = 0
        self.test_hist = []

    def get_test_eval(self, test_y, pred_y):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        alpha_level = self.test_tree.alpha
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - alpha_level)
        #print("upper ci", np.mean(test_nlls), upper_ci)
        test_result = int(upper_ci < self.base_threshold)

        # update tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            # remove node and propagate weights
            self.test_tree.success.earn(self.test_tree.alpha * self.test_tree.success_edge)
            self.test_tree = self.test_tree.success
        else:
            self.test_tree.failure.earn(self.test_tree.alpha * self.test_tree.failure_edge)
            self.test_tree = self.test_tree.failure

        return test_result

class GraphicalFFSDP(GraphicalBonfDP):
    name = "graphical_ffs"

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(self.alpha)
        self._create_tree(self.test_tree, num_adapt_queries)

        # reset num queries
        self.num_queries = 0
        self.test_hist = []

    def _get_prior_losses(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_losses] + self._get_prior_losses(node.failure, last_node)

    def _get_prior_thres(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_thres] + self._get_prior_thres(node.failure, last_node)

    def _solve_t_statistic_thres(self, est_cov, prior_thres, alpha_level):
        mvn = multivariate_normal(cov=est_cov)
        num_prior = len(prior_thres)
        def check_reject_prob_marg(thres):
            reject_prob = mvn.cdf([np.inf] * num_prior + [thres])
            return np.abs(reject_prob - alpha_level)
        def check_reject_prob_ffs(thres):
            reject_prob = mvn.cdf([np.inf] * num_prior + [thres]) - mvn.cdf(prior_thres + [thres])
            return np.abs(reject_prob - alpha_level)

        if num_prior == 0:
            res_marg = scipy.optimize.minimize_scalar(check_reject_prob_marg)
            assert res_marg.success
            return res_marg.x
        else:
            res_ffs = scipy.optimize.minimize_scalar(check_reject_prob_ffs)
            assert res_ffs.success
            return res_ffs.x


    def get_test_eval(self, test_y, pred_y):
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        self.test_tree.observe_losses(test_nlls)
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        # compute critical levels
        alpha_level = self.test_tree.alpha
        # Need to traverse subfam parent nodes to decide local level
        prior_test_nlls = self._get_prior_losses(self.test_tree.subfam_root, self.test_tree)
        prior_thres = self._get_prior_thres(self.test_tree.subfam_root, self.test_tree)
        est_cov = np.cov(np.array(prior_test_nlls + [test_nlls]))
        t_thres = self._solve_t_statistic_thres(est_cov, prior_thres, alpha_level)
        self.test_tree.set_test_thres(t_thres)

        #print("upper ci", np.mean(test_nlls), np.mean(test_nlls) + np.sqrt(np.var(test_nlls)/test_nlls.size) * norm.ppf(1 - alpha_level))
        t_statistic = np.sqrt(test_nlls.size) * (np.mean(test_nlls) - self.base_threshold)
        test_result = int(t_statistic < t_thres)
        #print("test statistic", test_result, t_statistic, t_thres, self.base_threshold)

        # update tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            # remove node and propagate weights
            self.test_tree.success.earn(self.test_tree.alpha * self.test_tree.success_edge)
            self.test_tree = self.test_tree.success
        else:
            self.test_tree.failure.earn(self.test_tree.alpha * self.test_tree.failure_edge)
            self.test_tree = self.test_tree.failure

        return test_result

