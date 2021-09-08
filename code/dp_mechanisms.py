import logging

import numpy as np
from scipy.stats import norm, multivariate_normal
import scipy.optimize

def get_losses(test_y, pred_y):
    test_y = test_y.flatten()
    pred_y = np.maximum(np.minimum(1 - 1e-10, pred_y.flatten()), 1e-10)
    test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
    return test_nlls


class NoDP:
    name = "no_dp"
    def __init__(self):
        return

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf without any DP, return NLL
        """
        return get_losses(test_y, pred_y).mean()

class BinaryThresholdDP(NoDP):
    name = "binary_thres"
    def __init__(self, base_threshold, alpha):
        self.base_threshold = base_threshold
        self.alpha = alpha

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls = get_losses(test_y, pred_y)
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.alpha)
        print("upper ci", upper_ci)
        return int(upper_ci < self.base_threshold)

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs)/loss_diffs.size)
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(1 - self.alpha)
        print("upper ci", upper_ci)
        return int(upper_ci < 0)

class BonferroniThresholdDP(BinaryThresholdDP):
    name = "bonferroni_thres"
    def __init__(self, base_threshold, alpha):
        self.base_threshold = base_threshold
        self.alpha = alpha

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.correction_factor = np.power(2, num_adapt_queries)
        print(num_adapt_queries, self.correction_factor)

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)
        t_statistic = (np.mean(test_nlls) - self.base_threshold)/t_stat_se
        t_thres = norm.ppf(self.alpha/self.correction_factor)
        print("BONF t statistic", t_statistic, t_thres)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.alpha/self.correction_factor)
        print("BONF upper ci", upper_ci)
        return int(t_statistic < t_thres)

    def get_test_compare(self, test_y, pred_y, prev_pred_y):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs)/loss_diffs.size)
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(1 - self.alpha/self.correction_factor)
        print("BONF upper ci", upper_ci)
        return int(upper_ci < 0)


class BonferroniNonAdaptDP(BonferroniThresholdDP):
    """
    Assumes person is not adaptive at all
    """
    name = "bonferroni_nonadapt"
    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.correction_factor = num_adapt_queries
        print(num_adapt_queries, self.correction_factor)


class Node:
    def __init__(self, weight, success_edge, history, subfam_root=None, parent=None):
        """
        @param subfam_root: which node is the subfamily's root node. if none, this is the root
        """
        self.success = None
        self.success_edge = success_edge
        self.failure_edge = 1 -  success_edge
        self.failure = None
        self.weight = weight
        self.history = history
        self.subfam_root = subfam_root if subfam_root is not None else self
        self.parent = parent

    def observe_losses(self, test_losses):
        self.test_losses = test_losses

    def set_test_thres(self, thres):
        self.test_thres = thres

    def earn(self, weight_earn):
        self.weight += weight_earn
        self.local_alpha = None

class GraphicalBonfDP(BinaryThresholdDP):
    name = "graphical_bonf_thres"
    def __init__(self, base_threshold, alpha, success_weight, alpha_alloc_max_depth: int = 0):
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight
        self.alpha_alloc_max_depth = alpha_alloc_max_depth
        self.parallel_ratio = 0

    def _create_children(self, node):
        child_weight = (1 - self.parallel_ratio)/np.power(2, self.alpha_alloc_max_depth) if self.num_queries < self.alpha_alloc_max_depth else 0
        node.success = Node(child_weight, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
        node.failure = Node(child_weight, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)

    def set_num_queries(self, num_adapt_queries):
        # reset num queries
        self.num_queries = 0
        self.test_hist = []

        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(1/np.power(2, self.alpha_alloc_max_depth), success_edge=self.success_weight, history=[], subfam_root=None)
        self._create_children(self.test_tree)

    def _do_tree_update(self, test_result):
        # update tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            print("DO EARN")
            # remove node and propagate weights
            self.test_tree.success.earn(self.test_tree.weight * self.test_tree.success_edge)
            self.test_tree = self.test_tree.success
        else:
            print("failre", self.test_tree.failure.weight)
            self.test_tree.failure.earn(self.test_tree.weight * self.test_tree.failure_edge)
            print("failre new", self.test_tree.failure.weight)
            self.test_tree = self.test_tree.failure
        self.test_tree.local_alpha = self.alpha * self.test_tree.weight
        self._create_children(self.test_tree)

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls = get_losses(test_y, pred_y)
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        self.test_tree.local_alpha = self.alpha * self.test_tree.weight * self.test_tree.success_edge
        print("local alpha", self.test_tree.local_alpha)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.test_tree.local_alpha)
        print("upper ci", np.mean(test_nlls), upper_ci)
        test_result = int(upper_ci < self.base_threshold)

        self._do_tree_update(test_result)
        return test_result

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs)/loss_diffs.size)
        self.test_tree.local_alpha = self.test_tree.weight * self.alpha * self.test_tree.success_edge
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(1 - self.test_tree.local_alpha)
        test_result = int(upper_ci < 0)

        self._do_tree_update(test_result)
        return test_result

class GraphicalFFSDP(GraphicalBonfDP):
    name = "graphical_ffs"

    def set_num_queries(self, num_adapt_queries):
        # reset num queries
        self.num_queries = 0
        self.test_hist = []

        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(1, success_edge=self.success_weight, history=[])
        self._create_children(self.test_tree)

    def _get_prior_losses(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_losses] + self._get_prior_losses(node.failure, last_node)

    def _get_prior_thres(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_thres] + self._get_prior_thres(node.failure, last_node)

    def _solve_t_statistic_thres(self, est_cov, prior_thres, alpha_level):
        """
        @param est_cov: an estimated covariance matrix between the test statistics (not normalized, just differences of losses)
        @param prior_thres: the thresholds for all but the last test statistic
        @param alpha_level: the level of alpha we want to spend for the last test statistic

        @return the threshold for the last test statistic to satisfy the spending schedule
        """
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
            res_ffs = scipy.optimize.minimize_scalar(check_reject_prob_ffs, method="bounded", bounds=[-4, 0])
            assert res_ffs.success
            return res_ffs.x


    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        self.test_tree.observe_losses(test_nlls)

        # compute critical levels
        self.test_tree.local_alpha = self.alpha * self.test_tree.weight * self.test_tree.success_edge
        print("LOCAL ALPHA", self.test_tree.local_alpha)
        # Need to traverse subfam parent nodes to decide local level
        prior_test_nlls = self._get_prior_losses(self.test_tree.subfam_root, self.test_tree)
        prior_thres = self._get_prior_thres(self.test_tree.subfam_root, self.test_tree)
        est_cov = np.corrcoef(np.array(prior_test_nlls + [test_nlls]))
        t_thres = self._solve_t_statistic_thres(est_cov, prior_thres, self.test_tree.local_alpha)
        self.test_tree.set_test_thres(t_thres)

        #print("upper ci", np.mean(test_nlls), np.mean(test_nlls) + np.sqrt(np.var(test_nlls)/test_nlls.size) * norm.ppf(1 - alpha_level))
        std_err = np.sqrt(np.var(test_nlls)/test_nlls.size)
        t_statistic = (np.mean(test_nlls) - self.base_threshold)/std_err
        test_result = int(t_statistic < t_thres)
        print("test statistic", test_result, t_statistic, t_thres, self.base_threshold)

        # update tree
        self._do_tree_update(test_result)

        return test_result

class GraphicalParallelDP(GraphicalFFSDP):
    """
    Split alpha evenly across nodes generated by the "parallel" online procedure
    Model developer PRESPECIFies a parallel online procedure
    AND assumes correlation structure among models in a level
    """
    @property
    def name(self):
        return "graphical_par_%.1f" % self.loss_to_diff_std_ratio

    def __init__(self, base_threshold, alpha, success_weight, parallel_success_weight: float = 0.5, parallel_ratio: float = 0.9, loss_to_diff_std_ratio: float = 100.0, alpha_alloc_max_depth: int = 0):
        """
        @param loss_to_diff_std_ratio: the minimum ratio between the stdev of the loss of the predef model and the loss of the modifications in that level (maybe in the future, consider an avg?)
                                    bigger the ratio the more similar the adaptive strategy is to the prespecified strategy
        """
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight
        self.parallel_success_weight = parallel_success_weight
        self.parallel_ratio = parallel_ratio
        assert loss_to_diff_std_ratio >= 1
        self.loss_to_diff_std_ratio = loss_to_diff_std_ratio
        self.alpha_alloc_max_depth = alpha_alloc_max_depth

    def _create_children(self, node, next_par_node):
        child_weight = (1 - self.parallel_ratio)/np.power(2, self.alpha_alloc_max_depth) if self.num_queries < self.alpha_alloc_max_depth else 0
        node.success = Node(child_weight, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
        node.success.par_parent = next_par_node
        node.failure = Node(child_weight, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)
        node.failure.par_parent = next_par_node

    def set_num_queries(self, num_adapt_queries):
        # reset num queries
        self.num_queries = 0
        self.test_hist = []
        self.parallel_test_hist = []

        self.num_adapt_queries = num_adapt_queries

        # Create parallel sequence
        self.parallel_tree = Node(1/np.power(2, self.alpha_alloc_max_depth) * self.parallel_ratio, success_edge=self.parallel_success_weight, history=[], subfam_root=None)
        self.last_ffs_root = self.parallel_tree
        curr_par_node = self.parallel_tree
        for i in range(1, num_adapt_queries + 1):
            # TODO: this this history thing matter? It's just a placeholder
            weight = 1/num_adapt_queries * self.parallel_ratio if i < num_adapt_queries else 0
            next_par_node = Node(weight, success_edge=self.parallel_success_weight, history=[None] * i, subfam_root=self.parallel_tree, parent=curr_par_node)
            curr_par_node.par_child = next_par_node
            curr_par_node.par_weight  = self.parallel_success_weight
            curr_par_node.adapt_tree_node_weight  = (1 - self.parallel_success_weight)/np.power(2, i - 1)

            curr_par_node = next_par_node
            curr_par_node.par_child = None

        # Create adapt tree
        self.test_tree = Node(1 - self.parallel_ratio, success_edge=self.success_weight, history=[], subfam_root=None)
        self.test_tree.par_parent = self.parallel_tree
        self._create_children(self.test_tree, self.parallel_tree.par_child)
        self.num_same_level_nodes = 1
        self.same_level_node = Node(0, success_edge=self.success_weight, history=[], subfam_root=None)

    def _get_prior_par_losses(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_losses] + self._get_prior_par_losses(node.par_child, last_node)

    def _get_prior_par_thres(self, node, last_node):
        if node == last_node:
            return []
        return [node.test_thres] + self._get_prior_par_thres(node.par_child, last_node)

    def _get_test_eval_ffs(self, test_y, pred_y, predef_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return tuple(
            test perf where 1 means approve and 0 means not approved,
            test nlls)
        """
        test_nlls = get_losses(test_y, pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls)/predef_pred_test_nlls.size)
        self.parallel_tree.observe_losses(test_nlls)

        # Need to traverse subfam parent nodes to decide local level
        prior_test_nlls = self._get_prior_par_losses(self.last_ffs_root, self.parallel_tree)
        prior_thres = self._get_prior_par_thres(self.last_ffs_root, self.parallel_tree)
        est_corr = np.corrcoef(np.array(prior_test_nlls + [test_nlls]))
        print("LOCAL FFS par ALPHA", self.parallel_tree.local_alpha)
        if self.parallel_tree.subfam_root == self.parallel_tree:
            t_thres = norm.ppf(self.parallel_tree.local_alpha)
        else:
            #print("EST COV", np.corrcoef(np.array(prior_test_nlls + [test_nlls])))
            t_thres = self._solve_t_statistic_thres(est_corr, prior_thres, self.parallel_tree.local_alpha)
        self.parallel_tree.set_test_thres(t_thres)
        t_statistic = (np.mean(test_nlls) - self.base_threshold)/std_err
        #print("95 CI", np.mean(test_nlls) + std_err * 1.96)
        test_result = int(t_statistic < t_thres)
        print("t_statistics", t_statistic, t_thres)
        return test_result

    def _solve_t_statistic_thres_corr(self, prior_thres, alpha_level):
        """
        @param prior_thres: the thresholds for all but the last test statistic
        @param alpha_level: the level of alpha we want to spend for the last test statistic

        @return the threshold for the last test statistic to satisfy the spending schedule
        """
        new_thres = norm.ppf(alpha_level)/self.loss_to_diff_std_ratio + prior_thres
        return new_thres

    def _get_test_eval_corr(self, test_y, pred_y, predef_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return tuple(
            test perf where 1 means approve and 0 means not approved,
            test nlls)
        """
        test_nlls = get_losses(test_y, pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls)/test_nlls.size)
        self.test_tree.observe_losses(test_nlls)

        test_stat = (np.mean(test_nlls) - self.base_threshold)/std_err
        print("LOCAL ALPHA TREE", self.test_tree.local_alpha, norm.ppf(self.test_tree.local_alpha))
        t_thres = self._solve_t_statistic_thres_corr(self.test_tree.par_parent.test_thres, self.test_tree.local_alpha)
        print("T THRES adjust", norm.cdf(t_thres), t_thres)
        self.test_tree.set_test_thres(t_thres)
        test_result = int(test_stat < t_thres)
        print("corr test resl", test_stat, t_thres)
        return test_result

    def _get_test_compare_ffs(self, test_y, pred_y, baseline_pred_y, predef_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved,
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_old = get_losses(test_y, baseline_pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls - test_nlls_old)/test_nlls_old.size)
        test_nll_diffs = test_nlls_new - test_nlls_old
        self.parallel_tree.observe_losses(test_nll_diffs)

        # Need to traverse subfam parent nodes to decide local level
        prior_test_diffs = self._get_prior_par_losses(self.last_ffs_root, self.parallel_tree)
        prior_thres = self._get_prior_par_thres(self.last_ffs_root, self.parallel_tree)
        est_corr = np.corrcoef(np.array(prior_test_diffs + [test_nll_diffs]))
        t_thres = self._solve_t_statistic_thres(est_corr, prior_thres, self.parallel_tree.local_alpha)
        self.parallel_tree.set_test_thres(t_thres)
        t_statistic = (np.mean(test_nll_diffs))/std_err
        test_result = int(t_statistic < t_thres)
        print("t_statistics", t_statistic, t_thres)
        return test_result

    def _get_test_compare_corr(self, test_y, pred_y, baseline_pred_y, predef_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved,
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_old = get_losses(test_y, baseline_pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls - test_nlls_old)/test_nlls_old.size)
        test_nll_diffs = test_nlls_new - test_nlls_old
        self.test_tree.observe_losses(test_nll_diffs)

        test_stat = (np.mean(test_nll_diffs))/std_err
        t_thres = self._solve_t_statistic_thres_corr(self.test_tree.par_parent.test_thres, self.test_tree.local_alpha)
        self.test_tree.set_test_thres(t_thres)
        test_result = int(test_stat < t_thres)
        return test_result

    def _get_fwer(self, q, uniq_alpha_weights, weight_cts, desired_fwer, debug=False):
        def fwer_calc(c_thres):
            tot_alpha = 1 - norm.cdf(c_thres)
            offsets = norm.ppf(1 - uniq_alpha_weights * desired_fwer * q)
            assert np.all(offsets >= c_thres)
            extra_alphas = 1 - norm.cdf((offsets - c_thres) * self.loss_to_diff_std_ratio)
            tot_extra_alpha = np.sum(extra_alphas * weight_cts)
            #print("ALPHA SPLIT", tot_alpha, extra_alpha)
            return tot_alpha + tot_extra_alpha

        max_c_thres = norm.ppf(1 - uniq_alpha_weights.max() * q * desired_fwer)
        res = scipy.optimize.minimize_scalar(fwer_calc, bounds=[0, max_c_thres], method="bounded")
        #print("local alpha", res.fun, "desired fwer", desired_fwer)
        print("FWER...", res.fun, desired_fwer, (res.fun - desired_fwer)/desired_fwer, np.abs(res.fun - desired_fwer)/desired_fwer)
        #assert ((res.fun - desired_fwer)/desired_fwer < 1e-2) or (np.abs(res.fun - desired_fwer)/desired_fwer < 0.1)
        #print("INNER OPTIM", res.fun, res.x)
        return res.fun

    def _do_adapt_tree_update(self, test_result):
        # update adaptive tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            # remove node and propagate weights
            self.test_tree.success.earn(self.test_tree.weight * self.test_tree.success_edge)
            self.test_tree = self.test_tree.success
        else:
            self.test_tree.failure.earn(self.test_tree.weight * self.test_tree.failure_edge)
            self.test_tree = self.test_tree.failure

        self.num_same_level_nodes = np.power(2, len(self.test_tree.history))
        self._create_children(self.test_tree, self.parallel_tree.par_child)

    def _do_par_tree_update(self, test_result):
        # Update parallel tree
        self.parallel_test_hist.append(test_result)

        if test_result == 1:
            self.parallel_tree.par_child.earn(self.parallel_tree.weight * self.parallel_tree.par_weight)
            self.test_tree.earn(self.parallel_tree.weight * self.parallel_tree.adapt_tree_node_weight)
            # TODO: check if this FFS root is correct
            self.last_ffs_root = self.parallel_tree.par_child

        # Increment the par tree node regardless of success
        self.parallel_tree = self.parallel_tree.par_child
        self.parallel_tree.local_alpha = self.parallel_tree.weight * self.alpha

    def get_test_eval(self, test_y, pred_y, predef_pred_y):
        # Test the parallel tree stuff, do any weight propagation
        self.parallel_tree.local_alpha = self.parallel_tree.weight * self.alpha
        orig_alpha = self.parallel_tree.local_alpha
        print("ORIG_", orig_alpha)
        parallel_test_result = self._get_test_eval_ffs(test_y, predef_pred_y, predef_pred_y)
        self._do_par_tree_update(parallel_test_result)

        if self.test_tree.weight == 0:
            test_result = 0
        else:
            self.test_tree.local_alpha = self.test_tree.weight * self.alpha
            test_result = self._get_test_eval_corr(test_y, pred_y, predef_pred_y)

        # update tree
        self._do_adapt_tree_update(test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y):
        # Test the parallel tree stuff, do any weight propagation
        self.parallel_tree.local_alpha = self.parallel_tree.weight * self.alpha
        orig_alpha = self.parallel_tree.local_alpha
        parallel_test_result = self._get_test_compare_ffs(test_y, predef_pred_y, prev_pred_y, predef_pred_y, self.parallel_tree)
        self._do_par_tree_update(parallel_test_result)

        if self.test_tree.weight == 0:
            test_result = 0
        else:
            self.test_tree.local_alpha = self.test_tree.weight * self.alpha
            test_result = self._get_test_compare_corr(test_y, pred_y, predef_pred_y, self.test_tree)

        # update tree
        self._do_adapt_tree_update(test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result
