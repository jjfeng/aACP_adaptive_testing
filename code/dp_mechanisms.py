import logging
from typing import List
import subprocess

import numpy as np
from scipy.stats import norm, multivariate_normal
import scipy.optimize

from constants import RSCRIPT_PATH

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
        t_stat_se = np.sqrt(np.var(test_nlls) / test_nlls.size)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.alpha)
        print("upper ci", upper_ci, test_nlls.mean())
        return int(upper_ci < self.base_threshold)

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs) / loss_diffs.size)
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(1 - self.alpha)
        print("upper ci", upper_ci, loss_diffs.mean(), "THRES")
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
        t_stat_se = np.sqrt(np.var(test_nlls) / test_nlls.size)
        t_statistic = (np.mean(test_nlls) - self.base_threshold) / t_stat_se
        t_thres = norm.ppf(self.alpha / self.correction_factor)
        print("BONF t statistic", t_statistic, t_thres)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(
            1 - self.alpha / self.correction_factor
        )
        print("BONF upper ci", upper_ci)
        return int(t_statistic < t_thres)

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs) / loss_diffs.size)
        t_thres = norm.ppf(self.alpha / self.correction_factor)
        t_statistic = np.mean(loss_diffs) / t_stat_se
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(
            1 - self.alpha / self.correction_factor
        )
        print("BONF t statistic", t_statistic, t_thres)
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
        self.failure_edge = 1 - success_edge
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

    def __init__(
        self,
        base_threshold,
        alpha,
        success_weight,
        alpha_alloc_max_depth: int = 0,
        scratch_file: str = None,
    ):
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight
        self.alpha_alloc_max_depth = alpha_alloc_max_depth
        self.parallel_ratio = 0
        self.scratch_file = scratch_file

    def _create_children(self, node):
        child_weight = (
            (1 - self.parallel_ratio) / np.power(2, self.alpha_alloc_max_depth)
            if self.num_queries < self.alpha_alloc_max_depth
            else 0
        )
        node.success = Node(
            child_weight,
            success_edge=self.success_weight,
            history=node.history + [1],
            subfam_root=None,
            parent=node,
        )
        node.failure = Node(
            child_weight,
            success_edge=self.success_weight,
            history=node.history + [0],
            subfam_root=node.subfam_root,
            parent=node,
        )

    def set_num_queries(self, num_adapt_queries):
        # reset num queries
        self.num_queries = 0
        self.test_hist = []

        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(
            1 / np.power(2, self.alpha_alloc_max_depth),
            success_edge=self.success_weight,
            history=[],
            subfam_root=None,
        )
        self._create_children(self.test_tree)

    def _do_tree_update(self, test_result):
        # update tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            print("DO EARN")
            # remove node and propagate weights
            self.test_tree.success.earn(
                self.test_tree.weight * self.test_tree.success_edge
            )
            self.test_tree = self.test_tree.success
        else:
            print("failre", self.test_tree.failure.weight)
            self.test_tree.failure.earn(
                self.test_tree.weight * self.test_tree.failure_edge
            )
            print("failre new", self.test_tree.failure.weight)
            self.test_tree = self.test_tree.failure
        self.test_tree.local_alpha = self.alpha * self.test_tree.weight
        self._create_children(self.test_tree)

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls = get_losses(test_y, pred_y)
        t_stat_se = np.sqrt(np.var(test_nlls) / test_nlls.size)

        self.test_tree.local_alpha = (
            self.alpha * self.test_tree.weight * self.test_tree.success_edge
        )
        print("local alpha", self.test_tree.local_alpha)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(
            1 - self.test_tree.local_alpha
        )
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
        t_stat_se = np.sqrt(np.var(loss_diffs) / loss_diffs.size)
        self.test_tree.local_alpha = (
            self.test_tree.weight * self.alpha * self.test_tree.success_edge
        )
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(
            1 - self.test_tree.local_alpha
        )
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
        if len(prior_thres) == 0:
            thres = scipy.stats.norm.ppf(alpha_level)
            print("THRES", thres, scipy.stats.norm.cdf(thres), alpha_level)
            return thres
        else:
            np.savetxt(self.scratch_file, est_cov, delimiter=",")
            cmd = [
                "Rscript",
                RSCRIPT_PATH,
                self.scratch_file,
                str(alpha_level),
            ] + list(map(str, prior_thres))
            print(" ".join(cmd))
            res = subprocess.run(cmd, stdout=subprocess.PIPE)
            thres = float(res.stdout.decode("utf-8")[4:])
            print("THRES FROM R", thres)
            return thres

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        self.test_tree.observe_losses(test_nlls)

        # compute critical levels
        self.test_tree.local_alpha = (
            self.alpha * self.test_tree.weight * self.test_tree.success_edge
        )
        print("LOCAL ALPHA", self.test_tree.local_alpha)
        # Need to traverse subfam parent nodes to decide local level
        prior_test_nlls = self._get_prior_losses(
            self.test_tree.subfam_root, self.test_tree
        )
        prior_thres = self._get_prior_thres(self.test_tree.subfam_root, self.test_tree)
        est_cov = (
            np.corrcoef(np.array(prior_test_nlls + [test_nlls]))
            if len(prior_test_nlls)
            else np.array([[1]])
        )
        t_thres = self._solve_t_statistic_thres(
            est_cov, prior_thres, self.test_tree.local_alpha
        )
        self.test_tree.set_test_thres(t_thres)

        # print("upper ci", np.mean(test_nlls), np.mean(test_nlls) + np.sqrt(np.var(test_nlls)/test_nlls.size) * norm.ppf(1 - alpha_level))
        std_err = np.sqrt(np.var(test_nlls) / test_nlls.size)
        t_statistic = (np.mean(test_nlls) - self.base_threshold) / std_err
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
        return "graphical_par"

    def __init__(
        self,
        base_threshold,
        alpha,
        success_weight,
        parallel_ratio: float = 0.9,
        first_pres_weight: float = 0.5,
        alpha_alloc_max_depth: int = 0,
        scratch_file: str = None,
    ):
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight
        self.parallel_ratio = parallel_ratio
        self.first_pres_weight = first_pres_weight
        self.alpha_alloc_max_depth = alpha_alloc_max_depth
        self.scratch_file = scratch_file

    def _create_children(self, node):
        child_weight = (
            (1 - self.parallel_ratio) / np.power(2, self.alpha_alloc_max_depth)
            if self.num_queries < self.alpha_alloc_max_depth
            else 0
        )
        node.success = Node(
            child_weight,
            success_edge=self.success_weight,
            history=node.history + [1],
            subfam_root=None,
            parent=node,
        )
        node.failure = Node(
            child_weight,
            success_edge=self.success_weight,
            history=node.history + [0],
            subfam_root=node.subfam_root,
            parent=node,
        )

    def _get_prior_losses(self, node):
        if node is None:
            return []
        return self._get_prior_losses(node.parent) + [node.test_losses]

    def _get_prior_thres(self, node):
        if node is None:
            return []
        return self._get_prior_thres(node.parent) + [node.test_thres]

    def set_num_queries(self, num_adapt_queries):
        # reset num queries
        self.num_queries = 0
        self.test_hist = []
        self.parallel_test_hist = []

        self.num_adapt_queries = num_adapt_queries

        # Create parallel sequence
        self.parallel_tree = Node(
            self.first_pres_weight * self.parallel_ratio,
            success_edge=1,
            history=[],
            subfam_root=None,
        )
        self.parallel_tree.local_alpha = self.parallel_tree.weight * self.alpha
        self.last_ffs_root = self.parallel_tree
        curr_par_node = self.parallel_tree
        for i in range(1, num_adapt_queries + 1):
            weight = (
                (1 - self.first_pres_weight)/num_adapt_queries * self.parallel_ratio
                if i < num_adapt_queries
                else 0
            )
            next_par_node = Node(
                weight,
                success_edge=1,
                history=[None] * i,
                subfam_root=self.parallel_tree,
                parent=curr_par_node,
            )
            curr_par_node.failure = next_par_node
            curr_par_node.success = next_par_node
            curr_par_node = next_par_node
            curr_par_node.failure = None

        # Create adapt tree
        self.test_tree = Node(
            1 - self.parallel_ratio,
            success_edge=self.success_weight,
            history=[],
            subfam_root=None,
        )
        self.test_tree.local_alpha = self.test_tree.weight * self.alpha
        self._create_children(self.test_tree)

    def _get_test_eval_ffs(self, test_y, predef_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return tuple(
            test perf where 1 means approve and 0 means not approved,
            test nlls)
        """
        test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(test_nlls) / test_nlls.size)
        self.parallel_tree.observe_losses(test_nlls)

        # Need to traverse subfam parent nodes to decide local level
        prior_test_nlls = self._get_prior_losses(self.parallel_tree.parent)
        prior_thres = self._get_prior_thres(self.parallel_tree.parent)
        est_corr = (
            np.corrcoef(np.array(prior_test_nlls + [test_nlls]))
            if len(prior_test_nlls)
            else np.array([[1]])
        )
        print("LOCAL FFS par ALPHA", self.parallel_tree.local_alpha)
        t_thres = self._solve_t_statistic_thres(
            est_corr, prior_thres, self.parallel_tree.local_alpha
        )
        self.parallel_tree.set_test_thres(t_thres)
        t_statistic = (np.mean(test_nlls) - self.base_threshold) / std_err
        # print("95 CI", np.mean(test_nlls) + std_err * 1.96)
        test_result = int(t_statistic < t_thres)
        print("t_statistics", t_statistic, t_thres)
        return test_result

    def _get_test_eval_corr(self, test_y, pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return tuple(
            test perf where 1 means approve and 0 means not approved,
            test nlls)
        """
        test_nlls = get_losses(test_y, pred_y)
        std_err = np.sqrt(np.var(test_nlls) / test_nlls.size)
        self.test_tree.observe_losses(test_nlls)

        prior_test_nlls = self._get_prior_losses(self.parallel_tree)
        prior_thres = self._get_prior_thres(self.parallel_tree)
        est_corr = (
            np.corrcoef(np.array(prior_test_nlls + [test_nlls]))
            if len(prior_test_nlls)
            else np.array([[1]])
        )
        t_thres = self._solve_t_statistic_thres(
            est_corr, prior_thres, self.test_tree.local_alpha
        )

        test_stat = (np.mean(test_nlls) - self.base_threshold) / std_err
        print("T THRES adjust", norm.cdf(t_thres), t_thres)
        test_result = int(test_stat < t_thres)
        print("corr test resl", test_stat, t_thres)
        return test_result

    def _get_test_compare_ffs(self, test_y, predef_pred_y, prev_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved,
        """
        loss_new = get_losses(test_y, predef_pred_y)
        loss_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = loss_new - loss_prev
        std_err = np.sqrt(np.var(loss_diffs) / loss_prev.size)
        self.parallel_tree.observe_losses(loss_diffs)

        # Need to traverse subfam parent nodes to decide local level
        prior_test_diffs = self._get_prior_losses(
            self.parallel_tree.parent
        )
        prior_thres = self._get_prior_thres(self.parallel_tree.parent)
        est_corr = (
            np.corrcoef(np.array(prior_test_diffs + [loss_diffs]))
            if len(prior_test_diffs)
            else np.array([[1]])
        )
        t_thres = self._solve_t_statistic_thres(
            est_corr, prior_thres, self.parallel_tree.local_alpha
        )
        self.parallel_tree.set_test_thres(t_thres)
        t_statistic = (np.mean(loss_diffs)) / std_err
        test_result = int(t_statistic < t_thres)
        print("COMPARE t_statistics", t_statistic, t_thres)
        return test_result

    def _get_test_compare_corr(self, test_y, pred_y, prev_pred_y):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved,
        """
        loss_new = get_losses(test_y, pred_y)
        loss_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = loss_new - loss_prev
        std_err = np.sqrt(np.var(loss_diffs) / loss_prev.size)
        self.test_tree.observe_losses(loss_diffs)

        prior_test_diffs = self._get_prior_losses(self.parallel_tree)
        prior_thres = self._get_prior_thres(self.parallel_tree)
        est_corr = (
            np.corrcoef(np.array(prior_test_diffs + [loss_diffs]))
            if len(prior_test_diffs)
            else np.array([[1]])
        )
        t_thres = self._solve_t_statistic_thres(
            est_corr, prior_thres, self.test_tree.local_alpha
        )

        test_stat = (np.mean(loss_diffs)) / std_err
        print("ORIG T THRES", norm.ppf(self.alpha / np.power(2, self.num_queries)))
        test_result = int(test_stat < t_thres)
        print("ADAPT COMPARE", test_stat, t_thres)
        return test_result

    def _do_tree_update(self, par_tree_res, adapt_tree_res):
        # update adaptive tree
        self.num_queries += 1
        self.parallel_test_hist.append(par_tree_res)
        self.test_hist.append(adapt_tree_res)

        if adapt_tree_res == 1:
            # remove node and propagate weights
            self.test_tree.success.earn(
                self.test_tree.weight * self.test_tree.success_edge
            )
            self.test_tree = self.test_tree.success
        else:
            self.test_tree.failure.earn(
                self.test_tree.weight * self.test_tree.failure_edge
            )
            self.test_tree = self.test_tree.failure
        self.test_tree.local_alpha = self.test_tree.weight * self.alpha

        self._create_children(self.test_tree)
        # Increment the par tree node regardless of success
        self.parallel_tree = self.parallel_tree.success
        self.parallel_tree.local_alpha = self.parallel_tree.weight * self.alpha

    def get_test_eval(self, test_y, pred_y, predef_pred_y):
        parallel_test_result = self._get_test_eval_ffs(
            test_y, predef_pred_y
        )
        test_result = self._get_test_eval_corr(test_y, pred_y)
        self._do_tree_update(parallel_test_result, test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y):
        parallel_test_result = self._get_test_compare_ffs(
            test_y, predef_pred_y, prev_pred_y
        )
        test_result = self._get_test_compare_corr(
            test_y, pred_y, prev_pred_y
        )
        self._do_tree_update(parallel_test_result, test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result
