import logging

import numpy as np
from scipy.stats import norm, multivariate_normal
import scipy.optimize

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
        test_y = test_y.flatten()
        pred_y = np.maximum(np.minimum(1 - 1e-10, pred_y.flatten()), 1e-10)
        return -np.mean(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))

def get_losses(test_y, pred_y):
    test_y = test_y.flatten()
    pred_y = np.maximum(np.minimum(1 - 1e-10, pred_y.flatten()), 1e-10)
    test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
    return test_nlls


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

    def get_test_compare(self, test_y, pred_y, prev_pred_y):
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
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - self.alpha/self.correction_factor)
        return int(upper_ci < self.base_threshold)

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
    def __init__(self, alpha, success_edge, history, subfam_root=None, parent=None):
        """
        @param subfam_root: which node is the subfamily's root node. if none, this is the root
        """
        self.success = None
        self.success_edge = success_edge
        self.failure_edge = 1 -  success_edge
        self.failure = None
        self.alpha = alpha
        self.history = history
        self.subfam_root = subfam_root if subfam_root is not None else self
        self.parent = parent

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

    #def _create_tree(self, node, tree_depth):
    #    if tree_depth == 0:
    #        return

    #    node.success = Node(0, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
    #    node.failure = Node(0, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)
    #    self._create_tree(node.success, tree_depth - 1)
    #    self._create_tree(node.failure, tree_depth - 1)

    def _create_children(self, node):
        node.success = Node(0, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
        node.failure = Node(0, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(self.alpha, success_edge=self.success_weight, history=[], subfam_root=None)
        self._create_children(self.test_tree)

        # reset num queries
        self.num_queries = 0
        self.test_hist = []

    def _do_tree_update(self, test_result):
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

        self._create_children(self.test_tree)

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls = get_losses(test_y, pred_y)
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        alpha_level = self.test_tree.alpha * self.test_tree.success_edge
        print("ALPHA?", self.test_tree.alpha)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - alpha_level)
        print("upper ci", np.mean(test_nlls), upper_ci)
        test_result = int(upper_ci < self.base_threshold)

        self._do_tree_update(test_result)
        return test_result

    def get_test_compare(self, test_y, pred_y, prev_pred_y):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_prev = get_losses(test_y, prev_pred_y)
        loss_diffs = test_nlls_new - test_nlls_prev
        t_stat_se = np.sqrt(np.var(loss_diffs)/loss_diffs.size)
        alpha_level = self.test_tree.alpha * self.test_tree.success_edge
        print("alpha", alpha_level)
        upper_ci = np.mean(loss_diffs) + t_stat_se * norm.ppf(1 - alpha_level)
        test_result = int(upper_ci < 0)
        print("SDJFKLJSDFLJKSDF")

        self._do_tree_update(test_result)
        return test_result

class GraphicalParallelDP(GraphicalBonfDP):
    """
    Split alpha evenly across nodes generated by the "parallel" online procedure
    Model developer PRESPECIFies a parallel online procedure
    AND assumes correlation structure among models in a level
    """
    @property
    def name(self):
        return "graphical_parallel_%d" % self.loss_to_diff_std_ratio

    def __init__(self, base_threshold, alpha, success_weight, parallel_success_weight: float = 0.1, parallel_ratio: float = 0.6, loss_to_diff_std_ratio: float = 100.0):
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

    #def _create_tree(self, node, tree_depth, node_dict):
    #    if tree_depth == 0:
    #        return

    #    node.success = Node(0, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
    #    node.failure = Node(0, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)
    #    node_dict[len(node.history) + 1] += [node.success, node.failure]
    #    self._create_tree(node.success, tree_depth - 1, node_dict)
    #    self._create_tree(node.failure, tree_depth - 1, node_dict)

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(1 - self.parallel_ratio, success_edge=self.success_weight, history=[], subfam_root=None)
        self.num_same_level_nodes = 1
        self.same_level_node = Node(0, success_edge=self.success_weight, history=[], subfam_root=None)

        # Tracks nodes in each level
        self.node_dict = {i: [] for i in range(num_adapt_queries + 1)}
        self.node_dict[0] = [self.test_tree]
        self._create_children(self.test_tree)

        self.parallel_tree = Node(1/num_adapt_queries * self.parallel_ratio, success_edge=self.parallel_success_weight, history=[], subfam_root=None)
        curr_par_node = self.parallel_tree
        for i in range(1, num_adapt_queries + 1):
            next_par_node = Node(1/num_adapt_queries * self.parallel_ratio, success_edge=self.parallel_success_weight, history=[1] * i, subfam_root=None)
            curr_par_node.par_child = next_par_node
            curr_par_node.par_weight  = self.parallel_success_weight
            curr_par_node.adapt_tree_node_weight  = (1 - self.parallel_success_weight)/np.power(2, i - 1)

            curr_par_node = next_par_node
            curr_par_node.par_child = None

        # reset num queries
        self.num_queries = 0
        self.test_hist = []
        self.parallel_test_hist = []

    def _get_test_eval(self, test_y, pred_y, predef_pred_y, alpha):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls = get_losses(test_y, pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls)/test_nlls.size)

        test_stat = (np.mean(test_nlls) - self.base_threshold)/std_err
        print("95 upper ci", np.mean(test_nlls) + 1.96 * std_err)
        print("TEST EvAL", test_stat, std_err, np.mean(test_nlls), self.base_threshold)
        test_result = int(norm.cdf(test_stat) < alpha)
        return test_result

    def _get_test_compare(self, test_y, pred_y, baseline_pred_y, predef_pred_y, alpha):
        """
        NOTICE that the std err used here is not the usual one!!!

        @return test perf where 1 means approve and 0 means not approved
        """
        test_nlls_new = get_losses(test_y, pred_y)
        test_nlls_old = get_losses(test_y, baseline_pred_y)
        predef_pred_test_nlls = get_losses(test_y, predef_pred_y)
        std_err = np.sqrt(np.var(predef_pred_test_nlls - test_nlls_old)/test_nlls_old.size)

        test_stat = np.mean(test_nlls_new - test_nlls_old)/std_err
        print("TEST EvAL", test_stat, std_err, np.mean(test_nlls_new - test_nlls_old), alpha)
        test_result = int(norm.cdf(test_stat) < alpha)
        return test_result

    def _get_fwer(self, q, alpha_weights, desired_fwer, debug=False):
        def fwer_calc(c_thres):
            tot_alpha = 1 - norm.cdf(c_thres)
            offsets = norm.ppf(1 - alpha_weights * desired_fwer * q)
            assert np.all(offsets >= c_thres)
            extra_alpha = np.sum(1 - norm.cdf((offsets - c_thres) * self.loss_to_diff_std_ratio))
            #print("ALPHA SPLIT", tot_alpha, extra_alpha)
            return tot_alpha + extra_alpha

        max_c_thres = norm.ppf(1 - alpha_weights.max() * q * desired_fwer)
        res = scipy.optimize.minimize_scalar(fwer_calc, bounds=[0, max_c_thres], method="bounded")
        #print("local alpha", res.fun, "desired fwer", desired_fwer)
        print("FWER...", res.fun, desired_fwer, (res.fun - desired_fwer)/desired_fwer, np.abs(res.fun - desired_fwer)/desired_fwer)
        #assert ((res.fun - desired_fwer)/desired_fwer < 1e-2) or (np.abs(res.fun - desired_fwer)/desired_fwer < 0.1)
        #print("INNER OPTIM", res.fun, res.x)
        return res.fun

    def get_corrected_fwer_thresholds(self, alpha_weights, desired_fwer):
        def fwer_dist(x):
            fwer = self._get_fwer(x, alpha_weights[alpha_weights > 0], desired_fwer)
            return np.power(np.abs(fwer - desired_fwer)/desired_fwer, 2)

        # Now fine tune it using an optimization algo. maybe it'll tell us something new
        if np.sum(alpha_weights > 0) == 1:
            return alpha_weights * desired_fwer
        else:
            min_alpha_weight = alpha_weights[alpha_weights > 0].max()
            new_fwer_res = scipy.optimize.minimize_scalar(fwer_dist, bounds=[1, 1/min_alpha_weight], method="bounded")
            fwer_inflat = new_fwer_res.x
            print("alpha", alpha_weights)
            print("SEPNT", self._get_fwer(fwer_inflat, alpha_weights, desired_fwer), desired_fwer)
            return alpha_weights * fwer_inflat * desired_fwer

    def _do_adapt_tree_update(self, test_result):
        # update adaptive tree
        self.num_queries += 1
        self.test_hist.append(test_result)
        if test_result == 1:
            # remove node and propagate weights
            self.test_tree.success.earn(self.test_tree.alpha * self.test_tree.success_edge)
            self.test_tree = self.test_tree.success

            # TODO: there are other same level nodes with other weights. need to be a bit more careful here
            self.same_level_node = Node(self.test_tree.alpha * self.test_tree.failure_edge, success_edge = 0, history=[0] * len(self.test_tree.history))
        else:
            self.test_tree.failure.earn(self.test_tree.alpha * self.test_tree.failure_edge)
            self.test_tree = self.test_tree.failure

            self.same_level_node = Node(0, success_edge = 0, history=[0] * len(self.test_tree.history))

        self.num_same_level_nodes = np.power(2, len(self.test_tree.history))
        self._create_children(self.test_tree)

    def _do_par_tree_update(self, test_result):
        # Update parallel tree
        self.parallel_test_hist.append(test_result)

        if test_result == 1:
            self.parallel_tree.par_child.earn(self.parallel_tree.alpha * self.parallel_tree.par_weight)
            self.test_tree.earn(self.parallel_tree.alpha * self.parallel_tree.adapt_tree_node_weight)
            self.same_level_node = Node(self.parallel_tree.alpha * self.parallel_tree.adapt_tree_node_weight, success_edge = 0, history=[0] * len(self.test_tree.history))
            self.num_same_level_nodes = np.power(2, len(self.test_tree.history))

        # Increment the par tree node regardless of success
        self.parallel_tree = self.parallel_tree.par_child

    def get_test_eval(self, test_y, pred_y, predef_pred_y):
        parallel_test_result = self._get_test_eval(test_y, predef_pred_y, predef_pred_y, self.parallel_tree.alpha * self.alpha)
        self._do_par_tree_update(parallel_test_result)

        # get the corrected levels for testing the current node, accounting for modification similarity
        curr_level_alphas = np.array([self.test_tree.alpha] + [self.same_level_node.alpha] * (self.num_same_level_nodes - 1))
        desired_fwer = np.sum(curr_level_alphas) * self.alpha
        if desired_fwer == 0:
            test_result = 0
        else:
            alpha_weights = (curr_level_alphas + np.power(10.,-self.num_adapt_queries))/(np.sum(curr_level_alphas) + np.power(10.,-self.num_adapt_queries) * curr_level_alphas.size)
            corrected_alphas = self.get_corrected_fwer_thresholds(alpha_weights = alpha_weights, desired_fwer=desired_fwer)
            test_result = self._get_test_eval(test_y, pred_y, predef_pred_y, corrected_alphas[0])

        # update tree
        self._do_adapt_tree_update(test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result

    def get_test_compare(self, test_y, pred_y, prev_pred_y, predef_pred_y):
        parallel_test_result = self._get_test_compare(test_y, predef_pred_y, prev_pred_y, predef_pred_y, alpha=self.parallel_tree.alpha * self.alpha)
        self._do_par_tree_update(parallel_test_result)

        # get the corrected levels for testing the current node, accounting for modification similarity
        curr_level_alphas = np.array([self.test_tree.alpha] + [self.same_level_node.alpha] * (self.num_same_level_nodes - 1))
        desired_fwer = np.sum(curr_level_alphas) * self.alpha
        if desired_fwer == 0:
            test_result = 0
        else:
            print("orig alph", curr_level_alphas)
            alpha_weights = (curr_level_alphas + np.power(10.,-self.num_adapt_queries))/(np.sum(curr_level_alphas) + np.power(10.,-self.num_adapt_queries) * curr_level_alphas.size)
            print("alpha weights", alpha_weights)
            corrected_alphas = self.get_corrected_fwer_thresholds(alpha_weights = alpha_weights, desired_fwer=desired_fwer)
            print("CORRE ALPHA", corrected_alphas)
            test_result = self._get_test_compare(test_y, pred_y, prev_pred_y, predef_pred_y, alpha=corrected_alphas[0])

        # update tree
        self._do_adapt_tree_update(test_result)

        print("PARALLL", self.parallel_test_hist)
        print("TEST TREE", self.test_hist)
        return test_result

class GraphicalFFSDP(GraphicalBonfDP):
    name = "graphical_ffs"

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        self.test_tree = Node(self.alpha, success_edge=self.success_weight, history=[])
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


    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        self.test_tree.observe_losses(test_nlls)
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        # compute critical levels
        alpha_level = self.test_tree.alpha * self.test_tree.success_edge
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

class GraphicalSimilarityDP(GraphicalBonfDP):
    """
    Assumes the modeler will not test very different models at the same tree depth
    """
    name = "graphical_similarity"

    def __init__(self, base_threshold, alpha, success_weight, similarity: float = 1):
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.success_weight = success_weight
        assert similarity == 1
        self.similarity = similarity

    def _create_tree(self, node, tree_depth, alpha_alloc):
        if tree_depth == 0:
            return

        # TODO: this allocation only makes senses when all the nodes are super super similar
        node.success = Node(alpha_alloc, success_edge=self.success_weight, history=node.history + [1], subfam_root=None, parent=node)
        node.failure = Node(alpha_alloc, success_edge=self.success_weight, history=node.history + [0], subfam_root=node.subfam_root, parent=node)
        self._create_tree(node.success, tree_depth - 1, alpha_alloc)
        self._create_tree(node.failure, tree_depth - 1, alpha_alloc)

    def set_num_queries(self, num_adapt_queries):
        self.num_adapt_queries = num_adapt_queries
        # Use  the alpha value as a probability weight, not an alpha allocation
        self.test_tree = Node(alpha=1/num_adapt_queries, success_edge=self.success_weight, history=[], subfam_root=None)
        self._create_tree(self.test_tree, num_adapt_queries, alpha_alloc=1/num_adapt_queries)

        # reset num queries
        self.num_queries = 0
        self.test_hist = []

    def get_similarity_correction_factor(self, node):
        chain_node = node
        tot_alpha_spent = 0
        while chain_node.parent is not None:
            if chain_node.history[-1] == 0:
                # If we didn't reject this hypothesis, this hypothesis will count towards allocation
                tot_alpha_spent += chain_node.parent.alpha * chain_node.parent.success_edge
            chain_node = chain_node.parent

        # TODO: account for similarity value

        # assuming exactly the same when in the same level
        # rejecting an earlier level rejects all in the same level
        print("ALPHA spent", tot_alpha_spent, node.alpha)
        Q_factor = tot_alpha_spent + node.alpha
        assert Q_factor <= 1
        return Q_factor

    def get_test_eval(self, test_y, pred_y, predef_pred_y=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        test_y = test_y.flatten()
        pred_y = pred_y.flatten()
        test_nlls = -(np.log(pred_y) * test_y + np.log(1 - pred_y) * (1 - test_y))
        t_stat_se = np.sqrt(np.var(test_nlls)/test_nlls.size)

        correction_factor = self.get_similarity_correction_factor(self.test_tree)
        alpha_level = self.alpha * self.test_tree.alpha * self.test_tree.success_edge / correction_factor
        print("alpha. corr", alpha_level, correction_factor)
        upper_ci = np.mean(test_nlls) + t_stat_se * norm.ppf(1 - alpha_level)
        print("upper ci", np.mean(test_nlls), upper_ci)
        test_result = int(upper_ci < self.base_threshold)

        # update tre
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
