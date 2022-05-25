"""
Classes that perform multiple hypothesis testing
"""
import logging
from typing import List

import scipy.special
import numpy as np

from node import Node
from hypothesis_tester import HypothesisTester

class BinaryThresholdMTP:
    """
    Repeatedly test at level alpha
    """
    require_predef = False
    name = "binary_thres"

    def __init__(self, hypo_tester, alpha: float):
        self.hypo_tester = hypo_tester
        self.alpha = alpha
        self.correction_factor = 1

    @property
    def curr_alpha(self):
        return self.alpha * self.test_tree.weight

    @property
    def test_set_size(self):
        return self.hypo_tester.test_dat.size

    def _create_children(self, node, query_idx):
        """
        @param node: create children for this node
        @param query_idx: which adaptive query number we are on
        """
        children = [Node(
            weight=self.correction_factor,
            history=node.history + ([1] if query_idx >= 0 else []) + [0] * i,
            parent=node,
            ) for i in range(self.num_adapt_queries - query_idx - 1)]
        node.children = children
        node.children_weights = [0 for i in range(len(children))]

    def _do_tree_update(self, test_result):
        # update tree
        self.num_queries += 1

        if self.num_queries >= self.num_adapt_queries:
            # We are done
            #print("TREE UPDATE DONE")
            #1/0
            return

        if test_result == 1:
            print("**EARN weights", self.test_tree.weight)
            for child, cweight in zip(self.test_tree.children, self.test_tree.children_weights):
                child.weight += cweight * self.test_tree.weight
            self.parent_child_idx = 0
            self.test_tree = self.test_tree.children[self.parent_child_idx]
        else:
            print("**FAIL, parent", self.test_tree.parent)
            self.parent_child_idx += 1
            self.test_tree = self.test_tree.parent.children[self.parent_child_idx]
        self._create_children(self.test_tree, self.num_queries)
        self.test_tree.local_alpha = self.alpha * self.test_tree.weight

    def init_test_dat(self, test_dat, num_adapt_queries):
        raise NotImplementedError("need to define init_test_dat")

    def get_test_res(self, null_hypo: np.ndarray, orig_mdl, new_mdl, orig_predef_mdl=None, predef_mdl=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        node_obs = self.hypo_tester.get_observations(orig_mdl, new_mdl)
        self.test_tree.store_observations(node_obs)
        test_res, _  = self.hypo_tester.test_null(self.alpha, self.test_tree, null_hypo, prior_nodes=[])

        self._do_tree_update(test_res)

        return test_res

class BonferroniThresholdMTP(BinaryThresholdMTP):
    require_predef = False
    name = "bonferroni"

    def init_test_dat(self, test_dat, num_adapt_queries):
        self.hypo_tester.set_test_dat(test_dat)

        self.num_queries = -1
        self.num_adapt_queries = num_adapt_queries
        self.correction_factor = 1/(np.power(2, num_adapt_queries) - 1)

        self.start_node = Node(
            weight=self.correction_factor,
            history=[],
            parent=None,
        )
        self._create_children(self.start_node, self.num_queries)
        self.test_tree = self.start_node
        self._do_tree_update(1)

class WeightedBonferroniThresholdMTP(BinaryThresholdMTP):
    require_predef = False
    name = "weighted_bonferroni"

    def __init__(self, hypo_tester, alpha: float, bad_attempt_thres: int):
        self.hypo_tester = hypo_tester
        self.alpha = alpha
        self.bad_attempt_thres = bad_attempt_thres

    def _create_children(self, node, query_idx):
        """
        @param node: create children for this node
        @param query_idx: which adaptive query number we are on
        """
        children = [Node(
            weight=self.small_correction_factor if (np.array(node.history) == 0).sum() <= self.bad_attempt_thres else self.big_correction_factor,
            history=node.history + ([1] if query_idx >= 0 else []) + [0] * i,
            parent=node,
            ) for i in range(self.num_adapt_queries - query_idx - 1)]
        node.children = children
        node.children_weights = [0 for i in range(len(children))]


    def init_test_dat(self, test_dat, num_adapt_queries):
        self.hypo_tester.set_test_dat(test_dat)

        self.num_queries = -1
        self.num_adapt_queries = num_adapt_queries
        num_queries_below_thres = sum([
                sum([
                    scipy.special.binom(query_idx, j)
                    for j in range(min(query_idx + 1, self.bad_attempt_thres + 1))])
                for query_idx in range(num_adapt_queries)])
        num_queries_above_thres = np.power(2, num_adapt_queries) - 1 - num_queries_below_thres
        self.small_correction_factor = 0.5/num_queries_below_thres
        self.big_correction_factor = 0.5/num_queries_above_thres
        assert num_queries_above_thres >= 0

        self.start_node = Node(
            weight=self.small_correction_factor,
            history=[],
            parent=None,
        )
        self._create_children(self.start_node, self.num_queries)
        self.test_tree = self.start_node
        self._do_tree_update(1)

class GraphicalBonfMTP(BinaryThresholdMTP):
    require_predef = False
    name = "graphical_bonf_thres"

    def __init__(
        self,
        hypo_tester,
        alpha,
        success_weight,
        alpha_alloc_max_depth: int = 0,
        scratch_file: str = None,
    ):
        self.hypo_tester = hypo_tester
        self.alpha = alpha
        self.success_weight = success_weight
        assert alpha_alloc_max_depth == 0
        self.alpha_alloc_max_depth = alpha_alloc_max_depth
        self.parallel_ratio = 0
        self.scratch_file = scratch_file

    def _create_children(self, node, query_idx):
        """
        @param node: create children for this node
        @param query_idx: which adaptive query number we are on
        """
        children = [Node(
            weight=0,
            history=node.history + ([1] if query_idx >= 0 else []) + [0] * i,
            parent=node,
            ) for i in range(self.num_adapt_queries - query_idx - 1)]
        node.children = children
        node.children_weights = []
        for i in range(0, self.num_adapt_queries - query_idx - 1):
            spent_weight = sum(node.children_weights)
            node.children_weights.append((1 - spent_weight) * self.success_weight)
        if children:
            node.children_weights[-1] = 1 - np.sum(node.children_weights[:-1])

    def init_test_dat(self, test_dat, num_adapt_queries):
        self.hypo_tester.set_test_dat(test_dat)

        self.num_queries = -1
        self.num_adapt_queries = num_adapt_queries

        self.start_node = Node(
            1,
            history=[],
            parent=None,
        )
        self._create_children(self.start_node, self.num_queries)
        self.test_tree = self.start_node

        # propagate weights from start node
        self._do_tree_update(1)

        self.parent_child_idx = 0

    def get_test_res(self, null_hypo: np.ndarray, orig_mdl, new_mdl, orig_predef_mdl=None,  predef_mdl=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        node_obs = self.hypo_tester.get_observations(orig_mdl, new_mdl)
        self.test_tree.store_observations(node_obs)
        test_res, _ = self.hypo_tester.test_null(self.alpha, self.test_tree, null_hypo, prior_nodes=[])
        self._do_tree_update(test_res)
        return test_res

class GraphicalFFSMTP(GraphicalBonfMTP):
    require_predef = False
    name = "graphical_ffs"

    def get_test_res(self, null_hypo: np.ndarray, orig_mdl, new_mdl, orig_predef_mdl=None,  predef_mdl=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        node_obs = self.hypo_tester.get_observations(orig_mdl, new_mdl)
        self.test_tree.store_observations(node_obs)
        prior_nodes = self.test_tree.parent.children[:(self.parent_child_idx)]
        test_res, bound = self.hypo_tester.test_null(self.alpha, self.test_tree, null_hypo, prior_nodes=prior_nodes)
        self.test_tree.bounds = bound

        self._do_tree_update(test_res)

        return test_res

class GraphicalParallelMTP(GraphicalFFSMTP):
    """
    Split alpha evenly across nodes generated by the "parallel" online procedure
    Model developer PRESPECIFies a parallel online procedure
    AND assumes correlation structure among models in a level
    """
    require_predef = True
    name = "graphical_par"

    def __init__(
        self,
        hypo_tester,
        alpha,
        success_weight,
        parallel_ratio: float = 0.9,
        first_pres_weight: float = 0.5,
        alpha_alloc_max_depth: int = 0,
        scratch_file: str = None,
    ):
        self.hypo_tester = hypo_tester
        self.alpha = alpha
        self.success_weight = success_weight
        self.parallel_ratio = parallel_ratio
        self.first_pres_weight = first_pres_weight
        self.alpha_alloc_max_depth = alpha_alloc_max_depth
        self.scratch_file = scratch_file

    def init_test_dat(self, test_dat, num_adapt_queries):
        self.hypo_tester.set_test_dat(test_dat)

        self.num_queries = -1
        self.num_adapt_queries = num_adapt_queries

        # Create parallel sequence
        self.parallel_tree_nodes = []
        self.parallel_tree_nodes.append(Node(
            self.first_pres_weight * self.parallel_ratio,
            history=[],
        ))
        for i in range(1, num_adapt_queries + 1):
            weight = (
                (1 - self.first_pres_weight)/num_adapt_queries * self.parallel_ratio
                if i < num_adapt_queries
                else 0
                )
            self.parallel_tree_nodes.append(Node(
                weight,
                history=[None] * i,
                parent=None
            ))

        # Create adapt tree
        self.start_node = Node(
            1 - self.parallel_ratio,
            history=[],
            parent=None,
        )
        self._create_children(self.start_node, self.num_queries)
        self.test_tree = self.start_node

        # propagate weights from start node (but dont do any updating in the parallel)
        self._do_tree_update(1)

        self.parent_child_idx = 0

    def _do_tree_update(self, adapt_tree_res):
        # update adaptive tree
        self.num_queries += 1
        if self.num_queries >= self.num_adapt_queries:
            return

        if adapt_tree_res == 1:
            print("EARN weights")
            for child, cweight in zip(self.test_tree.children, self.test_tree.children_weights):
                child.weight += cweight * self.test_tree.weight
            self.parent_child_idx = 0
            self.test_tree = self.test_tree.children[self.parent_child_idx]
        else:
            self.parent_child_idx += 1
            print("num childs", len(self.test_tree.parent.children), self.parent_child_idx)
            self.test_tree = self.test_tree.parent.children[self.parent_child_idx]
        self._create_children(self.test_tree, self.num_queries)
        self.test_tree.local_alpha = self.alpha * self.test_tree.weight

    def get_test_res(self, null_hypo: np.ndarray, orig_mdl, new_mdl, orig_predef_mdl=None, predef_mdl=None):
        """
        @return test perf where 1 means approve and 0 means not approved
        """
        parallel_node = self.parallel_tree_nodes[self.num_queries]
        parallel_node_obs = self.hypo_tester.get_observations(orig_predef_mdl, predef_mdl)
        parallel_node.store_observations(parallel_node_obs)
        _, parallel_node_bounds = self.hypo_tester.test_null(self.alpha, parallel_node, null_hypo, prior_nodes=self.parallel_tree_nodes[:self.num_queries])
        parallel_node.bounds = parallel_node_bounds

        node_obs = self.hypo_tester.get_observations(orig_mdl, new_mdl)
        self.test_tree.store_observations(node_obs)
        prior_nodes = self.parallel_tree_nodes[:(self.num_queries + 1)]
        print("HIST", self.test_tree.history)
        test_res, node_bounds = self.hypo_tester.test_null(self.alpha, self.test_tree, null_hypo, prior_nodes=prior_nodes)
        self.test_tree.bounds = node_bounds

        self._do_tree_update(test_res)

        return test_res
