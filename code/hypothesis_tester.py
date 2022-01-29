"""
Classes for testing specific hypotheses
All hypotheses here evaluate candidate modifications using multiple performance metrics
"""
from typing import List

import pandas as pd
import numpy as np
import scipy

from node import Node

class HypothesisTester:
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, mdl):
        raise NotImplementedError

    def test_null(self, node: Node, null_hypo: np.ndarray, prior_nodes: List):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        raise NotImplementedError()

class SensSpecHypothesisTester(HypothesisTester):
    stat_dim = 2
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat
        self.orig_obs = pd.DataFrame({
            "pos": self.test_dat.y.flatten(),
            "neg": (1 - self.test_dat.y).flatten(),
            })

    def get_observations(self, mdl):
        pred_y = mdl.predict(self.test_dat.x).reshape((-1, 1))
        df = pd.DataFrame({
            "equal_pos": ((self.test_dat.y == pred_y) * self.test_dat.y).flatten(),
            "equal_neg": ((self.test_dat.y == pred_y) * (1 - self.test_dat.y)).flatten()
            })
        return df

    def test_null(self, alpha: float, node: Node, null_constraint: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        raw_estimates = np.array([
                self.orig_obs.pos.mean(),
                self.orig_obs.neg.mean()]
            + [
                a for prior_node in prior_nodes
                for a in [prior_node.obs.equal_pos.mean(), prior_node.obs.equal_neg.mean()]]
            + [
                node.obs.equal_pos.mean(),
                node.obs.equal_neg.mean()]
            )
        estimate = np.array([
            raw_estimates[-2]/raw_estimates[0],
            raw_estimates[-1]/raw_estimates[1]
            ])

        full_df = pd.concat([self.orig_obs] + [prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
        if np.unique(full_df).size == 2:
            # All observations are binary
            # use a better estimate of variance in that case?
            probs = full_df.mean(axis=1)
            raw_covariance = np.diag(probs * (1 - probs))
            for i in range(full_df.shape[0]):
                for j in range(i + 1, full_df.shape[0]):
                    raw_covariance[i,j] = np.mean(full_df[i] * full_df[j]) - probs[i] * probs[j]
                    raw_covariance[j,i] = raw_covariance[i,j]
            raw_covariance /= self.test_dat.size
        else:
            raw_covariance = np.cov(full_df)/self.test_dat.size

        num_nodes = len(prior_nodes) + 1
        delta_d0 = np.array([dg_d0
            for i in range(num_nodes)
            for dg_d0 in [-raw_estimates[(i + 1) * self.stat_dim]/(raw_estimates[0]**2), 0]]).reshape((1,-1))
        delta_d1 = np.array([dg_d1
            for i in range(num_nodes)
            for dg_d1 in [0, -raw_estimates[(i + 1) * self.stat_dim + 1]/(raw_estimates[1]**2)]]).reshape((1,-1))
        delta_dother = scipy.linalg.block_diag(
                *[np.array([[1/raw_estimates[0],0],[0,1/raw_estimates[1]]])] * num_nodes)
        delta_grad = np.vstack([
            delta_d0, delta_d1, delta_dother
            ])
        cov_est = delta_grad.T @ raw_covariance @ delta_grad

        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        boundaries = self.generate_spending_boundaries(
           cov_est,
           self.stat_dim,
           alpha * node_weights,
           num_particles=max(self.test_dat.size * 4, 5000)
           )

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        min_norm = self.solve_min_norm(estimate, null_constraint)
        test_res = min_norm > boundaries[-1]
        print("ESTIMATE", estimate)
        print("TEST RES", test_res, min_norm, boundaries[-1])

        return test_res

    def solve_min_norm(self, estimate, null_constraint):
        """
        @return closest distance from point to the constraints (just project onto this space)
        """
        if np.all(estimate > null_constraint[:,1]):
            return np.min(np.abs(estimate - null_constraint[:,1]))
        return 0

    def generate_spending_boundaries(
            self,
            cov: np.ndarray,
            stat_dim: int,
            alpha_spend: np.ndarray,
            num_particles: int=5000,
            ):
        """
        Simulates particle paths for alpha spending
        Assumes the test at each iteration is H_0: theta_i < 0 for some i (for i in stat_dim)
        """
        good_particles = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=num_particles)
        boundaries = []
        for i, alpha in enumerate(alpha_spend):
            start_idx = stat_dim * i
            keep_alpha = alpha/(1 - alpha_spend[:i].sum())

            step_particles = good_particles[:, start_idx:start_idx + stat_dim]
            particle_mask = np.all(step_particles > 0, axis=1)
            step_norms = particle_mask * np.min(np.abs(step_particles), axis=1)
            step_bound = np.quantile(step_norms, 1 - keep_alpha)
            if np.mean(step_norms < step_bound) < keep_alpha:
                # If the step bound is weird, just increment it and let all particles thru
                step_bound += 1
            boundaries.append(step_bound)
            good_particles = good_particles[step_norms < step_bound]
        return np.array(boundaries)

class AcceptAccurHypothesisTester(SensSpecHypothesisTester):
    stat_dim = 2
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, mdl):
        pred_y = mdl.predict(self.test_dat.x).reshape((-1, 1))
        pred_decision = mdl.get_decision(self.test_dat.x).reshape((-1, 1))

        df = pd.DataFrame({
            "accept": pred_decision.flatten(),
            "accept_accuracy": ((self.test_dat.y == pred_y) * pred_decision).flatten()
            })
        return df

    def test_null(self, alpha: float, node: Node, null_constraint: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        raw_estimates = np.array([
                a for prior_node in prior_nodes
                for a in [prior_node.obs.accept.mean(), prior_node.obs.accept_accuracy.mean()]]
            + [
                node.obs.accept.mean(),
                node.obs.accept_accuracy.mean()]
            )
        estimate = np.array([
            raw_estimates[-2],
            raw_estimates[-1]/raw_estimates[-2]
            ])

        full_df = pd.concat([prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
        if np.unique(full_df).size == 2:
            # All observations are binary
            # use a better estimate of variance in that case?
            probs = full_df.mean(axis=1)
            raw_covariance = np.diag(probs * (1 - probs))
            for i in range(full_df.shape[0]):
                for j in range(i + 1, full_df.shape[0]):
                    raw_covariance[i,j] = np.mean(full_df[i] * full_df[j]) - probs[i] * probs[j]
                    raw_covariance[j,i] = raw_covariance[i,j]
            raw_covariance /= self.test_dat.size
        else:
            raw_covariance = np.cov(full_df)/self.test_dat.size

        num_nodes = len(prior_nodes) + 1
        delta_grad = scipy.linalg.block_diag(
                *[np.array([
                    [1,raw_estimates[i * 2 + 1]],
                    [0,1/raw_estimates[i * 2 + 1]]]) for i in range(num_nodes)])
        cov_est = delta_grad.T @ raw_covariance @ delta_grad

        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        boundaries = self.generate_spending_boundaries(
           cov_est,
           self.stat_dim,
           alpha * node_weights,
           num_particles=max(self.test_dat.size * 4, 5000)
           )

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        min_norm = self.solve_min_norm(estimate, null_constraint)
        test_res = min_norm > boundaries[-1]
        print("ACCEPT ACCUR ESTIMATE", estimate)
        print("TEST RES", test_res, min_norm, boundaries[-1])

        return test_res
