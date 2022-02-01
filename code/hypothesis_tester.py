"""
Classes for testing specific hypotheses
All hypotheses here evaluate candidate modifications using multiple performance metrics
"""
import logging
from typing import List

import pandas as pd
import numpy as np
import scipy
import sklearn

from node import Node

MAX_PARTICLES = 500000
MIN_PARTICLES = 5000

def get_log_lik(y_true, y_pred):
    y_pred = np.minimum(np.maximum(y_pred, 1e-10), 1 - 1e-10)
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)


class HypothesisTester:
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, orig_mdl, new_mdl):
        raise NotImplementedError

    def test_null(self, node: Node, null_hypo: np.ndarray, prior_nodes: List):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        raise NotImplementedError()

class LogLikHypothesisTester(HypothesisTester):
    stat_dim = 1
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, orig_mdl, new_mdl):
        orig_pred_y = orig_mdl.predict_proba(self.test_dat.x)[:,1]
        new_pred_y = new_mdl.predict_proba(self.test_dat.x)[:,1]
        test_y = self.test_dat.y.flatten()
        log_lik_diff = get_log_lik(test_y, new_pred_y) - get_log_lik(test_y, orig_pred_y)
        df = pd.DataFrame({
            "log_lik_diff": log_lik_diff,
            })

        return df

    def test_null(self, alpha: float, node: Node, null_constraint: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        estimate = node.obs.log_lik_diff.mean()
        logging.info("test set log lik %s", estimate)

        full_df = pd.concat([prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
        cov_est = np.cov(full_df)/self.test_dat.size
        if full_df.shape[0] == 1:
            cov_est = np.array([[cov_est]])

        num_nodes = len(prior_nodes) + 1
        assert not np.any(np.isnan(cov_est))

        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        num_particles =  min(int(np.sum(1/(alpha * node_weights))), MAX_PARTICLES)
        #assert num_particles <= MAX_PARTICLES
        boundaries = self.generate_spending_boundaries(
           cov_est,
           self.stat_dim,
           alpha * node_weights,
           num_particles=min(max(num_particles, MIN_PARTICLES), MAX_PARTICLES)
           )

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        min_norm = max(0, estimate - null_constraint[0,1])
        test_res = min_norm > boundaries[-1]
        print("TEST RES", test_res, min_norm, boundaries[-1])
        logging.info("alpha level %f, bound %f", alpha * node_weights[-1], boundaries[-1])
        logging.info("norm %f", min_norm)
        if num_particles == MAX_PARTICLES:
            logging.info("MAX PARTICLES REACHED")

        return test_res

    def generate_spending_boundaries(
            self,
            cov,
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
            keep_ratio = np.mean(step_norms < step_bound)
            # if the keep ratio is not close to what we desired, do not rejecanything
            logging.info("keep ratio %f", (1 - keep_ratio)/keep_alpha)
            print("KEEP RATIO", keep_ratio, keep_alpha, (1 - keep_ratio)/keep_alpha)
            if keep_ratio < keep_alpha or (1 - keep_ratio)/keep_alpha > 2:
                print(np.max(step_norms), step_bound)
                # If the step bound is weird, do not reject anything
                step_bound = np.max(step_norms)
                # step_bound += 1
            boundaries.append(step_bound)
            good_particles = good_particles[step_norms < step_bound]
        return np.array(boundaries)

class SensSpecHypothesisTester(HypothesisTester):
    stat_dim = 2
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat
        self.orig_obs = pd.DataFrame({
            "pos": self.test_dat.y.flatten(),
            "neg": (1 - self.test_dat.y).flatten(),
            })

    def get_observations(self, orig_mdl, new_mdl):
        orig_pred_y = orig_mdl.predict(self.test_dat.x)
        new_pred_y = new_mdl.predict(self.test_dat.x)
        test_y = self.test_dat.y.flatten()
        acc_diff = (test_y == new_pred_y).astype(int) - (test_y == orig_pred_y).astype(int)
        df = pd.DataFrame({
            "equal_pos_diff": (acc_diff * test_y).flatten(),
            "equal_neg_diff": (acc_diff * (1 - test_y)).flatten()
            })

        # for fun -- auc check?
        logging.info("test accuracy orig %f", np.mean(test_y == orig_pred_y))
        logging.info("test accuracy new %f", np.mean(test_y == new_pred_y))
        orig_pred_y = orig_mdl.predict_proba(self.test_dat.x)[:,1]
        new_pred_y = new_mdl.predict_proba(self.test_dat.x)[:,1]
        logging.info("test auc orig %f", sklearn.metrics.roc_auc_score(test_y, orig_pred_y))
        logging.info("test auc new %f", sklearn.metrics.roc_auc_score(test_y, new_pred_y))
        logging.info("test log loss orig %f", sklearn.metrics.log_loss(test_y, orig_pred_y))
        logging.info("test log loss new %f", sklearn.metrics.log_loss(test_y, new_pred_y))

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
                for a in [prior_node.obs.equal_pos_diff.mean(), prior_node.obs.equal_neg_diff.mean()]]
            + [
                node.obs.equal_pos_diff.mean(),
                node.obs.equal_neg_diff.mean()]
            )
        estimate = np.array([
            raw_estimates[-2]/raw_estimates[0],
            raw_estimates[-1]/raw_estimates[1]
            ])
        logging.info("test set estimate %s", estimate.flatten())

        full_df = pd.concat([self.orig_obs] + [prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
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
        assert not np.any(np.isnan(cov_est))

        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        num_particles =  min(int(np.sum(1/(alpha * node_weights))), MAX_PARTICLES)
        print("NUM PARTC", num_particles, node_weights * alpha)
        #assert num_particles <= MAX_PARTICLES
        boundaries = self.generate_spending_boundaries(
           cov_est,
           self.stat_dim,
           alpha * node_weights,
           num_particles=min(max(num_particles, MIN_PARTICLES), MAX_PARTICLES)
           )

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        min_norm = self.solve_min_norm(estimate, null_constraint)
        test_res = min_norm > boundaries[-1]
        print("TEST RES", test_res, min_norm, boundaries[-1])
        logging.info("alpha level %f, bound %f", alpha * node_weights[-1], boundaries[-1])
        logging.info("norm %f", min_norm)
        if num_particles == MAX_PARTICLES:
            logging.info("MAX PARTICLES REACHED")

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
            keep_ratio = np.mean(step_norms < step_bound)
            # if the keep ratio is not close to what we desired, do not rejecanything
            logging.info("keep ratio %f", (1 - keep_ratio)/keep_alpha)
            print("KEEP RATIO", keep_ratio, keep_alpha, (1 - keep_ratio)/keep_alpha)
            if keep_ratio < keep_alpha or (1 - keep_ratio)/keep_alpha > 2:
                print(np.max(step_norms), step_bound)
                # If the step bound is weird, do not reject anything
                step_bound = np.max(step_norms)
                # step_bound += 1
            boundaries.append(step_bound)
            good_particles = good_particles[step_norms < step_bound]
        return np.array(boundaries)

class AcceptAccurHypothesisTester(SensSpecHypothesisTester):
    stat_dim = 2
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, orig_mdl, new_mdl):
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
        assert not np.any(np.isnan(cov_est))

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
