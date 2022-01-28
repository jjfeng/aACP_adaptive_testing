from typing import List

import pandas as pd
import numpy as np
import scipy

from node import Node

def _get_predictions(mdl, x):
    return mdl.predict(x).reshape((-1, 1))


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
        pred_y = _get_predictions(mdl, self.test_dat.x)
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
            self.orig_obs.neg.mean(),
            node.obs.equal_pos.mean(),
            node.obs.equal_neg.mean()
            ])
        estimate = np.array([
            raw_estimates[2]/raw_estimates[0],
            raw_estimates[3]/raw_estimates[1]
            ])

        #full_df = pd.concat([self.orig_obs, node.obs], axis=1).to_numpy().T
        #raw_covariance = np.cov(full_df)/self.test_dat.size

        #delta_grad = np.array([
        #    [-raw_estimates[2]/(raw_estimates[0]**2), 0],
        #    [0, -raw_estimates[3]/(raw_estimates[1]**2)],
        #    [1/raw_estimates[0], 0],
        #    [0, 1/raw_estimates[1]],
        #    ])
        #cov_est = delta_grad.T @ raw_covariance @ delta_grad
        #print("cov est", cov_est)

        ###
        # NEW SECTION
        ###
        #print("PRIOR NODES", prior_nodes)
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
        #print("cov est 1", cov_est)

        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        boundaries = generate_chi2_spending_boundaries(
           cov_est,
           self.stat_dim,
           alpha * node_weights,
           )
        print("BOUDNDARIES", boundaries)

        def get_norm(test_pt):
            return np.sum(np.power(estimate - test_pt, 2))

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        opt0_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
            (null_constraint[0,0], null_constraint[0,1]),
            (0,1)])
        opt1_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
            (0,1),
            (null_constraint[1,0], null_constraint[1,1])])
        #print(opt0_res)
        #print(opt1_res)
        assert opt0_res.success and opt1_res.success
        test_res = min(opt0_res.fun, opt1_res.fun) > boundaries[-1]
        print("ESTIMATE", estimate)
        print("TEST RES", test_res, opt0_res.fun, opt1_res.fun, boundaries[-1])

        ####
        # END OF NEW
        ####

        # check if null holds given the estimate
        #precision_mat = np.linalg.inv(cov_est)
        #def get_norm(test_pt):
        #    dist = (estimate - test_pt).reshape((-1,1))
        #    return (dist.T @ precision_mat @ dist)[0,0]

        ## Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        #opt0_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
        #    (null_constraint[0,0], null_constraint[0,1]),
        #    (0,1)])
        #opt1_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
        #    (0,1),
        #    (null_constraint[1,0], null_constraint[1,1])])

        #chi2_df2 = scipy.stats.chi2(df=2)
        #print("PVALS", 1 - chi2_df2.cdf(opt0_res.fun), 1 - chi2_df2.cdf(opt1_res.fun))
        #pval = 1 - min(chi2_df2.cdf(opt0_res.fun), chi2_df2.cdf(opt1_res.fun))
        #print("estim", estimate)
        #print("p-value", pval, "pthres", node.weight * alpha)
        #test_res = pval < (node.weight * alpha)
        #1/0

        return test_res

def generate_chi2_spending_boundaries(
        cov: np.ndarray,
        stat_dim: int,
        alpha_spend: np.ndarray,
        num_particles: int=1000,
        ):
        good_particles = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=num_particles)
        boundaries = []
        for i, alpha in enumerate(alpha_spend):
            start_idx = stat_dim * i
            step_particles = good_particles[:, start_idx:start_idx + stat_dim]
            step_norms = np.sum(np.power(step_particles, 2), axis=1)
            keep_alpha = alpha/(1 - alpha_spend[:i].sum())
            step_bound = np.quantile(step_norms, 1 - keep_alpha)
            boundaries.append(step_bound)
            good_particles = good_particles[step_norms > step_bound]
        return np.array(boundaries)
