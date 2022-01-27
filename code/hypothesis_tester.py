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
        assert len(prior_nodes) == 0
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

        full_df = pd.concat([self.orig_obs, node.obs], axis=1).to_numpy().T
        raw_covariance = np.cov(full_df)/self.test_dat.size

        delta_grad = np.array([
            [-raw_estimates[2]/(raw_estimates[0]**2), 0],
            [0, -raw_estimates[3]/(raw_estimates[1]**2)],
            [1/raw_estimates[0], 0],
            [0, 1/raw_estimates[1]],
            ])
        cov_est = delta_grad.T @ raw_covariance @ delta_grad

        # check if null holds given the estimate
        precision_mat = np.linalg.inv(cov_est)
        def get_norm(test_pt):
            dist = (estimate - test_pt).reshape((-1,1))
            return (dist.T @ precision_mat @ dist)[0,0]

        # Need to check if it is within any of the specified bounds (but not necessarily both bounds)
        opt0_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
            (null_constraint[0,0], null_constraint[0,1]),
            (0,1)])
        opt1_res = scipy.optimize.minimize(get_norm, x0=null_constraint.mean(axis=1), bounds=[
            (0,1),
            (null_constraint[1,0], null_constraint[1,1])])

        chi2_df2 = scipy.stats.chi2(df=2)
        print(1 - chi2_df2.cdf(opt0_res.fun), 1 - chi2_df2.cdf(opt1_res.fun))
        pval = 1 - min(chi2_df2.cdf(opt0_res.fun), chi2_df2.cdf(opt1_res.fun))
        print("p-value", pval)
        return pval < (node.weight * alpha)
