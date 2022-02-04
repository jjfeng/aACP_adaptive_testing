"""
Classes for testing specific hypotheses
All hypotheses here evaluate candidate modifications using multiple performance metrics
"""
import logging
from typing import List
import subprocess

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

from node import Node

MAX_PARTICLES = 50000
MIN_PARTICLES = 5000

def get_log_lik(y_true, y_pred):
    y_pred = np.minimum(np.maximum(y_pred, 1e-10), 1 - 1e-10)
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)


class HypothesisTester:
    def make_scratch(self, scratch_file: str):
        self.scratch_file = scratch_file

    def get_auc(self, test_y, score_y):
        score_y0 = score_y[test_y == 0]
        score_y1 = score_y[test_y == 1]
        all_ranks = score_y0.reshape((1,-1)) < score_y1.reshape((-1,1))
        return np.mean(all_ranks)

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

class AUCHypothesisTester(HypothesisTester):
    def get_influence_func(self, mdl):
        pred_y = mdl.predict_log_proba(self.test_dat.x)[:,1]
        test_y = self.test_dat.y.flatten()
        prob_y1 = test_y.mean()
        prob_y0 = 1 - prob_y1
        mask_y0 = test_y == 0
        mask_y1 = test_y == 1
        cdf_score_y0 = np.array([np.mean(pred_y[mask_y0] < pred_y[i]) for i in range(self.test_dat.size)])
        cdf_score_y1 = np.array([np.mean(pred_y[mask_y1] > pred_y[i]) for i in range(self.test_dat.size)])
        auc = self.get_auc(test_y, pred_y)

        # Note that this can actually be quite different!
        sklearn_auc = roc_auc_score(test_y, pred_y)
        logging.info("AUC %f SKLEARN %f", auc, sklearn_auc)

        influence_func = mask_y1/prob_y1 * cdf_score_y0 + mask_y0/prob_y0 * cdf_score_y1 - (mask_y0/prob_y0 + mask_y1/prob_y1) * auc + auc

        if np.abs(auc - sklearn_auc) > 0.2:
            raise ValueError("WEIRD AUC")

        return influence_func, auc

    def _get_observations(self, orig_mdl, new_mdl):
        orig_auc_ic, orig_auc = self.get_influence_func(orig_mdl)
        new_auc_ic, new_auc = self.get_influence_func(new_mdl)
        df = pd.DataFrame({
            "auc_diff_ic": new_auc_ic - orig_auc_ic,
            })

        return df, orig_auc, new_auc

    def get_observations(self, orig_mdl, new_mdl):
        return self._get_observations(orig_mdl, new_mdl)[0]

    def test_null(self, alpha: float, node: Node, null_constraint: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        estimate = node.obs.to_numpy().mean()
        logging.info("test set estimate %.3f", estimate)

        full_df = pd.concat([prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
        cov_est = np.cov(full_df)/self.test_dat.size
        if full_df.shape[0] == 1:
            cov_est = np.array([[cov_est]])
        logging.info("cov est %s", cov_est)
        if np.any(np.isnan(cov_est)):
            print(full_df)
            raise ValueError("something wrong with cov")

        num_nodes = len(prior_nodes) + 1
        assert not np.any(np.isnan(cov_est))
        node_weights = np.array([prior_node.weight for prior_node in prior_nodes] + [node.weight])
        prior_bounds = np.array([prior_node.upper_bound for prior_node in prior_nodes])

        test_res = False
        num_particles =  np.sum(1/(alpha * node_weights))
        alpha_spend = alpha * node_weights[-1]
        test_stat = estimate
        if len(prior_nodes) == 0:
            boundary = scipy.stats.norm.ppf(1 - alpha_spend, loc=null_constraint[0,1], scale=np.sqrt(cov_est[0,0]))
            stat, pval = scipy.stats.ttest_1samp(node.obs.to_numpy().flatten(), popmean=null_constraint[0,1], alternative="greater")
            logging.info("tstat %f pval %f alpha %f", stat, pval, alpha_spend)
        else:
            np.savetxt(self.scratch_file, cov_est, delimiter=",")
            prior_bound_str = " ".join(map(str, prior_bounds))
            rcmd = "Rscript R/pmvnorm.R %s %f %s" % (self.scratch_file, np.log10(alpha_spend), prior_bound_str)
            output = subprocess.check_output(
                rcmd,
                stderr=subprocess.STDOUT,
                shell=True,
                encoding='UTF-8'
            )
            boundary = float(output[4:])
            logging.info(rcmd)
            logging.info("Test bound %f, est %f, log alpha %f, prior_bound_str %s", boundary, estimate, np.log10(alpha_spend), prior_bound_str)

        test_res = test_stat > boundary
        return test_res, boundary

class LogLikHypothesisTester(AUCHypothesisTester):
    def _get_observations(self, orig_mdl, new_mdl):
        orig_pred_y = orig_mdl.predict_proba(self.test_dat.x)[:,1]
        new_pred_y = new_mdl.predict_proba(self.test_dat.x)[:,1]
        test_y = self.test_dat.y.flatten()
        new_loglik = get_log_lik(test_y, new_pred_y)
        orig_loglik = get_log_lik(test_y, orig_pred_y)
        log_lik_diff = new_loglik - orig_loglik
        logging.info("orig ll %f new ll %f diff %f", orig_loglik.mean(), new_loglik.mean(), new_loglik.mean() - orig_loglik.mean())
        df = pd.DataFrame({
            "log_lik_diff": log_lik_diff,
            })

        return df, orig_loglik, new_loglik

   # def generate_spending_boundaries(
   #         self,
   #         cov,
   #         stat_dim: int,
   #         alpha_spend: np.ndarray,
   #         num_particles: int=5000,
   #         batch_size: int= 50000
   #         ):
   #     """
   #     Simulates particle paths for alpha spending
   #     Assumes the test at each iteration is H_0: theta_i < 0 for some i (for i in stat_dim)
   #     """
   #     boundaries = []
   #     good_particles = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=batch_size)
   #     for i, alpha in enumerate(alpha_spend):
   #         start_idx = stat_dim * i
   #         keep_alpha = alpha/(1 - alpha_spend[:i].sum())

   #         step_particles = good_particles[:, start_idx:start_idx + stat_dim]
   #         particle_mask = np.all(step_particles > 0, axis=1)
   #         step_norms = particle_mask * np.min(np.abs(step_particles), axis=1)
   #         step_bound = np.quantile(step_norms, 1 - keep_alpha)
   #         keep_ratio = np.mean(step_norms < step_bound)
   #         # if the keep ratio is not close to what we desired, do not rejecanything
   #         logging.info("keep ratio %f", (1 - keep_ratio)/keep_alpha)
   #         print("KEEP RATIO", keep_ratio, keep_alpha, (1 - keep_ratio)/keep_alpha)
   #         if keep_ratio < keep_alpha or (1 - keep_ratio)/keep_alpha > 2:
   #             print(np.max(step_norms), step_bound)
   #             # If the step bound is weird, do not reject anything
   #             step_bound = np.max(step_norms)
   #             # step_bound += 1
   #         boundaries.append(step_bound)
   #         good_particles = good_particles[step_norms < step_bound]
   #     return np.array(boundaries)

