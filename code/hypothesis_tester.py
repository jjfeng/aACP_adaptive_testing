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

        #assert np.all(np.isfinite(score_y0))
        #assert np.all(np.isfinite(score_y1))

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
        logging.info("AUC %f", auc)
        #sklearn_auc = roc_auc_score(test_y, pred_y)
        #logging.info("AUC %f SKLEARN %f", auc, sklearn_auc)

        influence_func = mask_y1/prob_y1 * cdf_score_y0 + mask_y0/prob_y0 * cdf_score_y1 - (mask_y0/prob_y0 + mask_y1/prob_y1) * auc + auc

        #if np.abs(auc - sklearn_auc) > 0.2:
        #    raise ValueError("WEIRD AUC")

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
            boundary = scipy.stats.norm.ppf(1 - alpha_spend, scale=np.sqrt(cov_est[0,0]))
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

        test_res = (test_stat - null_constraint[0,1]) > boundary
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

class CalibHypothesisTester(AUCHypothesisTester):
    def get_influence_func(self, mdl):
        pred_y = mdl.predict_proba(self.test_dat.x)[:,1]
        test_y = self.test_dat.y.flatten()
        influence_func = (pred_y - test_y) * pred_y
        return influence_func, influence_func.mean(axis=0)

class CalibAUCHypothesisTester(AUCHypothesisTester):
    stats_dim = 2
    def __init__(self, calib_alloc_frac: float=0.1):
        """
        @param calib_alloc_frac: how much of the alpha to allocate to checking calibration
        """
        self.auc_hypo_tester = AUCHypothesisTester()
        self.calib_hypo_tester = CalibHypothesisTester()
        self.calib_alloc_frac = calib_alloc_frac

    def make_scratch(self, scratch_file: str):
        self.scratch_file_cov = scratch_file.replace(".txt", "_cov.txt")
        self.scratch_file_bounds = scratch_file.replace(".txt", "_bounds.txt")

    def set_test_dat(self, test_dat):
        self.test_dat = test_dat
        self.auc_hypo_tester.set_test_dat(test_dat)
        self.calib_hypo_tester.set_test_dat(test_dat)

    def get_influence_func(self, mdl):
        auc_ic, auc_diff = self.auc_hypo_tester.get_influence_func(mdl)
        calib_ic, calib = self.calib_hypo_tester.get_influence_func(mdl)

        influence_func = np.hstack([calib_ic.reshape((-1,1)), auc_ic.reshape((-1,1))])
        estimate = np.array([calib, auc_diff])

        return influence_func, estimate

    def _get_observations(self, orig_mdl, new_mdl):
        orig_ic, orig_est = self.get_influence_func(orig_mdl)
        new_ic, new_est = self.get_influence_func(new_mdl)
        df = pd.DataFrame(new_ic - orig_ic, columns=["calib_score_ic", "auc_diff_ic"])

        return df, orig_est, new_est

    def _get_boundary(self, prior_bounds, cov_est, alpha_spend: float, alt_greater: bool = False):
        #print("CALL BOUNDARY", prior_bounds, prior_bounds.size, alpha_spend)
        if prior_bounds.size == 0:
            boundary = scipy.stats.norm.ppf((1 - alpha_spend) if alt_greater else alpha_spend, scale=np.sqrt(cov_est[0,0]))
        else:
            np.savetxt(self.scratch_file_cov, cov_est, delimiter=",")
            np.savetxt(self.scratch_file_bounds, prior_bounds, delimiter=",")
            #print("ALPHA", alpha_spend)
            rcmd = "Rscript R/pmvnorm.R %s %s %f %d" % (self.scratch_file_cov, self.scratch_file_bounds, np.log10(alpha_spend), alt_greater)
            output = subprocess.check_output(
                rcmd,
                stderr=subprocess.STDOUT,
                shell=True,
                encoding='UTF-8'
            )
            boundary = float(output[4:])
            logging.info(rcmd)
            logging.info("Test bound %f, log alpha %f", boundary, np.log10(alpha_spend))
        return boundary

    def test_null(self, alpha: float, node: Node, null_constraint: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        estimate = node.obs.to_numpy().mean(axis=0)
        logging.info("test set estimate %s", estimate)
        print("ESTIMATE", node.obs, estimate)

        full_df = pd.concat([prior_node.obs for prior_node in prior_nodes] + [node.obs], axis=1).to_numpy().T
        print("FULL DF", full_df.shape)
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
        prior_bounds = np.array([prior_node.bounds for prior_node in prior_nodes]).reshape((-1,2))
        print("PRIOR BOUNDS", prior_bounds.shape)

        test_res = False
        num_particles =  np.sum(1/(alpha * node_weights))
        node_alpha_spend = alpha * node_weights[-1]
        alpha_spend = [
                node_alpha_spend * 0.5 * self.calib_alloc_frac, # alloted to calib upper
                node_alpha_spend * 0.5 * self.calib_alloc_frac, # alloted to calib upper
                node_alpha_spend * (1 - self.calib_alloc_frac) # alloted to auc
                ]
        print("node_wei", node_weights, alpha, alpha_spend)
        calib_lower_bound = self._get_boundary(prior_bounds, cov_est[:-1,:-1], alpha_spend[0], alt_greater=True)
        calib_upper_bound = self._get_boundary(prior_bounds, cov_est[:-1,:-1], alpha_spend[1], alt_greater=False)
        prior_bounds = np.vstack([prior_bounds, [calib_upper_bound, calib_lower_bound]])
        auc_lower_bound = self._get_boundary(prior_bounds, cov_est, alpha_spend[2], alt_greater=True)
        boundaries = np.array([
            [calib_upper_bound, calib_lower_bound],
            [-np.inf, auc_lower_bound]
            ])

        test_res1 = (estimate[0] - null_constraint[0,0]) > calib_lower_bound
        test_res2 = (estimate[0] - null_constraint[0,1]) < calib_upper_bound
        test_res3 = (estimate[1] - null_constraint[1,1]) > auc_lower_bound
        logging.info("estimate %s", estimate)
        logging.info("null_constraint %s", null_constraint)
        logging.info("boundaires %f %f %f", calib_lower_bound, calib_upper_bound, auc_lower_bound)
        test_res = test_res1 and test_res2 and test_res3
        logging.info("TEST REST %d %d %d", test_res1, test_res2, test_res3)
        logging.info("final TEST REST %d", test_res)
        return test_res, boundaries

