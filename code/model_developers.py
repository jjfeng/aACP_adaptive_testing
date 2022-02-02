import logging
from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.base

from models import *
from dataset import Dataset, DataGenerator
from hypothesis_tester import get_log_lik

class TestHistory:
    """
    Tracks the history of test results
    """
    def __init__(self, curr_mdl, res_detail):
        self.approval_times = [0]
        self.approved_mdls = [deepcopy(curr_mdl)]
        self.proposed_mdls = [deepcopy(curr_mdl)]
        self.curr_time = 0
        self.batch_numbers = []
        self.res_details = [res_detail]

    def update(self, test_res: int, res_detail: pd.DataFrame, proposed_mdl, batch_number: int = None):
        """
        @param test_res: 1 if we rejected the null, 0 if we failed to reject null
        @param res_detail: pd.DataFrame with one column for each performance measure that is being tracked
        @param proposed_mdl: the model that was proposed (but maybe not approved)
        """
        self.curr_time += 1
        proposed_mdl = deepcopy(proposed_mdl)
        if test_res == 1:
            self.approval_times.append(self.curr_time)
            self.approved_mdls.append(proposed_mdl)

        self.proposed_mdls.append(proposed_mdl)
        self.res_details.append(res_detail)
        self.batch_numbers.append(batch_number)

    @property
    def tot_approves(self):
        return len(self.approval_times)

    def get_perf_hist(self):
        perf_hist = pd.concat(self.res_details).reset_index()
        col_names = list(perf_hist.columns)
        value_vars = col_names[1:]
        perf_hist["time"] = np.arange(len(self.res_details))
        return pd.melt(perf_hist, id_vars=['time'], value_vars=value_vars)

def set_model(mdl, params):
    mdl.classes_ = np.array([0, 1])
    mdl.coef_ = params[1:].reshape((1, -1))
    mdl.intercept_ = np.array([params[0]])

class LockedModeler:
    """
    This modeler does not suggest any new model
    """
    def __init__(self, model_type:str = "Logistic", seed:int = 0):
        if model_type == "Logistic":
            self.modeler = MyLogisticRegression(penalty="none")
        elif model_type == "SelectiveLogistic":
            self.modeler = SelectiveLogisticRegression(penalty="none", target_acc=0.7)
        else:
            raise NotImplementedError("model type missing")

    def predict_prob(self, x):
        return self.modeler.predict_proba(x)[:, 1].reshape((-1, 1))

class BinaryAdversaryModeler(LockedModeler):
    """
    Given binary outputs, this adaptive modeler will try to propose modifications that are deleterious
    """
    update_dirs = [1,-1]

    def __init__(
            self, data_gen: DataGenerator, update_incr: float = 0.4, ni_margin: float = 0.01
    ):
        """
        @param update_incr: how much to perturb the coefficients
        """
        self.modeler = MyLogisticRegression()
        self.data_gen = data_gen
        self.num_sparse_theta = np.max(np.where(self.data_gen.beta.flatten() != 0)[0]) + 1
        self.update_incr = update_incr
        self.ni_margin = ni_margin

    def _get_sensitivity_specificity(self, mdl, test_size: int = 5000):
        dataset, _ = self.data_gen.generate_data(0,0,0,0,test_size,0)
        test_dat = dataset.test_dat
        pred_class = mdl.predict(test_dat.x)
        test_y = test_dat.y.flatten()
        acc = pred_class == test_y
        sensitivity = np.sum(acc * test_y)/np.sum(test_y)
        specificity = np.sum(acc * (1 - test_y))/np.sum(1 - test_y)
        #print("SENSE", sensitivity, "SPEC", specificity)
        return sensitivity, specificity

    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream=None, maxfev=10):
        """
        @param dat_stream: ignores this
        """
        # Train a good initial model
        self.modeler.fit(dat.x, dat.y.flatten())
        self.modeler.coef_[:] = self.data_gen.beta.flatten()
        self.modeler.intercept_[:] = 0
        orig_coefs = self.modeler.coef_[:]
        sens_curr, spec_curr = self._get_sensitivity_specificity(self.modeler)
        logging.info("orig %.3f %.3f", sens_curr, spec_curr)
        sens_test = sens_curr
        spec_test = spec_curr

        # Also have some predefined perturber for reference
        # just so we can use the parallel procedure
        self.predef_modeler = sklearn.base.clone(self.modeler)
        self.predef_modeler.fit(dat.x, dat.y.flatten())
        self.predef_modeler.coef_[:] = self.data_gen.beta.flatten()
        self.predef_modeler.intercept_[:] = 0

        # Now search in each direction and do a greedy search
        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
                "sensitivity_curr": [sens_curr],
                "specificity_curr": [spec_curr],
                }))
        while test_hist.curr_time < maxfev:
            # Test each coef (dont perturb intercept)
            for var_idx in range(self.num_sparse_theta, self.num_sparse_theta + dat.x.shape[1]):
                # Test update for the variable
                for update_dir in self.update_dirs:
                    test_res = 1
                    scale_factor = 1
                    while test_res == 1:
                        if test_hist.curr_time >= maxfev:
                            break
                        # Generate adaptive modification
                        curr_coef = np.concatenate(
                            [self.modeler.intercept_, self.modeler.coef_.flatten()]
                        )
                        #print("prev", curr_coef)
                        curr_coef[var_idx] += update_dir * self.update_incr * scale_factor
                        proposed_mdl = sklearn.base.clone(self.modeler)
                        set_model(proposed_mdl, curr_coef)
                        actual_sens_test, actual_spec_test = self._get_sensitivity_specificity(proposed_mdl)
                        print("PERTUB", var_idx, "ACTUAL TEST", actual_sens_test, actual_spec_test)
                        logging.info("coef prop %s", proposed_mdl.coef_[:])
                        logging.info("proposal %.3f %.3f", actual_sens_test, actual_spec_test)
                        #print("curr", var_idx, curr_coef)

                        # Generate predefined model
                        self.predef_modeler.coef_[0, :] = orig_coefs
                        predef_coef_idx = self.num_sparse_theta + (test_hist.curr_time // len(self.update_dirs))
                        predef_update_dir = test_hist.curr_time % len(self.update_dirs)
                        self.predef_modeler.coef_[0, predef_coef_idx] += (
                            self.update_dirs[predef_update_dir] * self.update_incr
                        )
                        print(self.predef_modeler.coef_)

                        # Test the performance
                        null_constraints = np.array([
                                [0, sens_test],
                                [0, spec_test]])
                        test_res = mtp_mechanism.get_test_res(
                            null_constraints, orig_mdl, proposed_mdl, predef_mdl=self.predef_modeler
                        )
                        print("perturb?", test_hist.curr_time, var_idx, update_dir, test_res)
                        if test_res:
                            print("TEST RES")
                            sens_curr = sens_test
                            spec_curr = spec_test
                            sens_test += self.ni_margin
                            spec_test += self.ni_margin
                            set_model(self.modeler, curr_coef)
                            logging.info("APPROVED %s", curr_coef)
                            # If we found a good direction, keep walking in that direction,
                            # be twice as aggressive
                            scale_factor *= 2

                        test_hist.update(
                            test_res=test_res,
                            res_detail=pd.DataFrame({
                                "sensitivity_curr": [sens_curr],
                                "specificity_curr": [spec_curr],
                                }),
                            proposed_mdl=proposed_mdl,
                        )
                    if test_res == 0 and scale_factor > 1:
                        break

        return test_hist

class OnlineAdaptNLLModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    """
    def __init__(self, model_type:str = "Logistic", seed:int = 0, validation_frac: float = 0.2, min_valid_dat_size: int = 200, power: float = 0.6, ni_margin: float = 0.02, lag_weight: float = 0.2, predef_alpha: float = 0.1):
        """
        @param lag_weight: used in the predefined model sequence to define what threshold we use to decide whether or not to test a particular model
        """
        assert model_type == "LogisticNLL"
        self.modeler = MyLogisticRegression(penalty="none")
        self.validation_frac = validation_frac
        self.min_valid_dat_size = min_valid_dat_size
        self.ni_margin = ni_margin
        self.power = power
        self.predef_alpha = predef_alpha

    def _do_power_calc_test_bound(self, orig_mdl, new_mdl, min_log_lik:float, valid_dat: Dataset, alpha: float, num_test: int, num_reps: int = 10000, se_factor: float = 1):
        """
        @param valid_dat: data for evaluating performance of model
        @param alpha: the type I error of the current test node
        """
        logging.info("predef alpha %f", alpha)
        # use valid_dat to evaluate the model first
        orig_pred_y = orig_mdl.predict_proba(valid_dat.x)[:,1]
        new_pred_y = new_mdl.predict_proba(valid_dat.x)[:,1]
        test_y = valid_dat.y.flatten()
        acc_diff = get_log_lik(test_y, new_pred_y) - get_log_lik(test_y, orig_pred_y)
        # Estimate the performance, but rather than using the estimate, use a slightly lower estimate
        mu_sim_raw = np.mean(acc_diff)
        var_sim = np.var(acc_diff)
        mu_sim = mu_sim_raw - np.sqrt(var_sim/valid_dat.size) * se_factor
        logging.info("power calc: MU SIM lower %s", mu_sim_raw)

        # for fun -- auc check?
        logging.info("fun: log lik orig %f", get_log_lik(test_y, orig_pred_y).mean())
        logging.info("fun: log lik new %f", get_log_lik(test_y, new_pred_y).mean())

        if mu_sim < 0:
            return 0, mu_sim

        candidate_log_lik = np.arange(min_log_lik, mu_sim, self.ni_margin/4)
        if candidate_log_lik.size == 0:
            logging.info("abort: no candidates found %f %f", min_log_lik, mu_sim)
            return 0, mu_sim

        obs_sim = np.random.normal(loc=mu_sim, scale=np.sqrt(var_sim), size=(num_test, num_reps))
        res = scipy.stats.ttest_1samp(obs_sim, popmean=candidate_log_lik.reshape((-1,1)))
        candidate_power = np.mean(res.statistic > scipy.stats.norm.ppf(1 - alpha), axis=1)

        if np.any(candidate_power > self.power):
            selected_idx = np.max(np.where(candidate_power > self.power)[0])
        else:
            logging.info("abort: power too low")
            selected_idx = np.argmax(candidate_power)

        selected_thres = candidate_log_lik[selected_idx]
        test_power = candidate_power[selected_idx]
        return test_power, selected_thres


    def _create_train_valid_dat(self, dat: Dataset):
        valid_n = max(self.min_valid_dat_size, int(dat.size * self.validation_frac))
        train_dat = dat.subset(dat.size - valid_n)
        print("valid_n", valid_n, train_dat.size)
        valid_dat = dat.subset(start_n=dat.size - valid_n, n=dat.size)
        return train_dat, valid_dat

    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev=10, side_dat_stream = None):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        train_dat, valid_dat = self._create_train_valid_dat(dat)
        self.modeler.fit(train_dat.x, train_dat.y.flatten())
        orig_mdl = self.modeler

        curr_log_lik = 0
        test_hist = TestHistory(orig_mdl, res_detail=pd.DataFrame({
                "log_lik_curr": [0],
                }))
        test_idx = 0
        adapt_read_idx = 0
        predef_test_mdls = []
        prior_predef_log_lik = 0
        while (test_idx < maxfev) and (adapt_read_idx < len(dat_stream)):
            print("ITERATION", test_idx)

            predef_dat = Dataset.merge([dat] + dat_stream[: adapt_read_idx + 1])
            predef_train_dat, predef_valid_dat = self._create_train_valid_dat(predef_dat)
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten())

            # calculate the threshold that we can test at such that the power of rejecting the null given Type I error at level alpha_node
            predef_test_power, predef_test_log_lik = self._do_power_calc_test_bound(
                    orig_mdl,
                    predef_lr,
                    min_log_lik=prior_predef_log_lik,
                    valid_dat=predef_valid_dat,
                    num_test=mtp_mechanism.test_set_size,
                    alpha=self.predef_alpha)
            do_predef_test = predef_test_power >= self.power

            logging.info("predef batch %d power %.5f", adapt_read_idx, predef_test_power)
            if do_predef_test:
                # Predef will not test if power is terrible
                predef_test_mdls.append(predef_lr)
                logging.info("predef test nll %.2f", predef_test_log_lik)
                logging.info("predef TEST idx %d, adapt idx %d, batch %d", len(predef_test_mdls) - 1, test_idx, adapt_read_idx)

            adapt_read_idx += 1
            #if (predef_test_log_lik + curr_log_lik)/2 > (curr_log_lik + self.ni_margin):
            if predef_test_log_lik > (curr_log_lik + self.ni_margin):
                nll_test = predef_test_log_lik # + curr_log_lik)/2
                logging.info("TEST idx: %d (batch_number) %d", test_idx, adapt_read_idx)
                logging.info("TEST (avg) nll %f", nll_test)

                null_constraints = np.array([
                        [0,nll_test]])
                test_res = mtp_mechanism.get_test_res(
                    null_constraints, orig_mdl, predef_lr, predef_mdl=predef_test_mdls[test_idx] if mtp_mechanism.require_predef else None
                )
                if test_res:
                    curr_log_lik = nll_test
                test_idx += 1
                logging.info("Test res %d", test_res)
                print("TEST RES", test_res)

                test_hist.update(
                        test_res=test_res,
                        res_detail = pd.DataFrame({
                            "log_lik_curr": [curr_log_lik]}),
                        proposed_mdl=predef_lr,
                        batch_number=adapt_read_idx,
                    )
            else:
                logging.info("CONTinuing to pull data until confident in NLL improvement")
        logging.info("adapt read idx %d", adapt_read_idx)
        print("adapt read", adapt_read_idx)
        logging.info("TEST batch numbers %s (len %d)", test_hist.batch_numbers, len(test_hist.batch_numbers))

        return test_hist

class OnlineAdaptLossModeler(OnlineAdaptNLLModeler):
    """
    Just do online learning on a separate dataset
    """
    def __init__(self, hypo_tester, validation_frac: float = 0.2, min_valid_dat_size: int = 200, power: float = 0.7, ni_margin: float = 0.02, predef_alpha: float = 0.1, se_factor: float = 1.96):
        """
        """
        self.modeler = MyLogisticRegression(penalty="l2")
        self.hypo_tester = hypo_tester
        self.validation_frac = validation_frac
        self.min_valid_dat_size = min_valid_dat_size
        self.ni_margin = ni_margin
        self.power = power
        self.predef_alpha = predef_alpha
        self.se_factor = se_factor

    def _do_power_calc_test_bound(self, orig_mdl, new_mdl, min_diff:float, valid_dat: Dataset, alpha: float, num_test: int, num_reps: int = 10000):
        """
        @param valid_dat: data for evaluating performance of model
        @param alpha: the type I error of the current test node
        """
        logging.info("predef alpha %f", alpha)
        # use valid_dat to evaluate the model first
        self.hypo_tester.test_dat = valid_dat
        res_df = self.hypo_tester.get_observations(orig_mdl, new_mdl).to_numpy().flatten()
        mu_sim_raw = np.mean(res_df)
        var_sim = np.var(res_df)
        mu_sim = mu_sim_raw - np.sqrt(var_sim/valid_dat.size) * self.se_factor
        logging.info("power calc: MU SIM lower %s", mu_sim_raw)

        if mu_sim < 0:
            return 0, mu_sim

        candidate_diffs = np.arange(min_diff, mu_sim, self.ni_margin/4)
        if candidate_diffs.size == 0:
            logging.info("abort: no candidates found %f %f", min_diff, mu_sim)
            return 0, mu_sim

        obs_sim = np.random.normal(loc=mu_sim, scale=np.sqrt(var_sim), size=(num_test, num_reps))
        res = scipy.stats.ttest_1samp(obs_sim, popmean=candidate_diffs.reshape((-1,1)))
        candidate_power = np.mean(res.statistic > scipy.stats.norm.ppf(1 - alpha), axis=1)

        if np.any(candidate_power > self.power):
            selected_idx = np.max(np.where(candidate_power > self.power)[0])
        else:
            logging.info("abort: power too low")
            selected_idx = np.argmax(candidate_power)

        selected_thres = candidate_diffs[selected_idx]
        test_power = candidate_power[selected_idx]
        return test_power, selected_thres


    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev=10, side_dat_stream = None):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        train_dat, valid_dat = self._create_train_valid_dat(dat)
        self.modeler.fit(train_dat.x, train_dat.y.flatten())
        orig_mdl = self.modeler

        curr_diff= 0
        test_hist = TestHistory(orig_mdl, res_detail=pd.DataFrame({
                "curr_diff": [0],
                }))
        test_idx = 0
        adapt_read_idx = 0
        predef_test_mdls = []
        prior_predef_diff = 0
        while (test_idx < maxfev) and (adapt_read_idx < len(dat_stream)):
            print("ITERATION", test_idx)

            predef_dat = Dataset.merge([dat] + dat_stream[: adapt_read_idx + 1])
            predef_train_dat, predef_valid_dat = self._create_train_valid_dat(predef_dat)
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten())

            # calculate the threshold that we can test at such that the power of rejecting the null given Type I error at level alpha_node
            predef_test_power, predef_test_diff = self._do_power_calc_test_bound(
                    orig_mdl,
                    predef_lr,
                    min_diff=prior_predef_diff,
                    valid_dat=predef_valid_dat,
                    num_test=mtp_mechanism.test_set_size,
                    alpha=self.predef_alpha)
            do_predef_test = predef_test_power >= self.power

            logging.info("predef batch %d power %.5f", adapt_read_idx, predef_test_power)
            if do_predef_test:
                # Predef will not test if power is terrible
                predef_test_mdls.append(predef_lr)
                logging.info("predef test %.2f", predef_test_diff)
                logging.info("predef TEST idx %d, adapt idx %d, batch %d", len(predef_test_mdls) - 1, test_idx, adapt_read_idx)

            adapt_read_idx += 1
            if predef_test_diff > (curr_diff + self.ni_margin):
                logging.info("TEST idx: %d (batch_number) %d", test_idx, adapt_read_idx)
                logging.info("TEST (avg) diff %f", predef_test_diff)

                null_constraints = np.array([
                        [0,predef_test_diff]])
                test_res = mtp_mechanism.get_test_res(
                    null_constraints, orig_mdl, predef_lr, predef_mdl=predef_test_mdls[test_idx] if mtp_mechanism.require_predef else None
                )
                if test_res:
                    curr_diff = predef_test_diff
                test_idx += 1
                logging.info("Test res %d", test_res)
                print("TEST RES", test_res)

                test_hist.update(
                        test_res=test_res,
                        res_detail = pd.DataFrame({
                            "curr_diff": [curr_diff]}),
                        proposed_mdl=predef_lr,
                        batch_number=adapt_read_idx,
                    )
            else:
                logging.info("CONTinuing to pull data until confident in NLL improvement")
        logging.info("adapt read idx %d", adapt_read_idx)
        print("adapt read", adapt_read_idx)
        logging.info("TEST batch numbers %s (len %d)", test_hist.batch_numbers, len(test_hist.batch_numbers))

        return test_hist


#class OnlineAdaptSensSpecModeler(LockedModeler):
#    """
#    Just do online learning on a separate dataset
#    """
#    def __init__(self, model_type:str = "Logistic", seed:int = 0, ni_margin: float = 0.02, validation_frac: float = 0.2, min_valid_dat_size: int = 200, power: float = 0.5):
#        assert model_type == "Logistic"
#        self.modeler = RevisedLogisticRegression(penalty="none")
#        self.ni_margin = ni_margin
#        self.validation_frac = validation_frac
#        self.min_valid_dat_size = min_valid_dat_size
#        self.power = power
#
#    def _get_sensitivity_specificity_lower_bound_diff(self, orig_mdl, new_mdl, valid_dat: Dataset, se_factor: float = 1.96):
#        orig_pred_class = orig_mdl.predict(valid_dat.x)
#        new_pred_class = new_mdl.predict(valid_dat.x)
#        test_y = valid_dat.y.flatten()
#        acc_diff = (new_pred_class == test_y).astype(int) - (orig_pred_class == test_y).astype(int)
#        print("new", (new_pred_class == test_y).astype(int).mean())
#        print("ORIG", (orig_pred_class == test_y).astype(int).mean())
#        sensitivity = np.sum(acc_diff * test_y)/np.sum(test_y)
#        specificity = np.sum(acc_diff * (1 - test_y))/np.sum(1 - test_y)
#        sensitivity_se = np.sqrt(np.var(acc_diff[test_y == 1])/np.sum(test_y))
#        specificity_se = np.sqrt(np.var(acc_diff[test_y == 0])/np.sum(1 - test_y))
#        print("sens", sensitivity, "spec", specificity, "sens-se", sensitivity_se, "spec-se", specificity_se)
#        return sensitivity - se_factor * sensitivity_se, specificity - se_factor * specificity_se
#
#    def _do_power_calc_test_bound(self, orig_mdl, new_mdl, min_sens:float, valid_dat: Dataset, alpha: float, num_test: int, num_reps: int = 10000, se_factor: float = 1.5):
#        """
#        @param valid_dat: data for evaluating performance of model
#        @param alpha: the type I error of the current test node
#        """
#        logging.info("predef alpha %f", alpha)
#        # use valid_dat to evaluate the model first
#        orig_pred_class = orig_mdl.predict(valid_dat.x)
#        new_pred_class = new_mdl.predict(valid_dat.x)
#        test_y = valid_dat.y.flatten()
#        acc_diff = (new_pred_class == test_y).astype(int) - (orig_pred_class == test_y).astype(int)
#        # Estimate the performance, but rather than using the estimate, use a slightly lower estimate
#        mu_sim_raw = np.array([
#            np.sum(acc_diff * test_y)/np.sum(test_y),
#            np.sum(acc_diff * (1 - test_y))/np.sum(1 - test_y)])
#        var_sim = np.array([np.var(acc_diff[test_y == 1]), np.var(acc_diff[test_y == 0])])
#        mu_sim = mu_sim_raw - np.sqrt(var_sim/valid_dat.size) * se_factor
#        logging.info("power calc: MU SIM lower %s", mu_sim_raw)
#
#        # for fun -- auc check?
#        logging.info("fun: accuracy orig %f", np.mean(test_y == orig_pred_class))
#        logging.info("fun: accuracy new %f", np.mean(test_y == new_pred_class))
#        orig_pred_y = orig_mdl.predict_proba(valid_dat.x)[:,1]
#        new_pred_y = new_mdl.predict_proba(valid_dat.x)[:,1]
#        logging.info("fun: log loss orig %f", sklearn.metrics.log_loss(test_y, orig_pred_y))
#        logging.info("fun: log loss new %f", sklearn.metrics.log_loss(test_y, new_pred_y))
#        logging.info("fun: auc orig %f", sklearn.metrics.roc_auc_score(test_y, orig_pred_y))
#        logging.info("fun: auc new %f", sklearn.metrics.roc_auc_score(test_y, new_pred_y))
#        logging.info("fun: sens spec diff %s", mu_sim)
#
#
#        if mu_sim[1] < self.ni_margin:
#            return 0, None
#
#        # Now calculate power for specificity at NI margin
#        obs_sim = np.random.normal(loc=mu_sim[1], scale=np.sqrt(var_sim[1]), size=(num_test, num_reps))
#        res1 = scipy.stats.ttest_1samp(obs_sim, popmean=-self.ni_margin, alternative="greater")
#        print("spec NI margin POWER", np.mean(res1.pvalue < alpha))
#        spec_power = np.mean(res1.pvalue < alpha)
#
#        # Now calculate power for sensitivity for candidate values
#        candidate_sens = np.arange(min_sens, mu_sim[0], self.ni_margin/4)
#        if candidate_sens.size == 0:
#            return 0, None
#
#        obs_sim = np.random.normal(loc=mu_sim[:1], scale=np.sqrt(var_sim[0]), size=(num_test, num_reps))
#        res0 = scipy.stats.ttest_1samp(obs_sim, popmean=candidate_sens.reshape((-1,1)), alternative="greater")
#        candidate_sens_power = np.mean(res0.pvalue < alpha, axis=1)
#
#        if np.any(candidate_sens_power > self.power):
#            selected_idx = np.max(np.where(candidate_sens_power > self.power)[0])
#            selected_sens_thres = candidate_sens[selected_idx]
#            sens_power = candidate_sens_power[selected_idx]
#            return min(sens_power, spec_power), selected_sens_thres
#        else:
#            return min(np.max(candidate_sens_power), spec_power), None
#
#
#    def _create_train_valid_dat(self, dat: Dataset):
#        #shuffle_idxs = np.random.choice(dat.size, dat.size, replace=False)
#        valid_n = max(self.min_valid_dat_size, int(dat.size * self.validation_frac))
#        train_dat = dat.subset(dat.size - valid_n)
#        print("valid_n", valid_n, train_dat.size)
#        valid_dat = dat.subset(start_n=dat.size - valid_n, n=dat.size)
#        return train_dat, valid_dat
#
#    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev=10, side_dat_stream = None):
#        """
#        @param dat_stream: a list of datasets for further training the model
#        @return perf_value
#        """
#        # TODO: clean up this
#        predef_mtp_mechanism = deepcopy(mtp_mechanism)
#
#        train_dat, valid_dat = self._create_train_valid_dat(dat)
#        self.modeler.fit_orig_mdl(train_dat.x, train_dat.y.flatten())
#        orig_mdl = self.modeler.orig_mdl
#
#        curr_idx = 0
#        curr_sens = 0
#        curr_spec =  -self.ni_margin
#        test_hist = TestHistory(orig_mdl, res_detail=pd.DataFrame({
#                "sensitivity_curr": [0],
#                "specificity_curr": [0],
#                }))
#        test_idx = 0
#        predef_test_idx = 0
#        adapt_read_idx = 0
#        predef_test_mdls = []
#        prior_predef_sens = 0
#        while (test_idx < maxfev) and (adapt_read_idx < len(dat_stream)):
#            print("ITERATION", test_idx)
#
#            predef_dat = Dataset.merge([dat] + dat_stream[: adapt_read_idx + 1])
#            predef_train_dat, predef_valid_dat = self._create_train_valid_dat(predef_dat)
#            predef_lr = sklearn.base.clone(self.modeler)
#            # TODO: we may need to update training procedure to optimize acceptance rate, update
#            # spec_buffer to get a desired specificity rate
#            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten(), orig_mdl)
#
#            # calculate the threshold that we can test at such that the power of rejecting the null given Type I error at level alpha_node
#            predef_test_power, predef_test_sens = self._do_power_calc_test_bound(
#                    orig_mdl,
#                    predef_lr,
#                    min_sens=prior_predef_sens + self.ni_margin/4,
#                    valid_dat=predef_valid_dat,
#                    num_test=predef_mtp_mechanism.hypo_tester.test_dat.size,
#                    alpha=0.1)
#            do_predef_test = predef_test_power >= self.power
#
#            new_sens, new_spec = self._get_sensitivity_specificity_lower_bound_diff(orig_mdl, predef_lr, predef_valid_dat)
#            print("PREDEF SENSE", new_sens, prior_predef_sens)
#            logging.info("predef batch %d power %.5f", adapt_read_idx, predef_test_power)
#            if do_predef_test:
#                # Predef will not test if sensitivity or specificity estimates are bad
#                predef_test_mdls.append(predef_lr)
#                predef_test_idx += 1
#                prior_predef_sens = predef_test_sens
#                print("PREDEF ETST", len(predef_test_mdls), test_idx, new_sens)
#                logging.info("predef test sens %.2f", predef_test_sens)
#                logging.info("predef TEST idx %d, adapt idx %d, batch %d", len(predef_test_mdls) - 1, test_idx, adapt_read_idx)
#                # TODO: Predef assumes all rejects of the null. do tree update???
#                #predef_mtp_mechanism._do_tree_update(1)
#
#
#            logging.info("SENStivity diff %.3f %.3f", curr_sens, new_sens)
#            logging.info("Specificity diff %.3f %.3f", curr_spec, new_spec)
#
#            adapt_read_idx += 1
#            if do_predef_test and ((predef_test_sens + curr_sens)/2 > (curr_sens + self.ni_margin/4)):
#                sens_test = (predef_test_sens + curr_sens)/2
#                spec_test = curr_spec
#                logging.info("TEST idx: %d (batch_number) %d", test_idx, adapt_read_idx)
#                logging.info("TEST (avg) sens %f spec %f", sens_test, spec_test)
#
#                # TODO: this should be defined adaptively
#                null_constraints = np.array([
#                        [0,sens_test],
#                        [0,spec_test]])
#                test_res = mtp_mechanism.get_test_res(
#                    null_constraints, orig_mdl, predef_lr, predef_mdl=predef_test_mdls[test_idx]
#                )
#                assert sens_test > 0
#                if test_res:
#                    curr_sens = sens_test
#                test_idx += 1
#                logging.info("Test res %d", test_res)
#                print("TEST RES", test_res)
#
#                test_hist.update(
#                        test_res=test_res,
#                        res_detail = pd.DataFrame({
#                            "sensitivity_curr": [curr_sens],
#                            "specificity_curr": [curr_spec]}),
#                        proposed_mdl=predef_lr,
#                        batch_number=adapt_read_idx,
#                    )
#            else:
#                logging.info("CONTinuing to pull data until confident in sensitivity improvement")
#        logging.info("adapt read idx %d", adapt_read_idx)
#        print("adapt read", adapt_read_idx)
#        logging.info("TEST batch numbers %s", test_hist.batch_numbers)
#
#        return test_hist
#
#class OnlineSensSpecModeler(OnlineAdaptSensSpecModeler):
#    """
#    adaptive online testing
#    """
#    def __init__(self, model_type:str = "Logistic", seed:int = 0, ni_margin: float = 1e-10, validation_frac: float = 0.2, min_valid_dat_size: int = 200, countdown_reset: int = 4):
#        super(OnlineSensSpecModeler,self).__init__(model_type, seed, ni_margin, validation_frac, min_valid_dat_size)
#        self.countdown_reset = countdown_reset
#
#    def _create_train_valid_dat(self, orig_dat: Dataset, new_dat: Dataset = None):
#        num_obs = orig_dat.size + new_dat.size if new_dat is not None else orig_dat.size
#        shuffle_idxs = np.flip(np.arange(num_obs))
#        valid_n = max(min(orig_dat.size, int(num_obs * self.validation_frac)), self.min_valid_dat_size)
#        print("valid_n", valid_n)
#        merge_dat = Dataset.merge([orig_dat, new_dat]) if new_dat is not None else orig_dat
#        train_dat = merge_dat.subset_idxs(shuffle_idxs[valid_n:])
#        valid_dat = merge_dat.subset_idxs(shuffle_idxs[:valid_n])
#        return train_dat, valid_dat
#
#    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev: int, side_dat_stream: Dataset):
#        """
#        @param dat_stream: a list of datasets for further training the model
#        @return perf_value
#        """
#        train_dat, valid_dat = self._create_train_valid_dat(dat)
#        self.modeler.fit(train_dat.x, train_dat.y.flatten())
#        orig_mdl = self.modeler
#
#        adapt_dat = []
#        countdown = self.countdown_reset
#        read_side_stream = False
#        curr_idx = 0
#        curr_sens = 0
#        curr_spec = 0
#        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
#                "sensitivity_curr": [curr_sens],
#                "specificity_curr": [curr_spec],
#                }))
#
#        for i in range(maxfev):
#            print("ITERATION", i)
#
#            predef_train_dat, _ = self._create_train_valid_dat(dat, Dataset.merge(dat_stream[:i + 1]))
#            predef_lr = sklearn.base.clone(self.modeler)
#            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten())
#
#            if read_side_stream:
#                batches_read = side_dat_stream[i: i + 1]
#            else:
#                batches_read = dat_stream[i : i + 1]
#            logging.info("BATCH countdown %d", countdown)
#            logging.info("BATCH idx %d", read_side_stream)
#
#            adapt_dat = adapt_dat + batches_read
#            adapt_train_dat, adapt_valid_dat = self._create_train_valid_dat(
#                    dat,
#                    Dataset.merge(adapt_dat))
#            print("ADAPT SIDE", adapt_valid_dat.size)
#            adapt_lr = sklearn.base.clone(self.modeler)
#            adapt_lr.fit(adapt_train_dat.x, adapt_train_dat.y.flatten())
#
#            new_sens, new_spec = self._get_sensitivity_specificity_lower_bound_diff(orig_mdl, adapt_lr, adapt_valid_dat)
#            print(curr_sens, new_sens)
#            print(curr_spec, new_spec)
#            #assert ((curr_sens + new_sens)/2 > curr_sens) or ((curr_spec + new_spec)/2 > curr_spec)
#            sens_test = max(curr_sens, (curr_sens + new_sens)/2)
#            spec_test = max(curr_spec, (curr_spec + new_spec)/2)
#            print("NEW", sens_test, spec_test)
#            logging.info("sens spec test %.3f %.3f", sens_test, spec_test)
#
#            # TODO: this should be defined adaptively
#            null_constraints = np.array([
#                    [0,sens_test],
#                    [0,spec_test]])
#            test_res = mtp_mechanism.get_test_res(
#                null_constraints, orig_mdl, adapt_lr, predef_mdl=predef_lr
#            )
#            logging.info("test res %d", test_res)
#            if test_res:
#                curr_sens = sens_test
#                curr_spec = spec_test
#                countdown = self.countdown_reset
#            else:
#                countdown -= 1
#                if countdown == 0:
#                    read_side_stream = not read_side_stream
#                    countdown = self.countdown_reset
#
#            test_hist.update(
#                    test_res=test_res,
#                    res_detail = pd.DataFrame({
#                        "sensitivity_curr": [curr_sens],
#                        "specificity_curr": [curr_spec]}),
#                    proposed_mdl=adapt_lr)
#        return test_hist
#
#
#class OnlineFixedSelectiveModeler(LockedModeler):
#    """
#    Just do online learning on a separate dataset
#    """
#    def __init__(self, model_type:str = "SelectiveLogistic", seed:int = 0, incr_accept: float = 0.02, init_accept= 0.6, target_acc: float = 0.85, acc_buffer: float = 0.05):
#        assert model_type == "SelectiveLogistic"
#        self.modeler = SelectiveLogisticRegression(penalty="none", target_acc=target_acc + acc_buffer)
#        self.incr_accept = incr_accept
#        self.curr_accept = init_accept
#        self.accept_test = init_accept + incr_accept
#        self.accuracy_test = target_acc
#
#    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev=10):
#        """
#        @param dat_stream: a list of datasets for further training the model
#        @return perf_value
#        """
#        self.modeler.fit(dat.x, dat.y.flatten())
#
#        predef_dat = dat
#        curr_idx = 0
#        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
#                "accept_curr": [self.curr_accept],
#                }))
#        for i in range(maxfev):
#            print("ITERATION", i)
#
#            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + 1])
#            predef_lr = sklearn.base.clone(self.modeler)
#            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())
#
#            null_constraints = np.array([
#                    [0,self.accept_test],
#                    [0,self.accuracy_test]])
#            test_res = mtp_mechanism.get_test_res(
#                null_constraints, predef_lr, predef_mdl=predef_lr
#            )
#            if test_res:
#                self.curr_accept = self.accept_test
#                self.accept_test += self.incr_accept
#
#            test_hist.update(
#                    test_res=test_res,
#                    res_detail = pd.DataFrame({
#                        "accept_curr": [self.curr_accept]}),
#                    proposed_mdl=predef_lr)
#        return test_hist
