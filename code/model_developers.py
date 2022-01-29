import logging
from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.base

from models import *
from dataset import Dataset, DataGenerator


class TestHistory:
    """
    Tracks the history of test results
    """
    def __init__(self, curr_mdl, res_detail):
        self.approval_times = [0]
        self.approved_mdls = [deepcopy(curr_mdl)]
        self.proposed_mdls = [deepcopy(curr_mdl)]
        self.curr_time = 0
        self.res_details = [res_detail]

    def update(self, test_res: int, res_detail: pd.DataFrame, proposed_mdl):
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
            self, data_gen: DataGenerator, update_incr: float = 0.4, incr_sens_spec: float = 0.01
    ):
        """
        @param update_incr: how much to perturb the coefficients
        """
        self.modeler = MyLogisticRegression()
        self.data_gen = data_gen
        self.update_incr = update_incr
        self.incr_sens_spec = incr_sens_spec

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

    def simulate_approval_process(self, dat, test_x, test_y, mtp_mechanism, dat_stream=None, maxfev=10, side_dat_stream=None):
        """
        @param dat_stream: ignores this
        """
        # Train a good initial model
        self.modeler.fit(dat.x, dat.y.flatten())
        self.modeler.coef_[:] = self.data_gen.beta.flatten()
        self.modeler.intercept_[:] = 0
        orig_coefs = self.modeler.coef_[:]
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))
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
            for var_idx in range(2, 1 + test_x.shape[1]):
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
                        self.predef_modeler.coef_[0, test_hist.curr_time] += (
                            self.update_dirs[0] * self.update_incr
                        )

                        # Test the performance
                        null_constraints = np.array([
                                [0, sens_test],
                                [0, spec_test]])
                        test_res = mtp_mechanism.get_test_res(
                            null_constraints, proposed_mdl, predef_mdl=self.predef_modeler
                        )
                        print("perturb?", test_hist.curr_time, var_idx, update_dir, test_res)
                        if test_res:
                            print("TEST RES")
                            sens_curr = sens_test
                            spec_curr = spec_test
                            sens_test += self.incr_sens_spec
                            spec_test += self.incr_sens_spec
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
                            proposed_mdl=proposed_mdl
                        )
                    if test_res == 0 and scale_factor > 1:
                        break

        return test_hist

class OnlineFixedSensSpecModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    """
    def __init__(self, model_type:str = "Logistic", seed:int = 0, incr_sens_spec: float = 0.02, init_sensitivity = 0.6, init_specificity = 0.6):
        if model_type == "Logistic":
            self.modeler = MyLogisticRegression(penalty="none")
        elif model_type == "SelectiveLogistic":
            self.modeler = SelectiveLogisticRegression(penalty="none", target_acc=0.85)
        else:
            raise NotImplementedError("model type missing")
        self.incr_sens_spec = incr_sens_spec
        self.curr_sensitivity = init_sensitivity
        self.curr_specificity = init_specificity
        self.sensitivity_test = init_sensitivity + incr_sens_spec
        self.specificity_test = init_specificity + incr_sens_spec

    def simulate_approval_process(self, dat, test_x, test_y, mtp_mechanism, dat_stream, maxfev=10, side_dat_stream=None):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))

        predef_dat = dat
        curr_idx = 0
        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
                "sensitivity_curr": [self.curr_sensitivity],
                "specificity_curr": [self.curr_specificity],
                }))
        for i in range(maxfev):
            print("ITERATION", i)

            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + 1])
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())

            # TODO: this should be defined adaptively
            null_constraints = np.array([
                    [0,self.sensitivity_test],
                    [0,self.specificity_test]])
            test_res = mtp_mechanism.get_test_res(
                null_constraints, predef_lr, predef_mdl=predef_lr
            )
            if test_res:
                self.curr_sensitivity = self.sensitivity_test
                self.curr_specificity = self.specificity_test
                self.sensitivity_test += self.incr_sens_spec
                self.specificity_test += self.incr_sens_spec

            test_hist.update(
                    test_res=test_res,
                    res_detail = pd.DataFrame({
                        "sensitivity_curr": [self.curr_sensitivity],
                        "specificity_curr": [self.curr_specificity]}),
                    proposed_mdl=predef_lr)
        return test_hist

class OnlineFixedSelectiveModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    """
    def __init__(self, model_type:str = "SelectiveLogistic", seed:int = 0, incr_accept: float = 0.02, init_accept= 0.6, target_acc: float = 0.85):
        assert model_type == "SelectiveLogistic"
        self.modeler = SelectiveLogisticRegression(penalty="none", target_acc=target_acc)
        self.incr_accept = incr_accept
        self.curr_accept = init_accept
        self.accept_test = init_accept + incr_accept
        self.accuracy_test = target_acc

    def simulate_approval_process(self, dat, test_x, test_y, mtp_mechanism, dat_stream, maxfev=10, side_dat_stream=None):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))

        predef_dat = dat
        curr_idx = 0
        test_hist = TestHistory(self.modeler)
        for i in range(maxfev):
            print("ITERATION", i)

            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + 1])
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())

            null_constraints = np.array([
                    [0,self.accept_test],
                    [0,self.accuracy_test]])
            test_res = mtp_mechanism.get_test_res(
                null_constraints, predef_lr, predef_mdl=predef_lr
            )
            if test_res:
                self.curr_accept = self.accept_test
                self.accept_test += self.incr_accept

            test_hist.update(
                    test_res=test_res,
                    res_detail = pd.DataFrame({
                        "accept_curr": [self.curr_accept]}),
                    proposed_mdl=predef_lr)
        return test_hist
