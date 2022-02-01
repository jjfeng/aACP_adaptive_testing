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
        self.num_sparse_theta = np.max(np.where(self.data_gen.beta.flatten() != 0)[0]) + 1
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
    def __init__(self, model_type:str = "Logistic", seed:int = 0, incr_sens_spec: float = 0.001, validation_frac: float = 0.2, min_valid_dat_size: int = 200):
        assert model_type == "Logistic"
        self.modeler = MyLogisticRegression(penalty="none")
        self.incr_sens_spec = incr_sens_spec
        self.validation_frac = validation_frac
        self.min_valid_dat_size = min_valid_dat_size

    def _get_sensitivity_specificity_lower_bound_diff(self, orig_mdl, new_mdl, valid_dat: Dataset, se_factor: float = 1.96):
        orig_pred_class = orig_mdl.predict(valid_dat.x)
        new_pred_class = new_mdl.predict(valid_dat.x)
        test_y = valid_dat.y.flatten()
        acc_diff = (new_pred_class == test_y).astype(int) - (orig_pred_class == test_y).astype(int)
        print("new", (new_pred_class == test_y).astype(int).mean())
        print("ORIG", (orig_pred_class == test_y).astype(int).mean())
        sensitivity = np.sum(acc_diff * test_y)/np.sum(test_y)
        specificity = np.sum(acc_diff * (1 - test_y))/np.sum(1 - test_y)
        sensitivity_se = np.sqrt(np.var(acc_diff[test_y == 1])/np.sum(test_y))
        specificity_se = np.sqrt(np.var(acc_diff[test_y == 0])/np.sum(1 - test_y))
        print("SandS", sensitivity, specificity, sensitivity_se, specificity_se)
        return sensitivity - se_factor * sensitivity_se, specificity - se_factor * specificity_se

    def _create_train_valid_dat(self, dat: Dataset):
        #shuffle_idxs = np.random.choice(dat.size, dat.size, replace=False)
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

        curr_idx = 0
        curr_sens = 0
        curr_spec = 0
        test_hist = TestHistory(orig_mdl, res_detail=pd.DataFrame({
                "sensitivity_curr": [0],
                "specificity_curr": [0],
                }))
        for i in range(maxfev):
            print("ITERATION", i)

            predef_dat = Dataset.merge([dat] + dat_stream[: i + 1])
            predef_train_dat, predef_valid_dat = self._create_train_valid_dat(predef_dat)
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten())
            new_sens, new_spec = self._get_sensitivity_specificity_lower_bound_diff(orig_mdl, predef_lr, predef_valid_dat)
            print("SENS SPEC VALUES", curr_sens, new_sens, curr_spec, new_spec)
            #assert (curr_sens + new_sens)/2 > curr_sens
            #assert (curr_spec + new_spec)/2 > curr_spec
            if ((curr_sens + new_sens)/2 > curr_sens) or ((curr_spec + new_spec)/2 > curr_spec):
                sens_test = max(curr_sens, (curr_sens + new_sens)/2)
                spec_test = max(curr_spec, (curr_spec + new_spec)/2)
            else:
                sens_test = curr_sens + self.incr_sens_spec
                spec_test = curr_spec + self.incr_sens_spec

            print("TEST", sens_test, spec_test)

            # TODO: this should be defined adaptively
            null_constraints = np.array([
                    [0,sens_test],
                    [0,spec_test]])
            test_res = mtp_mechanism.get_test_res(
                null_constraints, orig_mdl, predef_lr, predef_mdl=predef_lr
            )
            if test_res:
                curr_sens = sens_test
                curr_spec = spec_test

            test_hist.update(
                    test_res=test_res,
                    res_detail = pd.DataFrame({
                        "sensitivity_curr": [curr_sens],
                        "specificity_curr": [curr_spec]}),
                    proposed_mdl=predef_lr)
        return test_hist

class OnlineSensSpecModeler(OnlineFixedSensSpecModeler):
    """
    adaptive online testing
    """
    def __init__(self, model_type:str = "Logistic", seed:int = 0, incr_sens_spec: float = 1e-10, validation_frac: float = 0.2, min_valid_dat_size: int = 200, countdown_reset: int = 4):
        super(OnlineSensSpecModeler,self).__init__(model_type, seed, incr_sens_spec, validation_frac, min_valid_dat_size)
        self.countdown_reset = countdown_reset

    def _create_train_valid_dat(self, orig_dat: Dataset, new_dat: Dataset = None):
        num_obs = orig_dat.size + new_dat.size if new_dat is not None else orig_dat.size
        shuffle_idxs = np.flip(np.arange(num_obs))
        valid_n = max(min(orig_dat.size, int(num_obs * self.validation_frac)), self.min_valid_dat_size)
        print("valid_n", valid_n)
        merge_dat = Dataset.merge([orig_dat, new_dat]) if new_dat is not None else orig_dat
        train_dat = merge_dat.subset_idxs(shuffle_idxs[valid_n:])
        valid_dat = merge_dat.subset_idxs(shuffle_idxs[:valid_n])
        return train_dat, valid_dat

    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev: int, side_dat_stream: Dataset):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        train_dat, valid_dat = self._create_train_valid_dat(dat)
        self.modeler.fit(train_dat.x, train_dat.y.flatten())
        orig_mdl = self.modeler

        adapt_dat = []
        countdown = self.countdown_reset
        read_side_stream = False
        curr_idx = 0
        curr_sens = 0
        curr_spec = 0
        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
                "sensitivity_curr": [curr_sens],
                "specificity_curr": [curr_spec],
                }))

        for i in range(maxfev):
            print("ITERATION", i)

            predef_train_dat, _ = self._create_train_valid_dat(dat, Dataset.merge(dat_stream[:i + 1]))
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_train_dat.x, predef_train_dat.y.flatten())

            if read_side_stream:
                batches_read = side_dat_stream[i: i + 1]
            else:
                batches_read = dat_stream[i : i + 1]
            logging.info("BATCH countdown %d", countdown)
            logging.info("BATCH idx %d", read_side_stream)

            adapt_dat = adapt_dat + batches_read
            adapt_train_dat, adapt_valid_dat = self._create_train_valid_dat(
                    dat,
                    Dataset.merge(adapt_dat))
            print("ADAPT SIDE", adapt_valid_dat.size)
            adapt_lr = sklearn.base.clone(self.modeler)
            adapt_lr.fit(adapt_train_dat.x, adapt_train_dat.y.flatten())

            new_sens, new_spec = self._get_sensitivity_specificity_lower_bound_diff(orig_mdl, adapt_lr, adapt_valid_dat)
            print(curr_sens, new_sens)
            print(curr_spec, new_spec)
            #assert ((curr_sens + new_sens)/2 > curr_sens) or ((curr_spec + new_spec)/2 > curr_spec)
            sens_test = max(curr_sens, (curr_sens + new_sens)/2)
            spec_test = max(curr_spec, (curr_spec + new_spec)/2)
            print("NEW", sens_test, spec_test)
            logging.info("sens spec test %.3f %.3f", sens_test, spec_test)

            # TODO: this should be defined adaptively
            null_constraints = np.array([
                    [0,sens_test],
                    [0,spec_test]])
            test_res = mtp_mechanism.get_test_res(
                null_constraints, orig_mdl, adapt_lr, predef_mdl=predef_lr
            )
            logging.info("test res %d", test_res)
            if test_res:
                curr_sens = sens_test
                curr_spec = spec_test
                countdown = self.countdown_reset
            else:
                countdown -= 1
                if countdown == 0:
                    read_side_stream = not read_side_stream
                    countdown = self.countdown_reset

            test_hist.update(
                    test_res=test_res,
                    res_detail = pd.DataFrame({
                        "sensitivity_curr": [curr_sens],
                        "specificity_curr": [curr_spec]}),
                    proposed_mdl=adapt_lr)
        return test_hist


class OnlineFixedSelectiveModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    """
    def __init__(self, model_type:str = "SelectiveLogistic", seed:int = 0, incr_accept: float = 0.02, init_accept= 0.6, target_acc: float = 0.85, acc_buffer: float = 0.05):
        assert model_type == "SelectiveLogistic"
        self.modeler = SelectiveLogisticRegression(penalty="none", target_acc=target_acc + acc_buffer)
        self.incr_accept = incr_accept
        self.curr_accept = init_accept
        self.accept_test = init_accept + incr_accept
        self.accuracy_test = target_acc

    def simulate_approval_process(self, dat, mtp_mechanism, dat_stream, maxfev=10):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())

        predef_dat = dat
        curr_idx = 0
        test_hist = TestHistory(self.modeler, res_detail=pd.DataFrame({
                "accept_curr": [self.curr_accept],
                }))
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
