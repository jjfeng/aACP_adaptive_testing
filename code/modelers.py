from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.base
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from dataset import Dataset


class TestHistory:
    def __init__(self, curr_mdl):
        self.approval_times = [0]
        self.approved_mdls = [deepcopy(curr_mdl)]
        self.curr_time = 0
        self.num_trains = [0]

    def update(self, test_res, curr_mdl, num_train):
        self.curr_time += 1
        if test_res == 1:
            self.approval_times.append(self.curr_time)
            self.approved_mdls.append(deepcopy(curr_mdl))

        self.num_trains.append(num_train)

    @property
    def tot_approves(self):
        return len(self.approval_times)


class LockedModeler:
    """
    Logistic reg only right now
    """

    def __init__(self):
        self.modeler = LogisticRegression(penalty="none", solver="lbfgs")

    def set_model(self, mdl, params):
        mdl.classes_ = np.array([0, 1])
        mdl.coef_ = params[1:].reshape((1, -1))
        mdl.intercept_ = np.array([params[0]])

    def predict_prob(self, x):
        return self.modeler.predict_proba(x)[:, 1].reshape((-1, 1))

class CtsAdversaryModeler(LockedModeler):
    def __init__(
        self, preset_coef: float = 0, min_var_idx: int = 1, update_incr: float = 0.02
    ):
        """
        @param min_var_idx: nelder mead only tunes coefficients with idx at least min_var_idx
        """
        self.modeler = LogisticRegression(penalty="none", solver="lbfgs")
        self.update_incr = update_incr
        self.min_var_idx = min_var_idx
        self.preset_coef = preset_coef

    def do_minimize(self, dat, test_x, test_y, dp_engine, dat_stream=None, maxfev=10):
        """
        @param dat_stream: ignores this
        """
        # Train a good initial model
        self.modeler.fit(dat.x, dat.y.flatten())
        self.modeler.intercept_[:] = 0
        self.modeler.coef_[0, : self.min_var_idx] = self.preset_coef
        self.modeler.coef_[0, self.min_var_idx :] = 0

        def get_test_perf(params):
            lr = sklearn.base.clone(self.modeler)
            self.set_model(lr, params)
            pred_y = lr.predict_proba(test_x)[:, 1].reshape((-1, 1))
            mtp_answer = dp_engine.get_test_eval(test_y, pred_y)
            return mtp_answer

        # Now search in each direction and do a greedy search
        test_hist = TestHistory(curr_mdl=self.modeler)
        curr_coef = np.concatenate(
            [self.modeler.intercept_, self.modeler.coef_.flatten()]
        )
        curr_perf = get_test_perf(curr_coef)
        while test_hist.curr_time < maxfev:
            # Test each variable (that's known to be irrelevant)
            for var_idx in range(1 + self.min_var_idx, 1 + test_x.shape[1]):
                # Test each direction for the variable
                is_success = False
                for update_dir in [-1, 1]:
                    if test_hist.curr_time >= maxfev:
                        break
                    curr_coef = np.concatenate(
                        [self.modeler.intercept_, self.modeler.coef_.flatten()]
                    )
                    curr_coef[var_idx] += update_dir * self.update_incr
                    test_res = get_test_perf(curr_coef)
                    print("preturb?", var_idx, curr_perf, test_res)
                    test_hist.update(
                        test_res=test_res < curr_perf, curr_mdl=self.modeler, num_train=0
                    )
                    # print(test_res, curr_perf, var_idx, update_dir)
                    if test_res < curr_perf:
                        is_success = True
                        self.set_model(self.modeler, curr_coef)
                        curr_perf = test_res
                        break

                # If we found a good direction, keep walking in that direction
                ctr = 1
                while is_success:
                    print("success!!!")
                    if test_hist.curr_time >= maxfev:
                        break
                    curr_coef = np.concatenate(
                        [self.modeler.intercept_, self.modeler.coef_.flatten()]
                    )
                    curr_coef[var_idx] += update_dir * self.update_incr * ctr
                    test_res = get_test_perf(curr_coef)
                    print("preturb cont?", var_idx, curr_perf, test_res)
                    test_hist.update(
                        test_res=test_res < curr_perf, curr_mdl=self.modeler, num_train=0
                    )
                    if test_res < curr_perf:
                        self.set_model(self.modeler, curr_coef)
                        curr_perf = test_res
                        is_success = True
                        print("CTR", ctr)
                        ctr *= 2
                    else:
                        is_success = False

        return test_hist


class BinaryAdversaryModeler(LockedModeler):
    update_dirs = [1]

    def __init__(
        self, preset_coef: float = 0, min_var_idx: int = 1, update_incr: float = 0.02
    ):
        """
        @param min_var_idx: nelder mead only tunes coefficients with idx at least min_var_idx
        """
        self.modeler = LogisticRegression(penalty="none", solver="lbfgs")
        self.update_incr = update_incr
        self.min_var_idx = min_var_idx
        self.preset_coef = preset_coef

    def do_minimize(self, dat, test_x, test_y, dp_engine, dat_stream=None, maxfev=10):
        """
        @param dat_stream: ignores this
        """
        # Train a good initial model
        self.modeler.fit(dat.x, dat.y.flatten())
        self.modeler.intercept_[:] = 0
        self.modeler.coef_[0, : self.min_var_idx] = self.preset_coef
        self.modeler.coef_[0, self.min_var_idx :] = 0
        orig_coefs = self.modeler.coef_[:]
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))

        # Also have some predefined perturber for reference
        # just so we can use the parallel procedure
        self.predef_modeler = sklearn.base.clone(self.modeler)
        self.predef_modeler.fit(dat.x, dat.y.flatten())
        self.predef_modeler.intercept_[:] = 0
        self.predef_modeler.coef_[0, : self.min_var_idx] = self.preset_coef
        self.predef_modeler.coef_[0, self.min_var_idx :] = 0

        def get_test_perf(params, curr_time):
            prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))
            lr = sklearn.base.clone(self.modeler)
            self.set_model(lr, params)
            pred_y = lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            self.predef_modeler.coef_[0, :] = orig_coefs
            self.predef_modeler.coef_[0, curr_time + self.min_var_idx] += (
                self.update_dirs[0] * self.update_incr
            )
            print("PREFER", self.predef_modeler.coef_)
            predef_pred_y = self.predef_modeler.predict_proba(test_x)[:, 1].reshape(
                (-1, 1)
            )

            mtp_answer = dp_engine.get_test_compare(
                test_y, pred_y, prev_pred_y, predef_pred_y=predef_pred_y
            )
            return mtp_answer

        # Now search in each direction and do a greedy search
        test_hist = TestHistory(self.modeler)
        while test_hist.curr_time < maxfev:
            # Test each variable (that's known to be irrelevant)
            for var_idx in range(1 + self.min_var_idx, 1 + test_x.shape[1]):
                # Test update for the variable
                for update_dir in self.update_dirs:
                    if test_hist.curr_time >= maxfev:
                        break
                    curr_coef = np.concatenate(
                        [self.modeler.intercept_, self.modeler.coef_.flatten()]
                    )
                    curr_coef[var_idx] += update_dir * self.update_incr
                    print("curr", curr_coef)
                    test_res = get_test_perf(curr_coef, test_hist.curr_time)
                    print("perturb?", test_hist.curr_time, var_idx, test_res)
                    test_hist.update(test_res=test_res, curr_mdl=self.modeler, num_train=0)
                    if test_res == 1:
                        self.set_model(self.modeler, curr_coef)

                # If we found a good direction, keep walking in that direction
                ctr = 2
                while test_res == 1:
                    if test_hist.curr_time >= maxfev:
                        break
                    curr_coef = np.concatenate(
                        [self.modeler.intercept_, self.modeler.coef_.flatten()]
                    )
                    curr_coef[var_idx] += update_dir * self.update_incr * ctr
                    test_res = get_test_perf(curr_coef, test_hist.curr_time)
                    print("perturb cont", var_idx, test_res)
                    test_hist.update(test_res=test_res, curr_mdl=self.modeler, num_train=0)
                    if test_res == 1:
                        self.set_model(self.modeler, curr_coef)
                        ctr *= 2
        print("coefs", self.modeler.coef_)
        return test_hist


class AdversarialModeler(LockedModeler):
    def __init__(self, preset_coef: float = 0, min_var_idx: int = 1):
        self.cts_modeler = CtsAdversaryModeler(preset_coef, min_var_idx)
        self.binary_modeler = BinaryAdversaryModeler(preset_coef, min_var_idx)
        self.modeler = self.cts_modeler.modeler

    def do_minimize(self, dat, test_x, test_y, dp_engine, dat_stream=None, maxfev=10):
        """
        @param dat_stream: ignores this

        @return perf_value
        """
        if dp_engine.name == "no_dp":
            test_hist = self.cts_modeler.do_minimize(
                dat, test_x, test_y, dp_engine, dat_stream, maxfev
            )
            self.modeler = self.cts_modeler.modeler
        else:
            test_hist = self.binary_modeler.do_minimize(
                dat, test_x, test_y, dp_engine, dat_stream, maxfev
            )
            self.modeler = self.binary_modeler.modeler
        return test_hist


class OnlineLearnerFixedModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    only does logistic reg
    """

    def do_minimize(self, dat, test_x, test_y, dp_engine, dat_stream, maxfev=10):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())

        merged_dat = dat
        test_hist = TestHistory(self.modeler)
        for i, batch_dat in enumerate(dat_stream[:maxfev]):
            merged_dat = Dataset.merge([merged_dat, batch_dat])
            lr = sklearn.base.clone(self.modeler)
            lr.fit(merged_dat.x, merged_dat.y.flatten())

            pred_y = lr.predict_proba(test_x)[:, 1].reshape((-1, 1))
            test_res = dp_engine.get_test_eval(test_y, pred_y, predef_pred_y=pred_y)
            if test_res == 1:
                # replace current modeler only if successful
                self.modeler = lr
            test_hist.update(test_res=test_res, curr_mdl=self.modeler, num_train=merged_dat.size - dat.size)
        return test_hist


class OnlineAdaptiveLearnerModeler(OnlineLearnerFixedModeler):
    """
    Just do online learning on a separate dataset

    This learner adapts the number of training batches it will read.
    Collect more training data if the modification is not approved

    only does logistic reg
    """
    max_batches = 5
    predef_batches = 1

    def do_minimize(self, dat, test_x, test_y, dp_engine, dat_stream, maxfev=10):
        """
        @param dat_stream: a list of datasets for further training the model
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))

        adapt_dat = dat
        predef_dat = dat
        num_read_batches = 1
        curr_idx = 0
        test_hist = TestHistory(self.modeler)
        for i in range(maxfev):
            batches_read = dat_stream[curr_idx : curr_idx + num_read_batches]
            print(
                "BATCHES READ", len(batches_read), i, curr_idx, curr_idx + num_read_batches
            )
            adapt_dat = Dataset.merge([adapt_dat] + batches_read)
            curr_idx += num_read_batches
            adapt_lr = sklearn.base.clone(self.modeler)
            adapt_lr.fit(adapt_dat.x, adapt_dat.y.flatten())
            adapt_pred_y = adapt_lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + self.predef_batches])
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())
            predef_pred_y = predef_lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            test_res = dp_engine.get_test_compare(
                test_y, adapt_pred_y, prev_pred_y=prev_pred_y, predef_pred_y=predef_pred_y)
            if test_res == 1:
                # replace current modeler only if successful
                self.modeler = adapt_lr
                num_read_batches = max(num_read_batches - 1, self.predef_batches)
            else:
                # read more batches if failed
                print("ADAPT", num_read_batches)
                num_read_batches = min(1 + num_read_batches, self.max_batches)

            test_hist.update(test_res=test_res, curr_mdl=self.modeler, num_train=adapt_dat.size - dat.size)
        return test_hist
