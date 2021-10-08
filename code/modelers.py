from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.base
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from dataset import Dataset


class TestHistory:
    def __init__(self, curr_mdl):
        self.approval_times = [0]
        self.approved_mdls = [deepcopy(curr_mdl)]
        self.proposed_mdls = [deepcopy(curr_mdl)]
        self.curr_time = 0
        self.num_trains = [0]

    def update(self, test_res, proposed_mdl, num_train):
        self.curr_time += 1
        proposed_mdl = deepcopy(proposed_mdl)
        if test_res == 1:
            self.approval_times.append(self.curr_time)
            self.approved_mdls.append(proposed_mdl)

        self.proposed_mdls.append(proposed_mdl)
        self.num_trains.append(num_train)

    @property
    def tot_approves(self):
        return len(self.approval_times)


class LockedModeler:
    def __init__(self, model_type:str = "Logistic", seed:int = 0):
        if model_type == "Logistic":
            self.modeler = LogisticRegression(penalty="none")
        elif model_type == "GBT":
            self.modeler = GradientBoostingClassifier()
        else:
            raise NotImplementedError("model type missing")

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

    def simulate_approval_process(self, dat, test_x, test_y, dp_engine, iid_dat_stream=None, maxfev=10, side_dat_stream=None):
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
            return mtp_answer, lr

        # Now search in each direction and do a greedy search
        test_hist = TestHistory(curr_mdl=self.modeler)
        curr_coef = np.concatenate(
            [self.modeler.intercept_, self.modeler.coef_.flatten()]
        )
        curr_perf, _ = get_test_perf(curr_coef)
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
                    test_res, proposed_mdl = get_test_perf(curr_coef)
                    print("preturb?", var_idx, curr_perf, test_res)
                    # print(test_res, curr_perf, var_idx, update_dir)
                    test_hist.update(
                        test_res=test_res < curr_perf, proposed_mdl=proposed_mdl, num_train=0
                    )
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
                    test_res, proposed_mdl = get_test_perf(curr_coef)
                    print("preturb cont?", var_idx, curr_perf, test_res)
                    test_hist.update(
                        test_res=test_res, curr_mdl=proposed_mdl, num_train=0
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

    def simulate_approval_process(self, dat, test_x, test_y, dp_engine, dat_stream=None, maxfev=10, side_dat_stream=None):
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
            lr = sklearn.base.clone(self.modeler)
            self.set_model(lr, params)
            pred_y = lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            self.predef_modeler.coef_[0, :] = orig_coefs
            self.predef_modeler.coef_[0, curr_time + self.min_var_idx] += (
                self.update_dirs[0] * self.update_incr
            )
            predef_pred_y = self.predef_modeler.predict_proba(test_x)[:, 1].reshape(
                (-1, 1)
            )

            mtp_answer = dp_engine.get_test_compare(
                test_y, pred_y, prev_pred_y, predef_pred_y=predef_pred_y
            )
            return mtp_answer, lr

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
                    test_res, proposed_mdl = get_test_perf(curr_coef, test_hist.curr_time)
                    print("perturb?", test_hist.curr_time, var_idx, test_res)
                    test_hist.update(
                        test_res=test_res, proposed_mdl=proposed_mdl, num_train=0
                    )
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
                    test_res, proposed_mdl = get_test_perf(curr_coef, test_hist.curr_time)
                    test_hist.update(
                        test_res=test_res, proposed_mdl=proposed_mdl, num_train=0
                    )
                    if test_res == 1:
                        self.set_model(self.modeler, curr_coef)
                        ctr *= 2
        return test_hist


class AdversarialModeler(LockedModeler):
    def __init__(self, preset_coef: float = 0, min_var_idx: int = 1):
        self.cts_modeler = CtsAdversaryModeler(preset_coef, min_var_idx)
        self.binary_modeler = BinaryAdversaryModeler(preset_coef, min_var_idx)
        self.modeler = self.cts_modeler.modeler

    def simulate_approval_process(self, dat, test_x, test_y, dp_engine, dat_stream=None, maxfev=10, side_dat_stream=None):
        """
        @param dat_stream: ignores this

        @return perf_value
        """
        if dp_engine.name == "no_dp":
            test_hist = self.cts_modeler.simulate_approval_process(
                dat, test_x, test_y, dp_engine, dat_stream, maxfev, side_dat_stream
            )
            self.modeler = self.cts_modeler.modeler
        else:
            test_hist = self.binary_modeler.simulate_approval_process(
                dat, test_x, test_y, dp_engine, dat_stream, maxfev, side_dat_stream
            )
            self.modeler = self.binary_modeler.modeler
        return test_hist


class OnlineLearnerFixedModeler(LockedModeler):
    """
    Just do online learning on a separate dataset
    """

    def simulate_approval_process(self, dat, test_x, test_y, dp_engine, dat_stream, maxfev=10, side_dat_stream=None):
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

            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + 1])
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())
            predef_pred_y = predef_lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            test_res = dp_engine.get_test_compare(
                test_y, predef_pred_y, prev_pred_y=prev_pred_y, predef_pred_y=predef_pred_y)

            test_hist.update(test_res=test_res, proposed_mdl=predef_lr, num_train=predef_dat.size - dat.size)
        return test_hist


class OnlineAdaptiveLearnerModeler(OnlineLearnerFixedModeler):
    """
    Just do online learning on a separate dataset

    This learner adapts by deciding whether or not to read from a side data stream
    If modification not approved, reads from the side data stream
    """
    predef_batches = 1
    def __init__(self, model_type: str, start_side_batch: int = 0):
        """
        @param start_side_batch: whether to start with reading the side data stream
        """
        super(OnlineAdaptiveLearnerModeler,self).__init__(model_type)
        self.start_side_batch = start_side_batch

    def simulate_approval_process(self, dat, test_x, test_y, dp_engine, dat_stream, maxfev=10, side_dat_stream=None):
        """
        @param dat_stream: a list of datasets for further training the model
        @param side_dat_stream: a list of side datasets for further training the model (these datasets are not IID)
        @return perf_value
        """
        self.modeler.fit(dat.x, dat.y.flatten())
        prev_pred_y = self.modeler.predict_proba(test_x)[:, 1].reshape((-1, 1))

        adapt_dat = dat
        predef_dat = dat
        read_side_batch = self.start_side_batch
        curr_idx = 0
        test_hist = TestHistory(self.modeler)
        for i in range(maxfev):
            if read_side_batch:
                batches_read = side_dat_stream[i: i + 1]
            else:
                batches_read = dat_stream[i : i + 1]

            adapt_dat = Dataset.merge([adapt_dat] + batches_read)
            adapt_lr = sklearn.base.clone(self.modeler)
            adapt_lr.fit(adapt_dat.x, adapt_dat.y.flatten())
            adapt_pred_y = adapt_lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            predef_dat = Dataset.merge([predef_dat] + dat_stream[i : i + 1])
            predef_lr = sklearn.base.clone(self.modeler)
            predef_lr.fit(predef_dat.x, predef_dat.y.flatten())
            predef_pred_y = predef_lr.predict_proba(test_x)[:, 1].reshape((-1, 1))

            test_res = dp_engine.get_test_compare(
                test_y, adapt_pred_y, prev_pred_y=prev_pred_y, predef_pred_y=predef_pred_y)
            # read side batch next iter if modification not approved
            read_side_batch = not read_side_batch if (test_res == 0) else read_side_batch

            test_hist.update(test_res=test_res, proposed_mdl=adapt_lr, num_train=adapt_dat.size - dat.size)
        return test_hist
