#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import time
import argparse
import pickle
import logging
import progressbar
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve, log_loss

from dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="run simulation for approving algorithmic modifications")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--obs-batch-size", type=int, default=1)
    parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--data-file", type=str, default="_output/data.pkl")
    parser.add_argument("--mtp-mech-file", type=str, default="_output/mtp_mech.pkl")
    parser.add_argument("--model-file", type=str, default="_output/model.pkl")
    parser.add_argument("--out-csv", type=str, default="_output/res.csv")
    parser.add_argument("--scratch-file", type=str, default="_output/scratch.txt")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--plot-file", type=str, default=None)
    args = parser.parse_args()
    return args


def get_deployed_scores(mtp_mech, test_hist, test_dat, max_iter):
    """
    @return Dataframe with auc and nll for the approved models for the given test data
    """
    last_approve_time = 0
    test_y = test_dat.y.flatten()
    scores = []
    for approve_idx, (mdl, time_idx) in enumerate(
        zip(test_hist.approved_mdls, test_hist.approval_times)
    ):
        pred_prob = mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
        auc = mtp_mech.hypo_tester.get_auc(test_dat.y, pred_prob)
        nll = log_loss(test_dat.y, pred_prob)

        pred_y = mdl.predict(test_dat.x)
        sensitivity = np.sum((pred_y == test_y) * (test_y))/np.sum(test_y)
        specificity = np.sum((pred_y == test_y) * (1 - test_y))/np.sum(1 - test_y)

        next_approve_time = (
            test_hist.approval_times[approve_idx + 1]
            if test_hist.tot_approves > (approve_idx + 1)
            else max_iter + 1
        )
        for idx in range(time_idx, next_approve_time):
            scores.append({
                "auc": auc,
                "nll": nll,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "time": idx
                })
    scores = pd.DataFrame(scores)
    return pd.melt(scores, id_vars=['time'], value_vars=[c for c in list(scores.columns) if c != "time"])

#def get_good_bad_approved(test_hist, test_dat, max_iter):
#    """
#    @return tuple with total number of good approvals, total number of bad approvals, proportion of good models approved
#    """
#    orig_mdl = test_hist.approved_mdls[0]
#    orig_pred_y = orig_mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
#    approval_idxs = np.array(test_hist.approval_times[1:])
#
#    # Tracks whether or not proposed model at each time is good
#    is_good_list = np.array([], dtype=bool)
#    good_approved = [0]
#    prop_good_approved = [0]
#    bad_approved = [0]
#    for idx, mdl in enumerate(test_hist.proposed_mdls[1:]):
#        time_idx = idx + 1
#        pred_y = mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
#        is_good = proposed_nll <= orig_nll
#        is_good_list = np.concatenate([is_good_list, [is_good]])
#        is_bad_list = np.logical_not(is_good_list)
#
#        if approval_idxs.size > 0:
#            num_good_approved = np.sum(is_good_list[approval_idxs[approval_idxs <= time_idx] - 1])
#            num_bad_approved = np.sum(is_bad_list[approval_idxs[approval_idxs <= time_idx] - 1])
#        else:
#            num_good_approved = 0
#            num_bad_approved = 0
#        num_good = np.sum(is_good_list)
#        num_bad = np.sum(is_bad_list)
#        good_approved.append(num_good_approved)
#        prop_good_approved.append(num_good_approved/num_good if num_good > 0 else 0)
#        bad_approved.append(num_bad_approved)
#
#    return np.array(good_approved), np.array(bad_approved), np.array(prop_good_approved)

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    # parameters
    logging.info(args)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]
    print("data done")

    with open(args.mtp_mech_file, "rb") as f:
        mtp_mech = pickle.load(f)
        mtp_mech.hypo_tester.make_scratch(args.scratch_file)

    with open(args.model_file, "rb") as f:
        modeler = pickle.load(f)
        modeler.hypo_tester.make_scratch(args.scratch_file)

    np.random.seed(args.seed)

    # Run simulation
    mtp_mech.init_test_dat(data.reuse_test_dat, args.max_iter)
    full_hist = modeler.simulate_approval_process(
        data.init_train_dat,
        mtp_mech,
        dat_stream=data.iid_train_dat_stream,
        maxfev=args.max_iter,
        side_dat_stream=data.side_train_dat_stream
    )
    print("APPROVAL", full_hist.approval_times)
    logging.info("APPROVAL %s", full_hist.approval_times)

    conclusions_hist = full_hist.get_perf_hist()
    conclusions_hist["dataset"] = "reuse_test"
    reuse_res = get_deployed_scores(mtp_mech, full_hist, data.reuse_test_dat, args.max_iter)
    test_res = get_deployed_scores(mtp_mech, full_hist, data.test_dat, args.max_iter)
    reuse_res["dataset"] = "reuse_test"
    test_res["dataset"] = "test"
    #good_approvals, bad_approvals, prop_good_approvals = get_good_bad_approved(full_hist, data.test_dat, args.max_iter)
    num_approvals = np.array(
        [
            np.sum(np.array(full_hist.approval_times) <= i) - 1
            for i in range(args.max_iter + 1)
        ]
    )

    # Compile results
    times = np.arange(args.max_iter + 1)
    #bad_df = pd.DataFrame({"value": bad_approvals, "time": times})
    #bad_df["dataset"] = "test"
    #bad_df["variable"] = "bad_approvals"
    #good_df = pd.DataFrame({"value": good_approvals, "time": times})
    #good_df["dataset"] = "test"
    #good_df["variable"] = "good_approvals"
    count_df = pd.DataFrame({"value": num_approvals, "time": times})
    count_df["dataset"] = "test"
    count_df["variable"] = "num_approvals"
    approve_df = pd.DataFrame({"value": num_approvals > 0, "time": times})
    approve_df["dataset"] = "test"
    approve_df["variable"] = "did_approval"
    df = pd.concat([reuse_res, test_res, approve_df, count_df, conclusions_hist])
    df["procedure"] = mtp_mech.name

    # Plot
    if args.plot_file:
        #print(df)
        sns.set_context("paper", font_scale=2)
        rel_plt = sns.relplot(
            data=df,
            x="iteration",
            y="value",
            hue="dataset",
            col="measure",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
        )
        rel_plt.fig.suptitle(mtp_mech.name)
        plt.savefig(args.plot_file)
        print("Fig", args.plot_file)

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
