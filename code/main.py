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
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve

from dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="run simulation")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--obs-batch-size", type=int, default=1)
    parser.add_argument("--test-batch", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--data-file", type=str, default="_output/data.pkl")
    parser.add_argument("--dp-mech-file", type=str, default="_output/dp_mech.pkl")
    parser.add_argument("--model-file", type=str, default="_output/model.pkl")
    parser.add_argument("--out-csv", type=str, default="_output/res.csv")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--plot-file", type=str, default=None)
    args = parser.parse_args()
    return args


def get_nll(test_y, pred_y):
    test_y = test_y.flatten()
    pred_y = np.maximum(np.minimum(1 - 1e-10, pred_y.flatten()), 1e-10)
    return -np.mean(test_y * np.log(pred_y) + (1 - test_y) * np.log(1 - pred_y))


def get_deployed_scores(test_hist, test_dat, max_iter):
    """
    @return Dataframe with auc and nll for the approved models for the given test data
    """
    last_approve_time = 0
    scores = []
    for approve_idx, (mdl, time_idx) in enumerate(
        zip(test_hist.approved_mdls, test_hist.approval_times)
    ):
        pred_y = mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
        auc = roc_auc_score(test_dat.y, pred_y)
        nll = get_nll(test_dat.y, pred_y)
        next_approve_time = (
            test_hist.approval_times[approve_idx + 1]
            if test_hist.tot_approves > (approve_idx + 1)
            else max_iter + 1
        )
        for idx in range(time_idx, next_approve_time):
            scores.append({"auc": auc, "nll": nll, "time": idx})
    scores = pd.DataFrame(scores)
    print(scores)
    return scores

def get_good_bad_proportion_approved(test_hist, test_dat, max_iter):
    """
    @return tuple with good and bad proportions approved
    """
    orig_mdl = test_hist.approved_mdls[0]
    orig_pred_y = orig_mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
    orig_nll = get_nll(test_dat.y, orig_pred_y)
    approval_idxs = np.array(test_hist.approval_times[1:])

    is_good_list = np.array([], dtype=bool)
    good_proportion_approved = [0]
    bad_proportion_approved = [0]
    for idx, mdl in enumerate(test_hist.proposed_mdls[1:]):
        time_idx = idx + 1
        pred_y = mdl.predict_proba(test_dat.x)[:, 1].reshape((-1, 1))
        proposed_nll = get_nll(test_dat.y, pred_y)
        is_good = proposed_nll <= orig_nll
        is_good_list = np.concatenate([is_good_list, [is_good]])
        is_bad_list = np.logical_not(is_good_list)

        if approval_idxs.size > 0:
            num_good_approved = np.sum(is_good_list[approval_idxs[approval_idxs <= time_idx] - 1])
            num_bad_approved = np.sum(is_bad_list[approval_idxs[approval_idxs <= time_idx] - 1])
        else:
            num_good_approved = 0
            num_bad_approved = 0
        num_good = np.sum(is_good_list)
        num_bad = np.sum(is_bad_list)
        good_proportion_approved.append(num_good_approved/num_good if num_good > 0 else 0)
        bad_proportion_approved.append(num_bad_approved/num_bad if num_bad > 0 else 0)

    return np.array(good_proportion_approved), np.array(bad_proportion_approved)

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

    with open(args.dp_mech_file, "rb") as f:
        dp_mech = pickle.load(f)

    with open(args.model_file, "rb") as f:
        modeler = pickle.load(f)

    np.random.seed(args.seed)

    # Run simulation
    dp_mech.set_num_queries(args.max_iter)
    full_hist = modeler.do_minimize(
        data.init_train_dat,
        data.reuse_test_dat.x,
        data.reuse_test_dat.y,
        dp_mech,
        dat_stream=data.iid_train_dat_stream,
        maxfev=args.max_iter,
        side_dat_stream=data.side_train_dat_stream,
    )
    print("APPROVAL", full_hist.approval_times)

    reuse_res = get_deployed_scores(full_hist, data.reuse_test_dat, args.max_iter)
    test_res = get_deployed_scores(full_hist, data.test_dat, args.max_iter)
    good_prop_approvals, bad_prop_approvals = get_good_bad_proportion_approved(full_hist, data.test_dat, args.max_iter)
    num_approvals = np.array(
        [
            np.sum(np.array(full_hist.approval_times) <= i) - 1
            for i in range(args.max_iter + 1)
        ]
    )

    # Compile results
    max_iters = np.arange(args.max_iter + 1)
    reuse_nll_df = pd.DataFrame({"value": reuse_res.nll, "max_iter": max_iters})
    reuse_nll_df["dataset"] = "reuse_test"
    reuse_nll_df["measure"] = "nll"
    reuse_auc_df = pd.DataFrame({"value": reuse_res.auc, "max_iter": max_iters})
    reuse_auc_df["dataset"] = "reuse_test"
    reuse_auc_df["measure"] = "auc"
    bad_prop_df = pd.DataFrame({"value": bad_prop_approvals, "max_iter": max_iters})
    bad_prop_df["dataset"] = "test"
    bad_prop_df["measure"] = "bad_prop"
    good_prop_df = pd.DataFrame({"value": good_prop_approvals, "max_iter": max_iters})
    good_prop_df["dataset"] = "test"
    good_prop_df["measure"] = "good_prop"
    count_df = pd.DataFrame({"value": num_approvals, "max_iter": max_iters})
    count_df["dataset"] = "test"
    count_df["measure"] = "num_approvals"
    approve_df = pd.DataFrame({"value": num_approvals > 0, "max_iter": max_iters})
    approve_df["dataset"] = "test"
    approve_df["measure"] = "did_approval"
    test_nll_df = pd.DataFrame({"value": test_res.nll, "max_iter": max_iters})
    test_nll_df["dataset"] = "test"
    test_nll_df["measure"] = "nll"
    test_auc_df = pd.DataFrame({"value": test_res.auc, "max_iter": max_iters})
    test_auc_df["dataset"] = "test"
    test_auc_df["measure"] = "auc"
    df = pd.concat(
        [reuse_nll_df, reuse_auc_df, count_df, approve_df, test_auc_df, test_nll_df, good_prop_df, bad_prop_df, count_df],
    )
    df["dp"] = dp_mech.name
    print("results")
    print(df)

    # Plot
    if args.plot_file:
        print(df)
        sns.set_context("paper", font_scale=2)
        rel_plt = sns.relplot(
            data=df,
            x="max_iter",
            y="value",
            hue="dataset",
            col="measure",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
        )
        rel_plt.fig.suptitle(dp_mech.name)
        plt.savefig(args.plot_file)
        print("Fig", args.plot_file)

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
