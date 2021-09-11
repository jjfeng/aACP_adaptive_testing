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
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--obs-batch-size',
        type=int,
        default=1)
    parser.add_argument(
        '--test-batch',
        type=int,
        default=1)
    parser.add_argument(
        '--min-iter',
        type=int,
        default=0)
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1)
    parser.add_argument(
        '--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument(
        '--dp-mech-file',
        type=str,
        default="_output/dp_mech.pkl")
    parser.add_argument(
        '--model-file',
        type=str,
        default="_output/model.pkl")
    parser.add_argument(
        '--out-csv',
        type=str,
        default="_output/res.csv")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument(
        '--plot-file',
        type=str,
        default=None)
    args = parser.parse_args()
    return args

def get_nll(test_y, pred_y):
    test_y = test_y.flatten()
    pred_y = np.maximum(np.minimum(1 - 1e-10, pred_y.flatten()), 1e-10)
    return -np.mean(test_y * np.log(pred_y) + (1 - test_y) * np.log(1 - pred_y))

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    # parameters
    np.random.seed(args.seed)
    logging.info(args)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]
    print("data done")

    with open(args.dp_mech_file, "rb") as f:
        dp_mech = pickle.load(f)

    with open(args.model_file, "rb") as f:
        modeler = pickle.load(f)

    # Run simulation
    reuse_nlls = []
    reuse_aucs = []
    test_nlls = []
    test_aucs = []
    prev_approval_time = 0
    last_approval_times = []
    max_iters = np.arange(args.min_iter, args.max_iter + 1)
    for max_iter in max_iters:
        print("===========RUN PROCEDURE FOR NUM STPES", max_iter)
        curr_approval_time = 0
        if max_iter > 0:
            dp_mech.set_num_queries(max_iter)
            full_hist = modeler.do_minimize(data.reuse_test_dat.x, data.reuse_test_dat.y, dp_mech, dat_stream=data.train_dat_stream, maxfev=max_iter)
            print("APPROVAL", full_hist.approval_times)
            curr_approval_time = full_hist.approval_times[-1]
            logging.info("APPROVAL %s", full_hist.approval_times)

        reuse_pred_y = modeler.predict_prob(data.reuse_test_dat.x)
        reuse_auc = roc_auc_score(data.reuse_test_dat.y, reuse_pred_y)
        reuse_nll = get_nll(data.reuse_test_dat.y, reuse_pred_y)
        reuse_nlls.append(reuse_nll)
        reuse_aucs.append(reuse_auc)
        #print(dp_mech.name, "reuse", "AUC", reuse_auc, "NLL", reuse_nll)

        test_pred_y = modeler.predict_prob(data.test_dat.x)
        test_auc = roc_auc_score(data.test_dat.y, test_pred_y)
        test_nll = get_nll(data.test_dat.y, test_pred_y)
        test_nlls.append(test_nll)
        test_aucs.append(test_auc)
        #print(dp_mech.name, "test", "AUC", test_auc, "NLL", test_nll)
        if curr_approval_time == 0:
            last_approval_times.append(0)
        else:
            last_approval_times.append(len(full_hist.approval_times) - 1)

    # Compile results
    reuse_nll_df = pd.DataFrame({"value": reuse_nlls,  "max_iter": max_iters})
    reuse_nll_df["dataset"] = "reuse_test"
    reuse_nll_df["measure"] = "nll"
    reuse_auc_df = pd.DataFrame({"value": reuse_aucs,  "max_iter": max_iters})
    reuse_auc_df["dataset"] = "reuse_test"
    reuse_auc_df["measure"] = "auc"
    count_df = pd.DataFrame({"value": last_approval_times,  "max_iter": max_iters})
    count_df["dataset"] = "reuse_test"
    count_df["measure"] = "num_approvals"
    test_nll_df = pd.DataFrame({"value": test_nlls,  "max_iter": max_iters})
    test_nll_df["dataset"] = "test"
    test_nll_df["measure"] = "nll"
    test_auc_df = pd.DataFrame({"value": test_aucs,  "max_iter": max_iters})
    test_auc_df["dataset"] = "test"
    test_auc_df["measure"] = "auc"
    df = pd.concat([reuse_nll_df, reuse_auc_df, count_df, test_auc_df, test_nll_df])
    df["dp"] = dp_mech.name
    print("results")
    print(df)

    # Plot
    if args.plot_file:
        print(df)
        sns.set_context("paper", font_scale=2)
        sns.relplot(data=df, x="max_iter", y="value", hue="dataset", col="measure", kind="line", facet_kws={'sharey': False, 'sharex': True})
        plt.savefig(args.plot_file)
        print("Fig", args.plot_file)

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
