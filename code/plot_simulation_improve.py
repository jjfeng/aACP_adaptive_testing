#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys, os
import argparse
import pickle
import logging

import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize csvs by taking mean")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--max-batch", type=int, default=0, help="max batch")
    parser.add_argument("--results", type=str)
    parser.add_argument("--plot-file", type=str, default="_output/plot.png")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    args = parser.parse_args()
    args.results = args.results.split(",")
    return args

def perform_ttests(final_method_values, test_measure):
    # PERFORM paired t-test
    METHODS = ["presSRGP", "fsSRGP", "bonfSRGP", "Bonferroni", "wBonferroni"]
    print(final_method_values)
    for method1 in METHODS:
        for method2 in METHODS:
            if method1 <= method2:
                continue

            logging.info("METHODS %s vs %s (%s)", method1, method2, test_measure)
            print(method1, method2)
            method_compare_df = final_method_values[final_method_values.Procedure.isin([method1, method2])]
            seeds = method_compare_df.seed.unique()
            method_compare_df = method_compare_df[method_compare_df.seed.isin(seeds)].sort_values("seed")
            print(method_compare_df)
            method1_values = method_compare_df.Value[method_compare_df.Procedure ==
                method1]
            print(method1_values)
            method2_values = method_compare_df.Value[method_compare_df.Procedure ==
                method2]
            print(method2_values)
            ttest_res = scipy.stats.ttest_1samp(method1_values.to_numpy() -
                    method2_values.to_numpy(), popmean=0)
            logging.info(ttest_res)
            print(method1, method2)
            print(ttest_res)

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )

    all_res = []
    for idx, res_file in enumerate(args.results):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            seed = int(re.findall(r'seed_\d+', res_file)[0].split("_")[1])
            res["seed"] = seed
            batch_dict = res[res.variable == "batch_number"][["time", "value"]]
            batch_df = pd.DataFrame({
                "batch_number": batch_dict.value,
                "time": batch_dict.time}).drop_duplicates()

            res = res.merge(batch_df, on="time")
            print(batch_df)
            for i in range(batch_df.shape[0]):
                batch_start = int(batch_df.batch_number.iloc[i])
                if i == (batch_df.shape[0] - 1):
                    batch_end = args.max_batch
                else:
                    batch_end = int(batch_df.batch_number.iloc[i + 1])
                batch_filler = res[res.batch_number == batch_start].copy()
                for batch_fill_idx in range(batch_start + 1, batch_end):
                    batch_filler.batch_number = batch_fill_idx
                    res = pd.concat([
                        res,
                        batch_filler])
            all_res.append(res)
        else:
            print("file missing", res_file)

    num_replicates = len(all_res)
    logging.info("Number of replicates: %d", num_replicates)
    all_res = pd.concat(all_res)
    print(all_res)

    # Rename all the things for prettier figures
    measure_dict = {
            'curr_diff': 'Detected improvement',
            'num_approvals': 'Number of approvals',
            'auc': 'AUC',
            }
    mtp_dict = {
            'binary_thres': 'BinaryThres',
            'weighted_bonferroni': 'wBonferroni',
            'bonferroni': 'Bonferroni',
            'graphical_bonf_thres': 'bonfSRGP',
            'graphical_ffs': 'fsSRGP',
            'graphical_par': "presSRGP"
            }
    all_res = all_res.rename({
        "value": "Value",
        "time": "Iteration",
        "batch_number": "Time",
        }, axis=1)
    all_res["Measure"] = all_res.variable.map(measure_dict)
    all_res["Procedure"] = all_res.procedure.map(mtp_dict)
    all_res["Dataset"] = all_res.dataset
    all_res = all_res.reset_index()
    all_res = all_res[(all_res.Measure != "AUC") | (all_res.dataset == "test")]
    max_batch = all_res.Time.max()
    print("MAX bATCH", max_batch)

    # PERFORM paired t-test
    final_method_values = all_res[(all_res.Time == max_batch) &
            (all_res.Measure == "AUC") & (all_res.Dataset == "test")]
    perform_ttests(final_method_values, "AUC")
    final_method_values = all_res[(all_res.Time == max_batch) &
            (all_res.Measure == "Number of approvals") & (all_res.Dataset == "test")]
    perform_ttests(final_method_values, "Number of approvals")

    sns.set_context("paper", font_scale=2.5)
    rel_plt = sns.relplot(
        data=all_res[all_res.variable.isin(list(measure_dict.keys()))],
        x="Time",
        y="Value",
        hue="Procedure",
        col="Measure",
        kind="line",
        style="Procedure",
        facet_kws={"sharey": False, "sharex": True},
        linewidth=3,
    )
    rel_plt.set_titles('{col_name}')
    plt.savefig(args.plot_file)
    print("Fig", args.plot_file)


if __name__ == "__main__":
    main()
