#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle

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
    args = parser.parse_args()
    args.results = args.results.split(",")
    return args


def main():
    args = parse_args()

    all_res = []
    for idx, res_file in enumerate(args.results):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            batch_dict = res[res.variable == "batch_number"][["time", "value"]]
            batch_df = pd.DataFrame({
                "batch_number": batch_dict.value,
                "time": batch_dict.time}).drop_duplicates()

            res = res.merge(batch_df, on="time")
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
    print("Number of replicates:", num_replicates)
    all_res = pd.concat(all_res)
    print(all_res)

    # Rename all the things for prettier figures
    measure_dict = {
            'curr_diff': 'curr_diff',
            'num_approvals': 'num_approvals',
            'auc': 'auc',
            }
    data_dict = {'test':'Test', 'reuse_test': 'Reusable Test'}
    mtp_dict = {
            'binary_thres': 'BinaryThres',
            'bonferroni': 'bonf_std',
            'graphical_bonf_thres': 'bonfSRGP',
            'graphical_ffs': 'fsSRGP',
            'graphical_par': "presSRGP"
            }
    all_res = all_res.rename({
        "value": "Value",
        "time": "Iteration",
        }, axis=1)
    all_res["Measure"] = all_res.variable.map(measure_dict)
    all_res["Procedure"] = all_res.procedure.map(mtp_dict)
    all_res["Dataset"] = all_res.dataset.map(data_dict)
    all_res = all_res.reset_index()
    max_iter = all_res.Iteration.max()
    print("NUM APPROVALS")
    print(all_res[
        (all_res.variable == "num_approvals")
        & (all_res.Iteration == max_iter)
        ])

    sns.set_context("paper", font_scale=2.5)
    rel_plt = sns.relplot(
        data=all_res[all_res.variable.isin(list(measure_dict.keys()))],
        x="batch_number",
        y="Value",
        hue="Procedure",
        row="Dataset",
        col="Measure",
        kind="line",
        style="Procedure",
        facet_kws={"sharey": False, "sharex": True},
    )
    rel_plt.set_titles('{row_name}' ' | ' '{col_name}')
    plt.savefig(args.plot_file)
    print("Fig", args.plot_file)


if __name__ == "__main__":
    main()
