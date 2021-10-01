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
            all_res.append(res)
        else:
            print("file missing", res_file)
    num_replicates = len(all_res)
    print("Number of replicates:", num_replicates)
    all_res = pd.concat(all_res)

    # Rename all the things for prettier figures
    measure_dict = {'nll': 'NLL', 'auc': 'AUC', 'did_approval': 'did_approval'}
    data_dict = {'test':'Test', 'reuse_test': 'Reusable Test'}
    mtp_dict = {
            'binary_thres': 'BinaryThres',
            'graphical_bonf_thres': 'bonfSRGP',
            'graphical_ffs': 'fsSRGP',
            'graphical_par': "presSRGP"
            }
    all_res = all_res.rename({
        "value": "Value",
        "max_iter": "Iteration",
        }, axis=1)
    all_res["Measure"] = all_res.measure.map(measure_dict)
    all_res["Procedure"] = all_res.dp.map(mtp_dict)
    all_res["Dataset"] = all_res.dataset.map(data_dict)
    print("ALL RES", all_res.shape)

    sns.set_context("paper", font_scale=2.5)
    rel_plt = sns.relplot(
        data=all_res[all_res.measure.isin(list(measure_dict.keys()))],
        x="Iteration",
        y="Value",
        hue="Procedure",
        row="Dataset",
        col="Measure",
        kind="line",
        style="Procedure",
        facet_kws={"sharey": False, "sharex": True},
        linewidth=3
    )
    rel_plt.set_titles('{row_name}' ' | ' '{col_name}')
    print(rel_plt.axes_dict.keys())
    plt.delaxes(rel_plt.axes_dict[('Reusable Test', 'did_approval')])
    num_approve_ax = rel_plt.axes_dict[('Test', 'did_approval')]
    num_approve_ax.axhline(y=0.1, color='dimgray', linestyle='--')
    num_approve_ax.set_title("Error rate")
    num_approve_ax.set_ylim(-0.05,1)
    plt.savefig(args.plot_file)
    print("Fig", args.plot_file)


if __name__ == "__main__":
    main()
