#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

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
    logging.info(args)
    logging.info("Number of replicates: %d", len(args.results))

    all_res = []
    for idx, res_file in enumerate(args.results):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            all_res.append(res)
        else:
            print("file missing", res_file)
    num_replicates = len(all_res)
    all_res = pd.concat(all_res)

    # Rename all the things for prettier figures
    measure_dict = {'nll': 'NLL', 'auc': 'AUC',
            'good_approvals': 'Number of good approvals',
            #'prop_good_approvals': 'Fraction of good models approved',
            # This is actually an upper bound for bad approval rate.
            'bad_approvals': 'Error rate'}
    data_dict = {'test':'Test'}
    mtp_dict = {
            'bonferroni_thres': 'Bonferroni',
            'graphical_bonf_thres': 'bonfSRGP',
            'graphical_ffs': 'fsSRGP',
            'graphical_par': "presSRGP"}

    all_res = all_res[all_res.measure.isin(list(measure_dict.keys()))]
    all_res = all_res[all_res.dataset.isin(list(data_dict.keys()))]

    all_res = all_res.rename({
        "value": "Value",
        "iteration": "Iteration",
        }, axis=1)
    all_res["Measure"] = all_res.measure.map(measure_dict)
    all_res["Procedure"] = all_res.procedure.map(mtp_dict)
    all_res["Dataset"] = all_res.dataset.map(data_dict)
    all_res = all_res.sort_values(["Procedure", "Measure"])
    print(all_res)

    sns.set_context("paper", font_scale=3)
    rel_plt = sns.relplot(
        data=all_res[all_res.measure.isin(list(measure_dict.keys()))],
        x="Iteration",
        y="Value",
        hue="Procedure",
        col="Measure",
        kind="line",
        style="Procedure",
        facet_kws={"sharey": False, "sharex": True},
        col_order=["NLL", "AUC", "Error rate", "Number of good approvals"], #"Fraction of good models approved"],
        linewidth=5,
    )
    rel_plt.set_titles('{col_name}')
    print(rel_plt.axes_dict.keys())
    plt.savefig(args.plot_file)
    print("Fig", args.plot_file)


if __name__ == "__main__":
    main()
