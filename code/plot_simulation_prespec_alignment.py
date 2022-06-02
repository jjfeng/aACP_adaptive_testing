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
    parser.add_argument("--odd-results", type=str, default="")
    parser.add_argument("--mod-results", type=str, default="")
    parser.add_argument("--even-results", type=str, default="")
    parser.add_argument("--plot-file", type=str, default="_output/plot.png")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    args = parser.parse_args()
    args.odd_results = args.odd_results.split(",")
    args.mod_results = args.mod_results.split(",")
    args.even_results = args.even_results.split(",")
    return args

def read_res(results, max_batch, mdl_type):
    all_res = []
    for idx, res_file in enumerate(results):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            seed = int(re.findall(r'seed_\d+', res_file)[0].split("_")[1])
            res["seed"] = seed
            batch_dict = res[res.variable == "batch_number"][["time", "value"]]
            batch_df = pd.DataFrame({
                "batch_number": batch_dict.value,
                "time": batch_dict.time}).drop_duplicates()

            res = res.merge(batch_df, on="time")
            for i in range(batch_df.shape[0]):
                batch_start = int(batch_df.batch_number.iloc[i])
                if i == (batch_df.shape[0] - 1):
                    batch_end = max_batch
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
    all_res = pd.concat(all_res)
    all_res["mdl"] = mdl_type
    return all_res

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )

    all_odd_res = read_res(args.odd_results, args.max_batch, "Odd")
    all_mod_res = read_res(args.mod_results, args.max_batch, "Even/Odd")
    all_even_res = read_res(args.even_results, args.max_batch, "Even")
    all_res = pd.concat([
        all_odd_res,
        all_mod_res,
        all_even_res])

    # Rename all the things for prettier figures
    measure_dict = {
            'curr_diff': 'Detected improvement',
            'num_approvals': 'Number of approvals',
            'auc': 'AUC',
            }
    all_res = all_res.rename({
        "value": "Value",
        "time": "Iteration",
        "batch_number": "Time",
        }, axis=1)
    all_res["Measure"] = all_res.variable.map(measure_dict)
    all_res["Model"] = all_res.mdl
    all_res["Dataset"] = all_res.dataset
    all_res = all_res.reset_index()
    all_res = all_res[(all_res.Measure != "AUC") | (all_res.dataset == "test")]
    max_batch = all_res.Time.max()
    print("MAX bATCH", max_batch)

    sns.set_context("paper", font_scale=2.5)
    print(all_res)
    rel_plt = sns.relplot(
        data=all_res[all_res.variable.isin(list(measure_dict.keys()))],
        x="Time",
        y="Value",
        hue="Model",
        col="Measure",
        kind="line",
        style="Model",
        facet_kws={"sharey": False, "sharex": True},
        linewidth=3,
    )
    rel_plt.set_titles('{col_name}')
    plt.savefig(args.plot_file)
    print("Fig", args.plot_file)


if __name__ == "__main__":
    main()
