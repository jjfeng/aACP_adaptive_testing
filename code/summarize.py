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
    parser = argparse.ArgumentParser(description='Summarize csvs by taking mean')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--pivot-cols',
        type=str,
        help="pivot cols")
    parser.add_argument(
        '--pivot-rows',
        type=str,
        help="pivot rows")
    parser.add_argument(
        '--measure-filter',
        type=str,
        help="measure to summarize",
        default="nll")
    parser.add_argument(
        '--id-cols',
        type=str,
        help="columns to use as ids, remaining will be treated as values to summarize")
    parser.add_argument(
        '--value-col',
        type=str,
        help="value to plot")
    parser.add_argument(
        '--results',
        type=str)
    parser.add_argument(
        '--out-csv',
        type=str)
    args = parser.parse_args()
    args.measure_filter = args.measure_filter.split(",")
    args.pivot_rows = args.pivot_rows.split(",")
    args.pivot_cols = args.pivot_cols.split(",")
    args.results = args.results.split(",")
    args.id_cols = args.id_cols.split(",")
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
    print(all_res)
    all_res_mean = all_res.groupby(args.id_cols).mean().reset_index()
    all_res_std = (all_res.groupby(args.id_cols).std()/np.sqrt(num_replicates)).reset_index()
    all_res_std["zagg"] = "se"
    all_res_mean["zagg"] = "mean"
    all_res = pd.concat([all_res_mean, all_res_std]) #.sort_values(["measure", "zagg", "mdl"])
    mask = all_res.measure.isin(args.measure_filter)
    out_df = all_res[mask].pivot(args.pivot_rows, args.pivot_cols + ["zagg"], ["value"])
    print(out_df)

    with open(args.out_csv, "w") as f:
        f.writelines(out_df.to_csv(index=True, float_format="%.3f"))

if __name__ == "__main__":
    main()

