#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import scipy.stats
import numpy as np
import pandas as pd

from dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for simulations")
    parser.add_argument("--seed", type=int, default=1235, help="seed for determining meta-properties of the data")
    parser.add_argument("--reuse-test-n", type=int, default=300, help="how much data is in the reusable test data")
    parser.add_argument("--init-train-n", type=int, default=10, help="how much data was used to train the initial model")
    parser.add_argument("--train-batch-n", type=int, default=100, help="how much data is observed between each iteration, in the simulated data stream")
    parser.add_argument("--num-batches", type=int, default=10, help="total number of batches to create for the data stream")
    parser.add_argument("--test-n", type=int, default=2000, help="number of samples in the completely held out test dataset")
    parser.add_argument("--dat-file", type=str)
    parser.add_argument("--out-file", type=str, default="_output/data.pkl")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    # parameters
    np.random.seed(args.seed)

    # Prep data
    dat = np.genfromtxt(args.dat_file, delimiter=",", skip_header=True)
    print(dat)
    print(np.isfinite(dat[:,0]).mean())
    print(np.isfinite(dat[:,1]).mean())
    rand_idxs = np.random.choice(dat.shape[0], dat.shape[0], replace=False)
    # Shuffle data
    dat = dat[rand_idxs]
    print(dat[:,-1])
    print("OUTCOME RATE", dat[:,-1].mean())

    # Split data
    init_train_dat = Dataset(
            x=dat[:args.init_train_n,:-1],
            y=dat[:args.init_train_n,-1:],
            )
    start_idx = args.init_train_n
    reuse_test_dat = Dataset(
            x=dat[start_idx:start_idx + args.reuse_test_n,:-1],
            y=dat[start_idx:start_idx + args.reuse_test_n,-1:],
            )
    start_idx += args.reuse_test_n
    test_dat = Dataset(
            x=dat[start_idx:start_idx + args.test_n,:-1],
            y=dat[start_idx:start_idx + args.test_n,-1:],
            )
    start_idx += args.test_n
    iid_train_dats = []
    for batch_idx in range(args.num_batches):
        batch_start_idx = start_idx + batch_idx * args.train_batch_n
        dat_slice = dat[batch_start_idx:batch_start_idx + args.train_batch_n]
        iid_train_dats.append(
                Dataset(
                    x=dat_slice[:,:-1],
                    y=dat_slice[:,-1:],
            ))
    full_dat = FullDataset(
            init_train_dat,
            iid_train_dats,
            reuse_test_dat,
            test_dat)

    with open(args.out_file, "wb") as f:
        pickle.dump(
            {
                "full_dat": full_dat,
            },
            f,
        )


if __name__ == "__main__":
    main()
