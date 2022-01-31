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
    parser.add_argument("--meta-seed", type=int, default=1235, help="seed for determining meta-properties of the data")
    parser.add_argument("--data-seed", type=int, default=1235, help="seed for determining the training data stream")
    parser.add_argument("--sparse-p", type=int, default=4, help="number of nonzero coefficients in the true logistic regression model")
    parser.add_argument("--p", type=int, default=10, help="number of covariates")
    parser.add_argument("--sparse-beta", type=float, default=0.5, help="values for the nonzero coefficients in the true model")
    parser.add_argument("--perturb-beta", type=float, default=None, help="how much to perturb the coefficients in the true model to simulate a side population")
    parser.add_argument("--reuse-test-n", type=int, default=300, help="how much data is in the reusable test data")
    parser.add_argument("--init-train-n", type=int, default=10, help="how much data was used to train the initial model")
    parser.add_argument("--train-batch-n", type=int, default=100, help="how much data is observed between each iteration, in the simulated data stream")
    parser.add_argument("--num-batches", type=int, default=10, help="total number of batches to create for the data stream")
    parser.add_argument("--test-n", type=int, default=2000, help="number of samples in the completely held out test dataset")
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
    np.random.seed(args.meta_seed)

    # Prep data
    true_beta = np.zeros((args.p, 1))
    true_beta[: args.sparse_p] = args.sparse_beta
    perturb_delta = np.random.rand(args.p, 1)
    if args.perturb_beta is not None:
        perturbed_beta = true_beta - perturb_delta * args.perturb_beta
    else:
        perturbed_beta = None
    logging.info("perturbed beta %s", perturbed_beta)

    np.random.seed(args.meta_seed + args.data_seed + 1)
    data_generator = DataGenerator(true_beta, mean_x=0, perturb_beta=perturbed_beta)
    full_dat, betas = data_generator.generate_data(
        args.init_train_n,
        args.train_batch_n,
        args.num_batches,
        args.reuse_test_n,
        args.test_n,
        train_side_stream_n=args.train_batch_n,
    )

    with open(args.out_file, "wb") as f:
        pickle.dump(
            {
                "full_dat": full_dat,
                "betas": betas,
            },
            f,
        )


if __name__ == "__main__":
    main()
