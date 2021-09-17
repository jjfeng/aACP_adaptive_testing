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
    parser = argparse.ArgumentParser(description="run simulation")
    parser.add_argument("--meta-seed", type=int, default=1235, help="seed")
    parser.add_argument("--data-seed", type=int, default=1235, help="seed")
    parser.add_argument("--sparse-p", type=int, default=4)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--init-sparse-beta", type=float, default=0.5)
    parser.add_argument("--perturb-beta", type=float, default=None)
    parser.add_argument("--init-reuse-test-n", type=int, default=300)
    parser.add_argument("--init-train-n", type=int, default=1000)
    parser.add_argument("--train-batch-n", type=int, default=100)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--test-n", type=int, default=2000)
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
    init_beta = np.zeros((args.p, 1))
    init_beta[: args.sparse_p] = args.init_sparse_beta
    perturb_delta = np.random.rand(args.p, 1)
    if args.perturb_beta is not None:
        perturbed_beta = init_beta - perturb_delta * args.perturb_beta
    else:
        perturbed_beta = None

    np.random.seed(args.data_seed)
    data_generator = DataGenerator(init_beta, mean_x=0, perturb_beta=perturbed_beta)
    full_dat, betas = data_generator.generate_data(
        args.init_train_n,
        args.train_batch_n,
        args.train_iters,
        args.init_reuse_test_n,
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
