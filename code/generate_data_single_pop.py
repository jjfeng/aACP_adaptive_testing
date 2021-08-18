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
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--meta-seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--sparse-p',
        type=int,
        default=4)
    parser.add_argument(
        '--p',
        type=int,
        default=10)
    parser.add_argument(
        '--init-sparse-beta',
        type=float,
        default=0.5)
    parser.add_argument(
        '--init-reuse-test-n',
        type=int,
        default=300)
    parser.add_argument(
        '--init-train-n',
        type=int,
        default=1000)
    parser.add_argument(
        '--train-batch-n',
        type=int,
        default=100)
    parser.add_argument(
        '--train-iters',
        type=int,
        default=10)
    parser.add_argument(
        '--test-n',
        type=int,
        default=2000)
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    # parameters
    np.random.seed(args.meta_seed)

    # Prep data
    init_beta = np.random.normal(size=(args.p,1)) * 0.05
    init_beta[:args.sparse_p] += -args.init_sparse_beta
    data_generator = DataGenerator(init_beta, mean_x=0)
    full_dat, beta_time_varying = data_generator.generate_data(args.init_train_n, args.train_batch_n, args.train_iters, args.init_reuse_test_n, args.test_n)

    with open(args.out_file, "wb") as f:
        pickle.dump({
            "full_dat": full_dat,
            "betas": [beta_time_varying],
            }, f)


if __name__ == "__main__":
    main()
