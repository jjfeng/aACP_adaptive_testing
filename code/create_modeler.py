#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataset import Dataset
from modelers import *

def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--simulation',
        type=str,
        default="fixed",
        choices=["fixed", "neldermead", "online_fixed"])
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200)
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2)
    parser.add_argument(
        '--refit-freq',
        type=int,
        default=1)
    parser.add_argument(
        '--max-box',
        type=str,
        default="100")
    parser.add_argument(
        '--switch-time',
        type=int,
        default=None)
    parser.add_argument(
        '--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/models.pkl")
    args = parser.parse_args()
    args.max_box = list(map(int, args.max_box.split(",")))
    return args

def main():
    args = parse_args()
    # parameters
    np.random.seed(args.seed)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]

    # Create model
    if args.simulation == "neldermead":
        clf = NelderMeadModeler(data.init_train_dat)
    elif args.simulation == "online_fixed":
        clf = OnlineLearnerModeler(data.init_train_dat)


    with open(args.out_file, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()

