#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import time
import argparse
import pickle
import logging
import progressbar
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve

from dataset import *

def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--obs-batch-size',
        type=int,
        default=1)
    parser.add_argument(
        '--test-batch',
        type=int,
        default=1)
    parser.add_argument(
        '--maxfev',
        type=int,
        default=100)
    parser.add_argument(
        '--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument(
        '--dp-mech-file',
        type=str,
        default="_output/dp_mech.pkl")
    parser.add_argument(
        '--model-file',
        type=str,
        default="_output/model.pkl")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt")
    args = parser.parse_args()
    return args

def get_nll(test_y, pred_y):
    test_y = test_y.flatten()
    pred_y = pred_y.flatten()
    return -np.mean(test_y * np.log(pred_y) + (1 - test_y) * np.log(1 - pred_y))

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    # parameters
    np.random.seed(args.seed)
    logging.info(args)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]
    print("data done")

    with open(args.dp_mech_file, "rb") as f:
        dp_mech = pickle.load(f)

    with open(args.model_file, "rb") as f:
        modeler = pickle.load(f)

    # Run simulation
    modeler.do_minimize(data.reuse_test_dat.x, data.reuse_test_dat.y, dp_mech, maxfev=args.maxfev)

    reuse_pred_y = modeler.predict_prob(data.reuse_test_dat.x)
    reuse_auc = roc_auc_score(data.reuse_test_dat.y, reuse_pred_y)
    reuse_nll = get_nll(data.reuse_test_dat.y, reuse_pred_y)
    print(dp_mech.__class__, "reuse", "AUC", reuse_auc, "NLL", reuse_nll)

    test_pred_y = modeler.predict_prob(data.test_dat.x)
    test_auc = roc_auc_score(data.test_dat.y, test_pred_y)
    test_nll = get_nll(data.test_dat.y, test_pred_y)
    print(dp_mech.__class__, "test", "AUC", test_auc, "NLL", test_nll)


if __name__ == "__main__":
    main()
