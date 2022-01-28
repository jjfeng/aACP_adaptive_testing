#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataset import DataGenerator
from model_developers import *


def parse_args():
    parser = argparse.ArgumentParser(description="create model developer for generating algorithmic modifications")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--simulation", type=str, default="online", choices=["adversary", "online", "online_fixed"])
    parser.add_argument("--model-type", type=str, default="Logistic", choices=["Logistic", "GBT", "SelectiveLogistic"])
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    # ONLY RELEVANT TO ADVERSARIAL DEVELOPER
    parser.add_argument("--sparse-p", type=int, default=4, help="number of nonzero coefficients in the true logistic regression model")
    parser.add_argument("--p", type=int, default=10, help="number of covariates")
    parser.add_argument("--sparse-beta", type=float, default=0.5, help="values for the nonzero coefficients in the true model")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create model
    if args.simulation == "adversary":
        assert args.model_type == "Logistic"
        true_beta = np.zeros((args.p, 1))
        true_beta[: args.sparse_p] = args.sparse_beta
        data_generator = DataGenerator(true_beta, mean_x=0)
        clf = BinaryAdversaryModeler(data_generator)
    elif args.simulation == "online_fixed":
        if args.model_type != "SelectiveLogistic":
            clf = OnlineFixedSensSpecModeler(args.model_type, init_sensitivity=0.6, init_specificity=0.6)
        else:
            clf = OnlineFixedSelectiveModeler(args.model_type, target_acc=0.8, init_accept=0.4, incr_accept=0.05)
    elif args.simulation == "online":
        clf = OnlineAdaptiveLearnerModeler(args.model_type, start_side_batch=False)
    else:
        raise NotImplementedError("modeler missing")

    with open(args.out_file, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
