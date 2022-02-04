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
from create_mtp_mechanism import get_hypo_tester


def parse_args():
    parser = argparse.ArgumentParser(description="create model developer for generating algorithmic modifications")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--se-factor", type=float, default=0.1, help="se factor for estimating perf")
    parser.add_argument("--alpha", type=float, default=0.1, help="assumed alpha for adaptive testing")
    parser.add_argument("--power", type=float, default=0.3, help="desired power for testing")
    parser.add_argument("--update-incr", type=float, default=0.1, help="how much the adversary perturbs things")
    parser.add_argument("--simulation", type=str, default="online", choices=["adversary", "online_delta", "online_compare", "online_compare_adapt", "online_compare_restrict"])
    parser.add_argument("--hypo-tester", type=str, default="auc", choices=["log_lik", "auc", "accuracy"])
    parser.add_argument("--model-type", type=str, default="Logistic", choices=["Logistic", "GBT", "SelectiveLogistic"])
    parser.add_argument("--scratch-file", type=str, default="_output/scratch.txt")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    # ONLY RELEVANT TO ADVERSARIAL DEVELOPER
    parser.add_argument("--sparse-p", type=int, default=4, help="number of nonzero coefficients in the true logistic regression model")
    parser.add_argument("--p", type=int, default=10, help="number of covariates")
    parser.add_argument("--sparse-beta", type=float, default=0.5, help="values for the nonzero coefficients in the true model")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="number of obs used in validation")
    parser.add_argument("--min-valid-dat-size", type=int, default=200, help="number of observations to hold out for validation")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    hypo_tester = get_hypo_tester(args.hypo_tester, args.scratch_file)
    # Create model
    clf = None
    if args.simulation == "adversary":
        assert args.model_type == "Logistic"
        true_beta = np.zeros((args.p, 1))
        true_beta[: args.sparse_p] = args.sparse_beta
        data_generator = DataGenerator(true_beta, mean_x=0)
        #clf = BinaryAdversaryModeler(data_generator)
        clf = AdversaryLossModeler(data_generator, update_incr=args.update_incr)
    elif args.simulation == "online_delta":
        if args.model_type == "Logistic":
            clf = OnlineAdaptLossModeler(hypo_tester, min_valid_dat_size=args.min_valid_dat_size, predef_alpha=args.alpha, power=args.power, se_factor=args.se_factor)
        elif args.model_type == "SelectiveLogistic":
            clf = OnlineFixedSelectiveModeler(args.model_type, target_acc=0.8, init_accept=0.4, incr_accept=0.05)
    elif args.simulation == "online_compare_restrict":
        clf = OnlineAdaptRestrictCompareModeler(hypo_tester, min_valid_dat_size=args.min_valid_dat_size, predef_alpha=args.alpha, power=args.power, se_factor=args.se_factor)
    elif args.simulation == "online_delta_restrict":
        clf = OnlineAdaptRestrictLossModeler(hypo_tester, min_valid_dat_size=args.min_valid_dat_size, predef_alpha=args.alpha, power=args.power, se_factor=args.se_factor)
    elif args.simulation == "online_compare":
        if args.model_type == "Logistic":
            clf = OnlineAdaptCompareModeler(hypo_tester, min_valid_dat_size=args.min_valid_dat_size, validation_frac=args.valid_frac, predef_alpha=args.alpha, power=args.power, se_factor=args.se_factor)
    elif args.simulation == "online_compare_adapt":
        if args.model_type == "Logistic":
            clf = OnlineAdaptCompareSideModeler(hypo_tester, min_valid_dat_size=args.min_valid_dat_size, validation_frac=args.valid_frac, predef_alpha=args.alpha, power=args.power, se_factor=args.se_factor)
    elif args.simulation == "online":
        if args.model_type != "SelectiveLogistic":
            clf = OnlineSensSpecModeler(args.model_type, min_valid_dat_size=args.min_valid_dat_size)

    if clf is None:
        raise NotImplementedError("modeler missing")

    with open(args.out_file, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
