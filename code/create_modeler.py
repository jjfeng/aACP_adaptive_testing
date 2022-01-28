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
    parser = argparse.ArgumentParser(description="create model developer for generating algorithmic modifications")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--simulation", type=str, default="online", choices=["adversary", "online", "online_fixed"])
    parser.add_argument("--model-type", type=str, default="Logistic", choices=["Logistic", "GBT", "SelectiveLogistic"])
    parser.add_argument("--min-var-idx", type=int, default=0, help="What index to start perturbing coefficients in the adversarial model developer")
    parser.add_argument("--preset-coef", type=float, default=0, help="What is the true value of the nonzero coefficients, used for adversarial model developer")
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create model
    if args.simulation == "adversary":
        clf = BinaryAdversaryModeler(
            min_var_idx=args.min_var_idx, preset_coef=args.preset_coef
        )
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
