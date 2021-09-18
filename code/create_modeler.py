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
    parser = argparse.ArgumentParser(description="run simulation")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--simulation", type=str, default="online_fixed")
    parser.add_argument("--model-type", type=str, default="Logistic")
    parser.add_argument("--min-var-idx", type=int, default=0)
    parser.add_argument("--preset-coef", type=float, default=0)
    parser.add_argument("--out-file", type=str, default="_output/model.pkl")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create model
    if args.simulation == "neldermead":
        clf = NelderMeadModeler(min_var_idx=args.min_var_idx)
    elif args.simulation == "adversary":
        clf = AdversarialModeler(
            min_var_idx=args.min_var_idx, preset_coef=args.preset_coef
        )
    elif args.simulation == "online":
        clf = OnlineAdaptiveLearnerModeler(args.model_type, start_side_batch=False)
    elif args.simulation == "online_bad":
        clf = OnlineAdaptiveLearnerModeler(args.model_type, start_side_batch=True)
    elif args.simulation == "online_fixed":
        clf = OnlineLearnerFixedModeler(args.model_type)
    else:
        raise NotImplementedError("modeler missing")

    with open(args.out_file, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
