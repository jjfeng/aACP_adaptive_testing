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

from dp_mechanisms import *


def parse_args():
    parser = argparse.ArgumentParser(description="create dp mechanism")
    parser.add_argument("--dp-mech", type=str, default="no_dp")
    # choices=["no_dp", "binary_thres_dp", "bonferroni", "graphical_bonf", "graphical_ffs", "graphical_similarity"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for testing modifications",
    )
    parser.add_argument(
        "--parallel-ratio", type=float, default=1.0, help="parallel factor"
    )
    parser.add_argument(
        "--loss-to-diff-ratios",
        type=str,
        help="ratio in modeling assumptions, comma separated",
    )
    parser.add_argument(
        "--success-weight", type=float, default=0.8, help="recycling factor"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="ci alpha")
    parser.add_argument("--scratch-file", type=str, default="_output/scratch.txt")
    parser.add_argument("--out-file", type=str, default="_output/dp_mech.pkl")
    args = parser.parse_args()
    args.loss_to_diff_ratios = list(map(float, args.loss_to_diff_ratios.split(",")))
    return args


def main():
    args = parse_args()

    # Create DP mech
    if args.dp_mech == "no_dp":
        dp_mech = NoDP()
    elif args.dp_mech == "binary_thres_dp":
        dp_mech = BinaryThresholdDP(args.threshold, args.alpha)
    elif args.dp_mech == "bonferroni":
        dp_mech = BonferroniThresholdDP(args.threshold, args.alpha)
    elif args.dp_mech == "bonferroni_nonadapt":
        dp_mech = BonferroniNonAdaptDP(args.threshold, args.alpha)
    elif args.dp_mech == "graphical_bonf":
        dp_mech = GraphicalBonfDP(
            args.threshold, args.alpha, success_weight=args.success_weight
        )
    elif args.dp_mech == "graphical_parallel":
        dp_mech = GraphicalParallelDP(
            args.threshold,
            args.alpha,
            success_weight=args.success_weight,
            parallel_ratio=args.parallel_ratio,
            loss_to_diff_std_ratios=args.loss_to_diff_ratios,
            scratch_file=args.scratch_file,
        )
    elif args.dp_mech == "graphical_ffs":
        dp_mech = GraphicalFFSDP(
            args.threshold,
            args.alpha,
            success_weight=args.success_weight,
            scratch_file=args.scratch_file,
        )

    with open(args.out_file, "wb") as f:
        pickle.dump(dp_mech, f)


if __name__ == "__main__":
    main()
