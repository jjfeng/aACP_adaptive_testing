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
    parser = argparse.ArgumentParser(description='create dp mechanism')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--dp-mech',
        type=str,
        default="no_dp",
        choices=["no_dp"])
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/dp_mech.pkl")
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
    np.random.seed(args.seed)
    logging.info(args)

    # Create DP mech
    if args.dp_mech == "no_dp":
        dp_mech = NoDP()

    with open(args.out_file, "wb") as f:
        pickle.dump(dp_mech, f)

if __name__ == "__main__":
    main()

