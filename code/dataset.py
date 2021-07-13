import logging

import scipy.stats
import numpy as np
import pandas as pd

from constants import SIM_SETTINGS

def make_safe_prob(p, eps=1e-10):
    return np.maximum(eps, np.minimum(1-eps, p))

class FullDataset:
    def __init__(self, init_train_dat, reuse_test_dat, test_dat, timestamps=None):
        self.init_train_dat = init_train_dat
        self.reuse_test_dat = reuse_test_dat
        self.timestamps = timestamps
        self.test_dat = test_dat

    def get_timestamp(self, t_idx: int):
        if self.timestamps is not None:
            return self.timestamps[t_idx]
        else:
            return t_idx

    def get_accum_dat(self, t_idx: int):
        x = self.reuse_test_dat.x[:t_idx + 1]
        y = self.reuse_test_dat.y[:t_idx + 1]
        return x, y

class Dataset:
    def __init__(self, x, y, mu: np.ndarray =None, weight: np.ndarray=None):
        self.x = x
        self.y = y
        self.mu = mu
        self.weight = weight

    @property
    def size(self):
        return self.x.shape[0]

    def subset_idxs(self, selected_idxs):
        return Dataset(
                x=self.x[selected_idxs,:],
                y=self.y[selected_idxs,:],
                mu=self.mu[selected_idxs,:] if self.mu is not None else None,
                weight=self.weight[selected_idxs,:] if self.weight is not None else None)

    def subset(self, n, start_n=0):
        assert start_n >= 0
        return Dataset(
                x=self.x[start_n:n,:],
                y=self.y[start_n:n,:],
                mu=self.mu[start_n:n,:] if self.mu is not None else None,
                weight=self.weight[start_n:n,:] if self.weight is not None else None)

    @staticmethod
    def merge(datasets, dataset_weights=None):
        """
        @return merged dataset with weights
        """
        has_mu = datasets[-1].mu is not None
        if datasets[-1].weight is not None and dataset_weights is None:
            has_weight = True
            dataset_weights = [1] * len(datasets)
        elif dataset_weights is not None:
            has_weight = True
            for dat in datasets:
                if dat.weight is None:
                    dat.weight = np.ones((dat.size, 1))
        else:
            has_weight = False
        return Dataset(
                x=np.vstack([dat.x for dat in datasets]),
                y=np.vstack([dat.y for dat in datasets]),
                mu=np.vstack([dat.mu for dat in datasets]) if has_mu else None,
                weight=np.vstack([dat.weight * dat_weight for dat, dat_weight in zip(datasets, dataset_weights)]) if has_weight else None)

class DataGenerator:
    def __init__(self, init_beta: np.ndarray, mean_x: np.ndarray):
        self.init_beta = init_beta
        logging.info("init beta %s", init_beta.ravel())
        self.p = init_beta.size
        self.mean_x = mean_x

    def generate_data(self, init_train_n, init_reuse_test_n, test_n: int):
        init_train_dat = self.make_data(init_train_n, self.init_beta)
        reuse_test_dat = self.make_data(init_reuse_test_n, self.init_beta)
        test_dat = self.make_data(test_n, self.init_beta)

        full_dat = FullDataset(init_train_dat, reuse_test_dat, test_dat)

        return full_dat, [self.init_beta]

    def make_data(self, n, beta):
        p = beta.size
        x = np.random.normal(size=(n, p), loc=self.mean_x)

        mu = 1/(1 + np.exp(-(np.matmul(x, beta))))
        y = np.random.binomial(n=1, p=mu, size=(n, 1))

        return Dataset(x, y, mu)

