from typing import List

import pandas as pd
import numpy as np

from node import Node

def _get_predictions(mdl, x):
    return mdl.predict(x).reshape((-1, 1))


class HypothesisTester:
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat

    def get_observations(self, mdl):
        raise NotImplementedError

    def test_null(self, node: Node, null_hypo: np.ndarray, prior_nodes: List):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        raise NotImplementedError()

class SensSpecHypothesisTester(HypothesisTester):
    def set_test_dat(self, test_dat):
        self.test_dat = test_dat
        self.orig_df = pd.DataFrame({
            "pos": self.test_dat.y.flatten(),
            "neg": (1 - self.test_dat.y).flatten(),
            })

    def get_observations(self, mdl):
        pred_y = _get_predictions(mdl, self.test_dat.x)
        df = pd.DataFrame({
            "equal_pos": ((self.test_dat.y == pred_y) * self.test_dat.y).flatten(),
            "equal_neg": ((self.test_dat.y == pred_y) * (1 - self.test_dat.y)).flatten()
            })
        return df

    def test_null(self, node: Node, null_hypo: np.ndarray, prior_nodes: List = []):
        """
        @return the CI code this node, accounting for the previous nodes in the true
               and spending only the alpha allocated at this node
        """
        assert len(prior_nodes) == 0
        est = np.array([
            node.obs.equal_pos.mean()/self.orig_df.pos.mean(),
            node.obs.equal_neg.mean()/self.orig_df.neg.mean()
            ])
        print("EST sens, spec", est)
        1/0
