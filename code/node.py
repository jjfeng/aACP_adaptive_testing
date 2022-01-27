import numpy as np

class Node:
    """
    Node in the graph for sequentially rejective graphical procedure (SRGP)
    """
    def __init__(self, weight, history, parent=None):
        """
        @param subfam_root: which node is the subfamily's root node. if none, this is the root
        """
        self.weight = weight
        self.history = history
        self.parent = parent
        self.children = []
        self.children_weights = []

    def store_observations(self, obs: np.ndarray):
        self.obs = obs
