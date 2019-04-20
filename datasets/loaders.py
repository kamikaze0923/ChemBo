"""
Loading and handling chemical data
Author: kkorovin@cs.cmu.edu

This is a poorly structured module and needs re-thinking.

TODO:
* Is it possible to set random seed only for one object?
  a bad solution: use random.getstate() and random.setstate()
  before and after sampling calls in MolSampler, so that the resulting
  sampling sequences are the same.

"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from mols.molecule import Molecule

# Class used in CartesianGP
class MolSampler:
    def __init__(self, dataset="chembl", seed=42):
        # load the dataset
        logging.info("Creating a MolSampler")
        self.rnd_generator = None
        if seed is not None:
            self.rnd_generator = np.random.RandomState(seed)
        if dataset == "chembl":
            self.dataset = get_chembl()

    def __call__(self, n_samples):
        if self.rnd_generator is not None:
            ret = list(self.rnd_generator.choice(self.dataset, n_samples))
        else:
            ret = list(np.random.choice(self.dataset, n_samples))
        return ret

# Helper utilities
def get_chembl_prop(n_mols=None, as_mols=False):
    """ Returns (pool, smile->prop mappings) """
    path = "./datasets/ChEMBL_prop.txt"
    df = pd.read_csv(path, sep="\t", header=None)
    # smile: v for the first of two properties
    smile_to_prop = {s: v for (s, v) in zip(df[0], df[1])}
    smile_to_prop = defaultdict(int, smile_to_prop)
    return df[0].values, smile_to_prop

def get_chembl(n_mols=None, as_mols=True):
    """ Return list of SMILES """
    path = "./datasets/ChEMBL.txt"
    with open(path, "r") as f:
        if n_mols is None:
            res = [line.strip() for line in f]
        else:
            res = [f.readline().strip() for _ in range(n_mols)]
    return [Molecule(smile) for smile in res]

def get_initial_pool():
    """Used in chemist_opt.chemist"""
    return get_chembl(10)


if __name__ == "__main__":
    samp = MolSampler()
    print(samp(3))
