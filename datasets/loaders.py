"""
Loading and handling chemical data
Author: kkorovin@cs.cmu.edu

This is a poorly structured module and needs re-thinking.
"""

import numpy as np
import pandas as pd
import logging
import os
from collections import defaultdict
from mols.molecule import Molecule

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Class used in CartesianGP
class MolSampler(object):
    def __init__(self, dataset="chembl", sampling_seed=None):
        """
        Keyword Arguments:
            dataset {str} -- dataset to sample from (default: {"chembl"})
            sampling_seed {int or None} -- (default: {None})
        
        Raises:
            ValueError -- if invalid dataset name is passed.
        """
        # load the dataset
        logging.info(f"Creating a MolSampler from dataset {dataset}")
        self.rnd_generator = None
        if sampling_seed is not None:
            self.rnd_generator = np.random.RandomState(sampling_seed)
        if dataset == "chembl":
            self.dataset = get_chembl()
        elif dataset == "zinc250":
            self.dataset = get_zinc250()
        else:
            raise ValueError(f"Dataset {dataset} not supported.")

    def __call__(self, n_samples):
        if self.rnd_generator is not None:
            ret = list(self.rnd_generator.choice(self.dataset, n_samples))
        else:
            ret = list(np.random.choice(self.dataset, n_samples))
        return ret


# Helper utilities ------------------------------------------------------------
def get_chembl_prop(n_mols=None, as_mols=False):
    """ Returns (pool, smile->prop mappings) """
    path = os.path.join(__location__, "ChEMBL_prop.txt")
    df = pd.read_csv(path, sep="\t", header=None)
    # smile: v for the first of two properties
    smile_to_prop = {s: v for (s, v) in zip(df[0], df[1])}
    smile_to_prop = defaultdict(int, smile_to_prop)
    return df[0].values, smile_to_prop

def get_chembl(n_mols=None, as_mols=True):
    """ 
        Return list of SMILES.
        NOTE: this function should be located
        in the same directory as data files.
    """
    path = os.path.join(__location__, "ChEMBL.txt")
    with open(path, "r") as f:
        if n_mols is None:
            res = [line.strip() for line in f]
        else:
            res = [f.readline().strip() for _ in range(n_mols)]
    return [Molecule(smile) for smile in res]

def get_zinc250():
    path = os.path.join(__location__, "zinc250k.csv")
    zinc_df = pd.read_csv(path)
    list_of_smiles = list(map(lambda x: x.strip(), zinc_df.smiles.values))
    # other columns are logP, qed, and sas
    return [Molecule(smile) for smile in list_of_smiles]

def get_initial_pool():
    """Used in chemist_opt.chemist"""
    return get_chembl(10)


if __name__ == "__main__":
    samp = MolSampler()
    print(samp(3))
