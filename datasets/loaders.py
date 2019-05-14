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
from mols.mol_functions import get_objective_by_name

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
        if dataset.startswith("chembl"):
            idx = dataset.find('_')
            if idx == -1:
                option = ''
            else:
                option = dataset[idx+1:]
            self.dataset = get_chembl(option=option)
        elif dataset.startswith("zinc250"):
            idx = dataset.find('_')
            if idx == -1:
                option = ''
            else:
                option = dataset[idx+1:]
            self.dataset = get_zinc250(option=option)
        else:
            raise ValueError(f"Dataset {dataset} not supported.")

    def __call__(self, n_samples):
        if self.rnd_generator is not None:
            ret = list(self.rnd_generator.choice(self.dataset, n_samples, replace=False))
        else:
            ret = list(np.random.choice(self.dataset, n_samples, replace=False))
        return ret


# Helper utilities ------------------------------------------------------------
def get_chembl_prop(n_mols=None, as_mols=False):
    """ Returns (pool, smile->prop mappings) """
    path = os.path.join(__location__, "ChEMBL_prop.txt")
    df = pd.read_csv(path, sep="\t", header=None)
    # smile: v for the first of two properties
    smile_to_prop = {s: v for (s, v) in zip(df[0], df[1])}
    smile_to_prop = defaultdict(int, smile_to_prop)
    smile_strings = df[0].values
    if n_mols is not None:
        smile_strings = np.random.choice(smile_strings, n_mols)
    return smile_strings, smile_to_prop

def get_chembl(n_mols=None, as_mols=True, option='', max_size=1000):
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
    mols = [Molecule(smile) for smile in res]
    if len(mols) < max_size:
        return mols

    gen = np.random.RandomState(42)
    mols = list(gen.choice(mols, max_size, replace=False))
    if option == '':
        return mols
    elif option == 'small_qed':
        qed_func = get_objective_by_name("qed")
        return [mol for mol in mols if qed_func(mol) < 0.6]
    elif option == 'large_qed':
        qed_func = get_objective_by_name("qed")
        return [mol for mol in mols if qed_func(mol) >= 0.6]
    else:
        raise ValueError(f"Dataset filter {option} not supported.")

def get_zinc250(option='', max_size=1000):
    path = os.path.join(__location__, "zinc250k.csv")
    zinc_df = pd.read_csv(path)
    list_of_smiles = list(map(lambda x: x.strip(), zinc_df.smiles.values))
    # other columns are logP, qed, and sas
    mols = [Molecule(smile) for smile in res]
    if len(mols) < max_size:
        return mols

    gen = np.random.RandomState(42)
    mols = list(gen.choice(mols, max_size, replace=False))
    if option == '':
        return mols
    elif option == 'small_qed':
        qed_func = get_objective_by_name("qed")
        return [mol for mol in mols if qed_func(mol) < 0.6]
    elif option == 'large_qed':
        qed_func = get_objective_by_name("qed")
        return [mol for mol in mols if qed_func(mol) >= 0.6]
    else:
        raise ValueError(f"Dataset filter {option} not supported.")

def get_initial_pool():
    """Used in chemist_opt.chemist"""
    return get_chembl(10)

def print_pool_statistics(dataset, seed, n=30):
    from mols.mol_functions import get_objective_by_name
    objective = "qed"
    samp = MolSampler(dataset, seed)
    pool = samp(n)
    obj_func = get_objective_by_name(objective)
    props = [obj_func(mol) for mol in pool]
    print(f"Properties of pool: quantity {len(pool)}, min {np.min(props)}, avg {np.mean(props)}, max {np.max(props)}, std {np.std(props)}")


if __name__ == "__main__":
    dataset = "chembl"
    for seed in range(100):
        print('\tSeed: ', seed)
        print_pool_statistics(dataset, seed)

