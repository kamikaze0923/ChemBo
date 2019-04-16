"""
Class that performs molecule space traversal.
@author: kkorovin@cs.cmu.edu

TODO:
* add function evaluation counting
* better handling of fitness_func arguments
  (lists vs args etc.)

"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from time import time

from synth.forward_synth import RexgenForwardSynthesizer
from rdkit import Chem
from rdkit_contrib.sascorer import calculateScore as calculateSAScore
from mols.molecule import Molecule, Reaction
from datasets.loaders import get_chembl_prop, get_initial_pool


class Explorer:
    def __init__(self):
        pass
    def evolve(self):
        pass

class RandomExplorer(Explorer):
    """
    Implements a random evolutionary algorithm
    for exploring molecule space.
    """
    def __init__(self, fitness_func, capital_type='return_value',
                 initial_pool=None, max_pool_size=None):
        """
        Params:
        :fitness_func: function to optimize over evolution
        :capital_type: number of steps or other cost of exploration
        :initial_pool: just what it says
        :max_pool_size: int or None
        TODO:
        :mutation_op: mutates a given Molecule
        :crossover_op: takes two Molecules
                    and returns one new Molecule
        """
        self.fitness_func = fitness_func
        self.capital_type = capital_type
        self.synth = RexgenForwardSynthesizer()
        if initial_pool is None:
            initial_pool = get_initial_pool()
        self.pool = initial_pool
        self.max_pool_size = max_pool_size

        # TODO: think whether to add additional *synthesized* pool

    def evolve_step(self):
        # choose molecules to cross-over
        r_size = np.random.randint(2,3)
        mols = np.random.choice(self.pool, size=r_size)

        # evolve
        reaction = Reaction(mols)
        outcomes = self.synth.predict_outcome(reaction)
        top_outcome = sorted(outcomes, key=lambda mol: self.fitness_func([mol]))[-1]
        print("Newly generated mol value:", self.fitness_func([top_outcome]))
        self.pool.append(top_outcome)

        # filter
        if self.max_pool_size is not None:
            self.pool = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-self.max_pool_size:]

    def evolve(self, capital):
        """
        Params:
        :data: start dataset (list of Molecules)
        :capital: number of steps or other cost of exploration
        """
        # for step-count capital
        if self.capital_type == 'return_value':
            capital = int(capital)
            for _ in range(capital):
                self.evolve_step()
        else:
            raise NotImplementedError(f"Capital type {self.capital_type} not implemented.")

    def get_best(self, k):
        top = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-k:]
        return top
