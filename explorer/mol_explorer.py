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

    def evolve(self, capital):
        """ Main method of Explorers.

        Arguments:
            capital - for how long to run
        Returns:
            opt_value, opt_point, history
        """
        pass

class RandomExplorer(Explorer):
    """
    Implements a random evolutionary algorithm
    for exploring molecule space.
    """
    def __init__(self, fitness_func, capital_type='return_value',
                 initial_pool=None, max_pool_size=None,
                 n_outcomes=1):
        """
        Params:
            fitness_func {function} - objective to optimize over evolution
            capital_type {int/float} - number of steps or other cost of exploration
            initial_pool {list} - just what it says
            max_pool_size {int or None} - whether to keep the pool to top k most fit
            n_outcomes {int} - # of most likely reaction outcomes to keep and evaluate
        """
        self.fitness_func = fitness_func
        self.capital_type = capital_type
        self.synth = RexgenForwardSynthesizer()
        if initial_pool is None:
            initial_pool = get_initial_pool()
        self.pool = initial_pool
        self.max_pool_size = max_pool_size
        self.n_outcomes = n_outcomes
        # TODO: think whether to add additional *synthesized* pool

    def evolve_step(self):
        # choose molecules to cross-over
        r_size = np.random.randint(2,3)
        mols = np.random.choice(self.pool, size=r_size)

        # evolve
        reaction = Reaction(mols)
        outcomes = self.synth.predict_outcome(reaction, k=self.n_outcomes)

        if self.n_outcomes == 1:
            top_pt = outcomes[0]
            top_val = self.fitness_func([top_pt])
        else:
            top_pt = sorted(outcomes, key=lambda mol: self.fitness_func([mol]))[-1]
            top_val = self.fitness_func(top_pt)
        self.pool.append(top_pt)

        # filter
        if self.max_pool_size is not None:
            self.pool = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-self.max_pool_size:]

        # update history
        if self.history['objective_vals']:
            top_value = max(top_val, self.history['objective_vals'][-1])
            self.history['objective_vals'].append(top_value)

    def evolve(self, capital):
        """
        Params:
        :data: start dataset (list of Molecules)
        :capital: number of steps or other cost of exploration
        """
        self._initialize_history()
        # for step-count capital
        if self.capital_type == 'return_value':
            capital = int(capital)
            for _ in range(capital):
                self.evolve_step()
        else:
            raise NotImplementedError(f"Capital type {self.capital_type} not implemented.")

        top_pt = self.get_best(k=1)[0]
        top_val = self.fitness_func([top_pt])
        return top_val, top_pt, self.history

    def get_best(self, k):
        top = sorted(self.pool, key=lambda mol: self.fitness_func([mol]))[-k:]
        return top

    def _initialize_history(self):
        n_init = len(self.pool)
        max_over_pool = np.max([self.fitness_func([mol]) for mol in self.pool])
        self.history = {
                        'objective_vals': [max_over_pool] * n_init
                        }

