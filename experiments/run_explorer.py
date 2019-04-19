"""
Run pure exploration.
@author: kkorovin@cs.cmu.edu

This experiment is both for comparison against Chemist,
and for validation of explored output.

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from explorer.mol_explorer import RandomExplorer
from mols.mol_functions import get_objective_by_name
from datasets.loaders import get_chembl
from synth.validators import compute_min_sa_score, check_validity


def explore_and_validate_synth(init_pool_size, n_steps,
                               objective_name='logp'):
    """
    This experiment is equivalent to unlimited-evaluation optimization.
    It compares optimal found vs optimal over pool, and checks if synthesizeability is improved.
    """

    obj_func = get_objective_by_name(objective_name)
    pool = get_chembl(init_pool_size)
    exp = RandomExplorer(obj_func, initial_pool=pool)

    props = [obj_func(mol) for mol in pool]
    print(f"Properties of pool: quantity {len(pool)}, min {np.min(props)}, avg {np.mean(props)}, max {np.max(props)}")
    print(f"Starting {objective_name} optimization")
    top_value, top_point, history = exp.evolve(n_steps)

    print(f"Is a valid molecule: {check_validity(top_point)}")
    print(f"Top score: {obj_func(top_point)}")
    print(f"Minimum synthesis score over the path: {compute_min_sa_score(top_point)}")

    sorted_by_prop = sorted(pool, key=obj_func)[-5:]
    for opt_mol in sorted_by_prop:
        min_sa_score = compute_min_sa_score(opt_mol)
        print(f"Minimum synthesis score of optimal molecules: {min_sa_score}")

    vals = history['objective_vals']
    plt.title(f'Optimizing {objective_name} with random explorer')
    plt.plot(range(len(vals)), vals)
    plt.savefig(f'./experiments/results/explorer_{objective_name}.png')


if __name__ == "__main__":
    explore_and_validate_synth(init_pool_size=5, n_steps=20)
