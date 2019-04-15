"""
Run pure exploration.
This experiment is both for comparison against Chemist,
and for validation of explored output.

@author: kkorovin@cs.cmu.edu
"""

from time import time
import numpy as np

from explore.mol_explorer import RandomExplorer
from mols.mol_functions import get_objective_by_name
from datasets.loaders import get_chembl
from synth.validators import compute_min_sa_score, check_validity


def explore_and_validate_synth(n_steps):
    """
    This experiment is equivalent to unlimited-evaluation optimization.
    It compares optimal found vs optimal over pool, and checks if synthesizeability is improved.
    """

    sas_func = get_objective_by_name("logp")
    pool = get_chembl(1000)
    exp = RandomExplorer(sas_func, initial_pool=pool)

    props = [sas_func(mol) for mol in pool]
    print(f"Properties of pool: quantity {len(pool)}, min {np.min(props)}, avg {np.mean(props)}, max {np.max(props)}")
    print("Starting LogP optimization")
    exp.evolve(n_steps)

    top = exp.get_best(1)[0]
    print(f"Is a valid molecule: {check_validity(top)}")
    print(f"Top score: {sas_func(top)}")
    print(f"Minimum synthesis score over the path: {compute_min_sa_score(top)}")

    sorted_by_prop = sorted(pool, key=sas_func)[-5:]
    for opt_mol in sorted_by_prop:
        min_sa_score = compute_min_sa_score(opt_mol)
        print(f"Minimum synthesis score of optimal molecules: {min_sa_score}")


if __name__ == "__main__":
    explore_and_validate_synth(3)