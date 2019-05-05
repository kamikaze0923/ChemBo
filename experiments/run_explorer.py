"""
Run pure exploration.
@author: kkorovin@cs.cmu.edu

This experiment is both for comparison against Chemist,
and for validation of explored output.

"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from explorer.mol_explorer import RandomExplorer
from mols.mol_functions import get_objective_by_name
from datasets.loaders import get_chembl
from dragonfly.utils.reporters import get_reporter
from synth.validators import compute_min_sa_score, check_validity
from datasets.loaders import MolSampler

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/rand_exp_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
PLOT_FILE = os.path.join(EXP_DIR, 'explorer.png')
if os.path.exists(EXP_DIR):
    shutil.rmtree(EXP_DIR)
os.mkdir(EXP_DIR)


def explore_and_validate_synth(init_pool_size, seed, n_steps, objective, reporter):
    """
    This experiment is equivalent to unlimited-evaluation optimization.
    It compares optimal found vs optimal over pool, and checks if synthesizeability is improved.
    """
    obj_func = get_objective_by_name(objective)
    sampler = MolSampler("chembl", sampling_seed=seed)
    pool = sampler(init_pool_size)
    exp = RandomExplorer(obj_func, initial_pool=pool)

    props = [obj_func(mol) for mol in pool]
    reporter.writeln(f"Properties of pool: quantity {len(pool)}, min {np.min(props)}, avg {np.mean(props)}, max {np.max(props)}")
    reporter.writeln(f"Starting {objective} optimization")
    top_value, top_point, history = exp.run(n_steps)

    reporter.writeln(f"Is a valid molecule: {check_validity(top_point)}")
    reporter.writeln(f"Top score: {obj_func(top_point)}")
    reporter.writeln(f"Minimum synthesis score over the path: {compute_min_sa_score(top_point)}")

    sorted_by_prop = sorted(pool, key=obj_func)[-5:]
    for opt_mol in sorted_by_prop:
        min_sa_score = compute_min_sa_score(opt_mol)
        reporter.writeln(f"Minimum synthesis score of optimal molecules: {min_sa_score}")

    vals = history['objective_vals']
    plt.title(f'Optimizing {objective} with random explorer')
    plt.plot(range(len(vals)), vals)
    plt.savefig(PLOT_FILE)


if __name__ == "__main__":
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
    exp_settings = {'init_pool_size': 10, 'seed': 7, 'n_steps': 100, 'objective': 'qed'}
    reporter.writeln(f"RandomExplorer experiment settings: objective {exp_settings['objective']}," +
                     f"init pool of size {exp_settings['init_pool_size']}," +
                     f"seed {exp_settings['seed']}, budget {exp_settings['n_steps']}")
    explore_and_validate_synth(**exp_settings, reporter=reporter)

