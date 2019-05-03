"""
Runner of the Chemist optimization.
Can be used as a usage example.

@author: kkorovin@cs.cmu.edu

TODO:
* visualization in mols.visualize

NOTE:
* all datasets now are assumed to live in the same folder
  as loaders.py (which contains the Sampler and dataset getters it uses)
"""

from myrdkit import *  # :(

from argparse import Namespace
import time
import os
import shutil
import logging
import tensorflow as tf

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from chemist_opt.chemist import Chemist
from mols.mol_functions import get_objective_by_name
from mols.visualize import visualize_mol

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))

EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
RUN_LOG_FILE = os.path.join(EXP_DIR, 'run_log')
LOGGING_LEVEL = logging.INFO
TF_LOGGING_LEVEL = tf.logging.ERROR

DATASET = "chembl"  # chembl or zinc250
N_WORKERS = 1
OBJECTIVE = "qed"
BUDGET = 100

# Create exp directory and point the logger -----------------------------------
def setup_logging():
    # Make directories
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.mkdir(EXP_DIR)

    # necessary fix for setting the logging after some imports
    from imp import reload
    reload(logging)

    logging.basicConfig(filename=RUN_LOG_FILE, filemode='w',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=LOGGING_LEVEL)
    tf.logging.set_verbosity(TF_LOGGING_LEVEL)


# Runner ----------------------------------------------------------------------
def main():
    setup_logging()

    # Obtain a reporter and worker manager
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))
    worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS,
                                            time_distro='const')

    # Problem settings
    objective_func = get_objective_by_name(OBJECTIVE)
    # check MolDomain constructor for full argument list:
    domain_config = {'data_source': DATASET,
                     'constraint_checker': 'organic',  # not specifying constraint_checker defaults to None
                     'sampling_seed': 42}
    chemist_args = {
        'acq_opt_method': 'rand_explorer',
        'init_capital': 10,
        'dom_mol_kernel_type': 'similarity_kernel',  # e.g. 'distance_kernel_expsum', 'similarity_kernel', 'wl_kernel'
        'acq_opt_max_evals' : 10 // N_WORKERS
    }

    chemist = Chemist(
        objective_func,
        domain_config=domain_config,
        chemist_args=chemist_args,
        is_mf=False,
        worker_manager=worker_manager,
        reporter=reporter
    )
    opt_val, opt_point, history = chemist.run(BUDGET)

    # convert to raw format
    raw_opt_point = chemist.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    reporter.writeln(f'\nOptimum value found: {opt_val}')
    reporter.writeln(f'Optimum molecule: {opt_mol} with formula {opt_mol.to_formula()}')
    reporter.writeln(f'Synthesis path: {opt_mol.get_synthesis_path()}')

    # visualize mol/synthesis path
    visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
    reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
    visualize_mol(opt_mol, visualize_file)

if __name__ == "__main__":
    main()

