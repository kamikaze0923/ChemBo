"""
Adapted from demo_nas.py
@author: kkorovin@cs.cmu.edu

TODO:
* most of TODO-s are in chemist_opt.gp_bandit
* visualization in mols.visualize

"""

from argparse import Namespace
import time
import os
import shutil

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from chemist_opt.chemist import Chemist
from chemist_opt.mol_function_caller import MolFunctionCaller
from mols.mol_functions import get_objective_by_name

# if molecular visualization is implemented, use it
try:
    # TODO: this function should plot a molecule and draw a synthesis plan
    from mols.visualize import visualize_mol
except ImportError as e:
    visualize_mol = None


DATASET = "" # TODO

# data directory
MOL_DATA_DIR = 'datasets'

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
LOG_FILE = os.path.join(EXP_DIR, 'log')

N_WORKERS = 1
BUDGET = 5


# Runner ----------------------------------------------------------------------
def main():
    # Make directories
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.mkdir(EXP_DIR)

    # Obtain a reporter
    reporter = get_reporter(open(LOG_FILE, 'w'))

    objective_func = get_objective_by_name("logp")  # just a function
    func_caller = MolFunctionCaller(objective_func,
                                    config=None,
                                    reporter=reporter)
    worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS, time_distro='const')
    data_params = Namespace(data_dir=MOL_DATA_DIR, dataset=DATASET)

    chemist = Chemist(func_caller, worker_manager, data_source=data_params,
                      is_mf=False, reporter=reporter)
    opt_val, opt_point, _ = chemist.run(BUDGET)

    # convert to raw format
    raw_opt_point = func_caller.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    reporter.writeln('\nOptimum value found: %0.5f'%(opt_val))
    reporter.writeln('Optimum molecule: %s'%(opt_mol))
    reporter.writeln('Synthesis path: %s'%opt_mol.get_synthesis_path())

    if visualize_mol is not None:
        visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
        reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
        visualize_mol(opt_mol, visualize_file)
    else:
        reporter.writeln('\nMolecule visualization not yet implemented.')

if __name__ == "__main__":
    main()

