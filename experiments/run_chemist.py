"""
Adapted from demo_nas.py
@author: kkorovin@cs.cmu.edu

TODO:
* most of TODO-s are in chemist_opt.gp_bandit
* visualization in mols.visualize
* look into dragonfly gp_bandit:get_all_cp_gp_bandit_args and make own combination:

def get_chemist_options(acq_opt_method, ...):
    '''(maybe put this in chemist.py)'''
    dflt_list_of_options = get_all_cp_gp_bandit_args()
    options = load_options(dflt_list_of_options,
                           reporter=reporter)
    options.acq_opt_method = acq_opt_method  # e.g. 'rand_explorer'
    # ... others: e.g. init_capital ... #
    return options

"""

from argparse import Namespace
import time
import os
import shutil
import logging

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from chemist_opt.chemist import Chemist #, get_chemist_options  <--- TODO
from chemist_opt.mol_function_caller import MolFunctionCaller
from mols.mol_functions import get_objective_by_name

# if molecular visualization is implemented, use it
try:
    # TODO: this function should plot a molecule and draw a synthesis plan
    from mols.visualize import visualize_mol
except ImportError as e:
    visualize_mol = None

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
MOL_DATA_DIR = 'datasets'

EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
RUN_LOG_FILE = os.path.join(EXP_DIR, 'run_log')
LOGGING_LEVEL = logging.INFO

DATASET = "chembl" # TODO: use it in - ?
N_WORKERS = 1
OBJECTIVE = "qed"
BUDGET = 10


# Runner ----------------------------------------------------------------------
def main():
    # Make directories
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.mkdir(EXP_DIR)

    logging.basicConfig(filename=RUN_LOG_FILE, filemode='w',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=LOGGING_LEVEL)

    # Obtain a reporter
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))

    objective_func = get_objective_by_name(OBJECTIVE)
    func_caller = MolFunctionCaller(objective_func, config=None, reporter=reporter)
    worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS, time_distro='const')
    data_params = Namespace(data_dir=MOL_DATA_DIR, dataset=DATASET)

    chemist_args = {'acq_opt_method': 'rand_explorer', 'init_capital': 10}
    chemist = Chemist(func_caller, worker_manager, data_source=data_params,
                      chemist_args=chemist_args, is_mf=False, reporter=reporter)
    opt_val, opt_point, history = chemist.run(BUDGET)

    # convert to raw format
    raw_opt_point = func_caller.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    reporter.writeln(f'\nOptimum value found: {opt_val}')
    reporter.writeln(f'Optimum molecule: {opt_mol}')
    reporter.writeln(f'Synthesis path: {opt_mol.get_synthesis_path()}')

    if visualize_mol is not None:
        visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
        reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
        visualize_mol(opt_mol, visualize_file)
    else:
        reporter.writeln('\nMolecule visualization not yet implemented.')

if __name__ == "__main__":
    main()

