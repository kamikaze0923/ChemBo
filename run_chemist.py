"""

Adapted from demo_nas.py
@author: kkorovin@cs.cmu.edu

TODO:
* requires bo_from_func_caller
* requires MolFunctionCaller

"""

from argparse import Namespace
import time
import os
import shutil

# dragonfly imports
from dragonfly.opt.gp_bandit import bo_from_func_caller
# TODO: instead, would be
# from chemist.chemist_opt.chemist import optimize_chemist

from dragonfly.exd.worker_manager import RealWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from mol_function_caller import MolFunctionCaller
# if molecular visualization is implemented, use it
try:
    # this function should plot a molecule
    # and draw a synthesis plan
    from mol.visualize import visualize_mol
except ImportError as e:
    print(e)
    visualize_mol = None


DATASET = "" # TODO

GPU_IDS = []

# Config file which specifies the domain
MOL_DATA_DIR = 'cifar-10-data'  # some analog of this
MOL_CONFIG_FILE = 'config_mol.json'

# Where to store temporary model checkpoints
EXP_DIR = 'experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
LOG_FILE = os.path.join(EXP_DIR, 'log')
TMP_DIR = './tmp_' + DATASET


# Specify the budget (in seconds) -- this is 8 hours
BUDGET = 8 * 60 * 60


# Runner ----------------------------------------------------------------------
def main():
    # Make directories
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)
    os.mkdir(TMP_DIR)
    os.mkdir(EXP_DIR)

    # Obtain a reporter
    reporter = get_reporter(open(LOG_FILE, 'w'))

    train_params = Namespace(data_dir=MOL_DATA_DIR)
    func_caller = MolFunctionCaller(MOL_CONFIG_FILE, train_params, reporter=reporter,
                                    tmp_dir=TMP_DIR)

    worker_manager = RealWorkerManager(GPU_IDS, EXP_DIR)
    opt_val, opt_point, _ = bo_from_func_caller(func_caller, worker_manager, BUDGET,
                                                is_mf=False,  # not multifidelity for now
                                                reporter=reporter)

    # convert to raw format
    raw_opt_point = func_caller.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0] # Because first index in the config file is the neural net.
    # Print the optimal value and visualise the best network.
    reporter.writeln('\nOptimum value found: %0.5f'%(opt_val))
    
    if visualize_mol is not None:
        visualize_file = os.path.join(EXP_DIR, 'optimal_molecule')
        reporter.writeln('Optimal molecule visualized in %s.eps.'%(visualize_file))
        visualize_mol(opt_mol, visualize_file)
    else:
        # TODO
        reporter.writeln('Install graphviz (pip install graphviz) to visualize the molecule.')

if __name__ == "__main__":
    main()

