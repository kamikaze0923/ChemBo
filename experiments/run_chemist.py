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
import logging
import tensorflow as tf

# dragonfly imports
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.utils.reporters import get_reporter

# a few local imports here
from chemist_opt.chemist import Chemist
from chemist_opt.mol_function_caller import MolFunctionCaller
from mols.mol_functions import get_objective_by_name
from mols.visualize import visualize_mol
from dist.ot_dist_computer import OTChemDistanceComputer
from mols.mol_kernels import MOL_DISTANCE_KERNEL_TYPES

# Where to store temporary model checkpoints
EXP_DIR = 'experiments/experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
MOL_DATA_DIR = 'datasets'

EXP_LOG_FILE = os.path.join(EXP_DIR, 'exp_log')
RUN_LOG_FILE = os.path.join(EXP_DIR, 'run_log')
LOGGING_LEVEL = logging.INFO
TF_LOGGING_LEVEL = tf.logging.ERROR

DATASET = "chembl" # TODO: use it in - ?
N_WORKERS = 1
OBJECTIVE = "qed"
BUDGET = 20


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
    tf.logging.set_verbosity(TF_LOGGING_LEVEL)

    # Obtain a reporter
    reporter = get_reporter(open(EXP_LOG_FILE, 'w'))

    objective_func = get_objective_by_name(OBJECTIVE)
    func_caller = MolFunctionCaller(objective_func, config=None, reporter=reporter)
    worker_manager = SyntheticWorkerManager(num_workers=N_WORKERS, time_distro='const')
    data_params = Namespace(data_dir=MOL_DATA_DIR, dataset=DATASET)

    chemist_args = {
        'acq_opt_method': 'rand_explorer',
        'init_capital': 10,
        'dom_mol_kernel_type': 'wl_kernel'
    }

    domain_dist_computers = []
    for domain, kernel_type in zip(func_caller.domain.list_of_domains, func_caller.domain_orderings.kernel_ordering):
        domain_type = domain.get_type()
        if domain_type == "molecule":
            # TODO: kernel resolve order
            if kernel_type is None or '':
                kernel_type = chemist_args["dom_mol_kernel_type"]
            if kernel_type == "default":
                pass
            if kernel_type in MOL_DISTANCE_KERNEL_TYPES:
                computer = OTChemDistanceComputer()
                domain_dist_computers.append(computer.evaluate)
            else:
                domain_dist_computers.append(None)
        else:
            raise NotImplementedError("distance computers not implemented for other domains")
    print(f"domain_dist_computers: {domain_dist_computers}")

    chemist = Chemist(
        func_caller, worker_manager,
        data_source=data_params,
        chemist_args=chemist_args,
        is_mf=False, reporter=reporter,
        domain_dist_computers=domain_dist_computers
    )
    opt_val, opt_point, history = chemist.run(BUDGET)

    # convert to raw format
    raw_opt_point = func_caller.get_raw_domain_point_from_processed(opt_point)
    opt_mol = raw_opt_point[0]

    # Print the optimal value and visualize the molecule and path.
    reporter.writeln(f'\nOptimum value found: {opt_val}')
    reporter.writeln(f'Optimum molecule: {opt_mol}')
    reporter.writeln(f'Synthesis path: {opt_mol.get_synthesis_path()}')

    # visualize mol/synthesis path
    visualize_file = os.path.join(EXP_DIR, 'optimal_molecule.png')
    reporter.writeln(f'Optimal molecule visualized in {visualize_file}')
    visualize_mol(opt_mol, visualize_file)

if __name__ == "__main__":
    main()

