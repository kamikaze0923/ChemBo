"""
Module for BO with graph kernel and synthesizeable exploration.
@author: kkorovin@cs.cmu.edu

Available Explorers: 'rand_explorer', ...

TODO:
* make a class with options for different explorers,
  starter datasets, etc.

"""

from dragonfly.opt.gp_bandit import get_all_cp_gp_bandit_args
from dragonfly.utils.reporters import get_reporter
from dragonfly.utils.option_handler import load_options
from dragonfly.exd.worker_manager import RealWorkerManager, SyntheticWorkerManager
from dragonfly.exd.exd_utils import get_cp_domain_initial_qinfos
from chemist_opt.gp_bandit import CPGPBandit, get_cp_domain_initial_qinfos


class Chemist:
    def __init__(self):
        raise NotImplementedError


def optimize_chemist(func_caller, worker_manager, max_capital, is_mf=False, mode=None,
                     acq=None, mf_strategy=None,
                     domain_add_max_group_size=-1,
                     domain_dist_computers=None,
                     options=None, reporter='default'):
    """
    :param func_caller:
    :param worker_manager:
    :param max_capital:
    :param is_mf: whether multi-fidelity
    :param mode:
    :param acq: the default acquisition to use, if not None, will override the `options.acq`
    :param mf_strategy:
    :param domain_add_max_group_size:
    :param domain_dist_computers:
        a list of functions for each domain to compute the pairwise distance between two lists of data
        i.e.: for two lists of length $n_1$ and $n_2$ respectively, return a $(n_1, n_2)$ matrix of pair-wise distance
    :param options:
    :param reporter:
    :return:
    """
    optimiser_constructor = CPGPBandit
    dflt_list_of_options = get_all_cp_gp_bandit_args()

    # TODO --------------------------------------------------------------------
    reporter = get_reporter(reporter)
    if options is None:
        options = load_options(dflt_list_of_options, reporter=reporter)
    options.acq_opt_method = 'rand_explorer'
    if acq is not None:
        options.acq = acq
    if mode is not None:
        options.mode = mode
    if mf_strategy is not None:
        options.mf_strategy = mf_strategy
    if isinstance(worker_manager, RealWorkerManager):
        options.capital_type = 'realtime'
    elif isinstance(worker_manager, SyntheticWorkerManager):
        options.capital_type = 'return_value'

    def get_initial_qinfos(num):
        return get_cp_domain_initial_qinfos(func_caller.domain, num)
    options.get_initial_qinfos = get_initial_qinfos

    # create optimiser and return
    optimiser = optimiser_constructor(
        func_caller,
        worker_manager,
        is_mf=is_mf,
        options=options,
        reporter=reporter,
        domain_dist_computers=domain_dist_computers
    )
    return optimiser.optimise(max_capital)

