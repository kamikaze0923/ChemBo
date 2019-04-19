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
    def __init__(self, func_caller, worker_manager, data_source,
                 options=None, reporter='default',
                 is_mf=False, mf_strategy=None, domain_dist_computers=None):
        """
        :param func_caller:
        :param worker_manager:
        :param data_source:
        :param options:
        :param reporter:
        :param is_mf: whether multi-fidelity
        :param mf_strategy:
        :param domain_dist_computers:
            a list of functions for each domain to compute the pairwise distance between two lists of data
            i.e.: for two lists of length $n_1$ and $n_2$ respectively, return a $(n_1, n_2)$ matrix of pair-wise distance
        """
        self.func_caller = func_caller
        self.worker_manager = worker_manager
        self.data_source = data_source
        self.is_mf = is_mf
        self.reporter = get_reporter(reporter)
        self.domain_dist_computers = domain_dist_computers

        if options is None:
            dflt_list_of_options = get_all_cp_gp_bandit_args()
            self.options = load_options(dflt_list_of_options,
                                        reporter=reporter)
        # TODO: passing explorer options
        self.options.acq_opt_method = 'rand_explorer'
        if mf_strategy is not None:
            self.options.mf_strategy = mf_strategy
        if isinstance(worker_manager, RealWorkerManager):
            self.options.capital_type = 'realtime'
        elif isinstance(worker_manager, SyntheticWorkerManager):
            self.options.capital_type = 'return_value'

        def get_initial_qinfos(num):
            return get_cp_domain_initial_qinfos(func_caller.domain, num)
        self.options.get_initial_qinfos = get_initial_qinfos

    def run(self, max_capital):
        """ Main Chemist method

        Returns:
            opt_val, opt_point, history
        """
        optimiser_constructor = CPGPBandit

        # create optimiser and return
        optimiser = optimiser_constructor(
            self.func_caller,
            self.worker_manager,
            is_mf=self.is_mf,
            options=self.options,
            reporter=self.reporter,
            domain_dist_computers=self.domain_dist_computers
        )
        return optimiser.optimise(max_capital)
