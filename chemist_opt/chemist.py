"""
Module for BO with graph kernel and synthesizeable exploration.
@author: kkorovin@cs.cmu.edu

Available Explorers: 'rand_explorer', ...

TODO:
* Make the reporter report prepared options, not the initial ones
  ()
* options for different explorers,
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
                 chemist_args=None, reporter='default', 
                 is_mf=False, mf_strategy=None,
                 domain_dist_computers=None):
        self.func_caller = func_caller
        self.worker_manager = worker_manager
        self.data_source = data_source
        self.is_mf = is_mf
        self.mf_strategy = mf_strategy
        self.reporter = get_reporter(reporter)
        self.domain_dist_computers = domain_dist_computers
        if chemist_args is None:
            chemist_args = self.get_default_chemist_args()
        self.options = self.prepare_chemist_options(**chemist_args)

    def get_default_chemist_args(self):
        chemist_args = {'acq_opt_method': 'rand_explorer',
                        'init_capital': 'default'}
        return chemist_args

    def reset_default_options(self, list_of_options, chemist_args):
        """ Reset entries in list with entries in kwargs
            if name matches. Hence non-matching entries will be ignored.
        """
        for d in list_of_options:
            if d['name'] in chemist_args:
                d['default'] = chemist_args[d['name']]
        return list_of_options

    def prepare_chemist_options(self, **kwargs):
        """ Resets default gp_bandit options with chemist arguments """
        dflt_list_of_options = get_all_cp_gp_bandit_args()
        list_of_options = self.reset_default_options(dflt_list_of_options, kwargs)
        options = load_options(list_of_options, reporter=self.reporter)
 
        if self.mf_strategy is not None:
            options.mf_strategy = self.mf_strategy
        if isinstance(self.worker_manager, RealWorkerManager):
            options.capital_type = 'realtime'
        elif isinstance(self.worker_manager, SyntheticWorkerManager):
            options.capital_type = 'return_value'
        options.get_initial_qinfos = lambda num: \
                                        get_cp_domain_initial_qinfos(
                                            self.func_caller.domain, num
                                            )
        return options


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
