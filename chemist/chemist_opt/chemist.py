"""
Module for BO with graph kernel and synthesizeable exploration.
Core class is Chemist
@author: kkorovin@cs.cmu.edu


TODO: may need to implement:
* Molecular domain (if cart prod does not work)
* corresponding MolGPBandit

"""

import numpy as np

# my imports
from mols.mol_gp import mol_gp_args, MolGPFitter
from mols.mol_kernels import *   # kernel names
from datasets.loaders import get_initial_pool

# Dragonfly imports
from dragonfly.opt.blackbox_optimiser import blackbox_opt_args
from dragonfly.opt.gp_bandit import GPBandit, gp_bandit_args
from dragonfly.utils.general_utils import block_augment_array
from dragonfly.utils.reporters import get_reporter
from dragonfly.utils.option_handler import get_option_specs, load_options




from dragonfly.opt.gp_bandit import CPGPBandit

def optimize_chemist(func_caller, worker_manager, max_capital, is_mf, mode=None,
                     acq=None, mf_strategy=None, domain_add_max_group_size=-1,
                     options=None, reporter='default'):
    # TODO: a different Bandit:
    # if domain of the func caller is cart product:
    optimiser_constructor = CPGPBandit
    dflt_list_of_options = get_all_cp_gp_bandit_args()
    # potentially use it here instead

    #--- TODO options ---#
    # .................. #
    #--- TODO options ---#

    # create optimiser and return
    optimiser = optimiser_constructor(func_caller, worker_manager, is_mf=is_mf,
                                      options=options, reporter=reporter)
    return optimiser.optimise(max_capital)













# Old stuff, maybe not needed any more ========================================

# Options for acquisition optimizers:
# - random_ga: randomly select subsets and synthesize
# - ToBeAdded

chemist_specific_args = [
    get_option_specs('chemist_acq_opt_method', False, 'ga',
    'Which method to use when optimising the acquisition. Will override acq_opt_method' +
    ' in the arguments for gp_bandit.'),
]


all_chemist_args = chemist_specific_args + gp_bandit_args + \
                     blackbox_opt_args + mol_gp_args


class Chemist(GPBandit):
    """
    Analog of NASBOT class.
    To not have it inherit from any GPBandit,
    must merge and simplify all functionality.
    """
    def __init__(self, func_caller, worker_manager, options=None, reporter=None):
        if options is None:
            reporter = get_reporter(reporter)
            options = load_options(all_chemist_args, reporter=reporter)
        super(Chemist, self).__init__(func_caller, worker_manager,
                                      options=options, reporter=reporter)

    def _child_set_up(self):
        """ Child up. """
        # First override the acquisition optisation method
        self.options.acq_opt_method = self.options.chemist_acq_opt_method
        # No cal the super function
        super(Chemist, self)._child_set_up()
        self.list_of_dists = None
        self.already_evaluated_dists_for = None
        # Create a GP fitter with no data and use its tp_comp as the bandit's tp_comp
        init_gp_fitter = MolGPFitter([], [], options=self.options, reporter=self.reporter)

    def _set_up_acq_opt_ga(self):
        self.ga_init_pool = get_initial_pool()
        self.ga_mutation_op = lambda x: x
        # In future, implement Domains:
        # # The initial pool
        # self.ga_init_pool = get_initial_pool(self.domain.get_type())
        # # The number of evaluations
        if self.get_acq_opt_max_evals is None:
            #lead_const = min(5, self.domain.get_dim())**2
            lead_const = 25
            self.get_acq_opt_max_evals = lambda t: np.clip(
                          lead_const * int(np.sqrt(t)), 50, 500)

    # def _compute_list_of_dists(self, X1, X2):
    #     raise NotImplementedError("ImplementMe")

    def _get_gp_fitter(self, reg_X, reg_Y):
        """ Builds a NN GP. """
        return MolGPFitter(reg_X, reg_Y,
                           options=self.options,
                           reporter=self.reporter)


    def _add_data_to_gp(self, new_points, new_vals):
        """Adds data to the GP. Also tracks list_of_dists."""
        ## First add it to the list of distances
        # if self.list_of_dists is None:
        #     # This is the first time, so use all the data.
        #     reg_X, _ = self._get_reg_X_reg_Y()
        #     self.list_of_dists = self._compute_list_of_dists(reg_X, reg_X)
        #     self.already_evaluated_dists_for = reg_X
        # else:
        #     list_of_dists_old_new = self._compute_list_of_dists(
        #                             self.already_evaluated_dists_for, new_points)
        #     list_of_dists_new_new = self._compute_list_of_dists(new_points, new_points)
        #     self.already_evaluated_dists_for.extend(new_points)
        #     for idx in range(len(list_of_dists_old_new)):
        #         self.list_of_dists[idx] = block_augment_array(
        #             self.list_of_dists[idx], list_of_dists_old_new[idx],
        #             list_of_dists_old_new[idx].T, list_of_dists_new_new[idx])

        # Now add to the GP
        if self.gp_processor.fit_type == 'fitted_gp':
            self.gp.add_data(new_points, new_vals, build_posterior=False)
            #self.gp.set_list_of_dists(self.list_of_dists)
            self.gp.build_posterior()

    def _child_set_gp_data(self, reg_X, reg_Y):
        """ Set Data for the child. """
        # if self.list_of_dists is None:
        #     self.list_of_dists = self._compute_list_of_dists(reg_X, reg_X)
        #     self.already_evaluated_dists_for = reg_X
        # if (len(reg_X), len(reg_Y)) != self.list_of_dists[0].shape:
        #     print (len(reg_X)), len(reg_Y), self.list_of_dists[0].shape, self.step_idx
        # assert (len(reg_X), len(reg_Y)) == self.list_of_dists[0].shape

        # self.gp.set_list_of_dists(self.list_of_dists)
        self.gp.set_data(reg_X, reg_Y, build_posterior=True)


# APIs ---------------------------------------------------------

def optimize_chemist(func_caller, worker_manager, budget, mode=None,
           init_pool=None, acq='hei', options=None, reporter='default'):
    """ Chemist optimization from a function caller. """
    if options is None:
        reporter = get_reporter(reporter)
        options = load_options(all_chemist_args, reporter=reporter)

    # TODO: what is this option?
    if acq is not None:
        options.acq = acq

    if mode is not None:
        options.mode = mode

    # Initial queries
    if not hasattr(options, 'pre_eval_points') or options.pre_eval_points is None:
        if init_pool is None:
            init_pool = get_initial_pool()
        options.get_initial_points = lambda n: init_pool[:n]

    return (Chemist(func_caller, worker_manager,
             options=options, reporter=reporter)).optimise(budget)
