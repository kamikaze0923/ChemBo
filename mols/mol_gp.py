"""
CartesianProductGP implementation working on molecular domains.
Kernels are in mols/mol_kernels.py
@author: kkorovin@cs.cmu.edu

TODO: implement CPGP and CPGPFitter
* Either redefine or inherit

"""

import numpy as np
from argparse import Namespace

# dragonfly imports: TODO - not all of these will be needed
from dragonfly.exd import domains
from dragonfly.exd.cp_domain_utils import load_cp_domain_from_config_file
from dragonfly.gp import gp_core, mf_gp
# from dragonfly.euclidean_gp import get_euclidean_integral_gp_kernel_with_scale, \
                                   # prep_euclidean_integral_kernel_hyperparams
from dragonfly.gp.kernel import CartesianProductKernel, HammingKernel
from dragonfly.utils.general_utils import get_idxs_from_list_of_lists
from dragonfly.utils.option_handler import get_option_specs, load_options
from dragonfly.utils.reporters import get_reporter
# local imports
from mols.mol_kernels import MolKernel


# classes and functions to redefine
import dragonfly.gp.cartesian_product_gp as cartesian_product_gp

# options - import them
from dragonfly.gp.cartesian_product_gp import cartesian_product_gp_args,\
                                              cartesian_product_mf_gp_args
# append to this list the mol-dependent args
cartesian_product_gp_args += [get_option_specs('dom_mol_kernel_type', False, 'default',
                                               'Kernel type for Mol Domains.'),]
# in the future, may do the same for mf

_DFLT_DOMAIN_MOL_KERNEL_TYPE = "edgehist_kernel"

###############################################################################
 # Setup helpers new defitions

def get_default_kernel_type(domain_type):
    """ Returns default kernel type for the domain. """
    if domain_type == 'molecule':
        return _DFLT_DOMAIN_MOL_KERNEL_TYPE
    else:
        raise ValueError('domain_type %s not yet supported'%(domain_type))

def _set_up_hyperparams_for_domain(fitter, X_data, gp_domain, dom_prefix,
                                   kernel_ordering, kernel_params_for_each_domain,
                                   dist_computers, lists_of_dists):
    """ This modifies the fitter object. """
    for dom_idx, dom, kernel_type in zip(range(gp_domain.num_domains), gp_domain.list_of_domains, kernel_ordering):
        dom_type = dom.get_type()
        dom_identifier = '%s-%d-%s'%(dom_prefix, dom_idx, dom_type)
        # Kernel type
        if kernel_type == '' or kernel_type is None:
            # If none is specified, use the one given in options
            kernel_type = _get_kernel_type_from_options(dom_type, dom_prefix,
                                                        fitter.options)
        if kernel_type == 'default':
            kernel_type = get_default_kernel_type(dom.get_type())
        # Iterate through each individual domain and add it to the hyper parameters
        curr_dom_Xs = get_idxs_from_list_of_lists(X_data, dom_idx)
        # Some conditional options
        if dom.get_type() == 'molecule':
            if kernel_type == 'wl_kernel':
                fitter.dscr_hp_vals.append([1, 2, 3])
                fitter.param_order.append(['int_par', 'dscr'])
            elif kernel_type == 'edgehist_kernel':
                fitter.cts_hp_bounds.append([0, 5])
                fitter.param_order.append(['cont_par'])
            else:
                raise ValueError('Unknown kernel type "%s" for "%s" spaces.'%(kernel_type,
                                                                              dom.get_type()))
        else:
            raise ValueError('Unknown kernel type "%s" for "%s" spaces.'%(kernel_type,
                                                                          dom.get_type()))

def _get_kernel_type_from_options(dom_type, dom_prefix, options):
    """ Returns kernel type from options. """
    dom_type_descr_dict = {'euclidean': 'euc',
                         'discrete_euclidean': 'euc',
                         'integral': 'int',
                         'prod_discrete_numeric': 'disc_num',
                         'prod_discrete': 'disc',
                         'neural_network': 'nn',
                         'molecule': 'mol'
                        }
    if dom_type not in dom_type_descr_dict.keys():
        raise ValueError('Unknown domain type %s.'%(dom_type))
    attr_name = '%s_%s_kernel_type'%(dom_prefix, dom_type_descr_dict[dom_type])
    return getattr(options, attr_name)

def _prep_kernel_hyperparams_for_molecular_kernels(kernel_type, dom, kernel_params_for_dom):
    """ Prepares the kernel hyper-parameters. """
    ret = vars(kernel_params_for_dom)
    ret['mol_type'] = dom.mol_type
    ret['kernel_type'] = kernel_type
    # TODO: from options
    ret['int_par'] = 2
    ret['cont_par'] = 2.
    return ret

def get_molecular_kernel(kernel_type, kernel_hyperparams,
                         gp_cts_hps, gp_dscr_hps):
    """ kernel_hyperparams: dictionary
        gp_cts_hps, gp_dscr_hps - this may be modified and returned
    """
    # TODO: modify gp_cts_hps, gp_dscr_hps?
    kern = MolKernel(kernel_type, kernel_hyperparams)
    return kern, gp_cts_hps, gp_dscr_hps

def _build_kernel_for_domain(domain, dom_prefix, kernel_scale, gp_cts_hps, gp_dscr_hps,
                            other_gp_params, options, kernel_ordering, kernel_params_for_each_domain):
    """ Builds the kernel for the domain. """
    kernel_list = []
    # Iterate through each domain and build the corresponding kernel
    for dom_idx, dom, kernel_type in zip(range(domain.num_domains), 
                                        domain.list_of_domains,
                                        kernel_ordering):
        dom_type = dom.get_type().lower()
        if kernel_type == '' or kernel_type is None:
            # If none is specified, use the one given in options
            kernel_type = _get_kernel_type_from_options(dom_type, 'dom', options)
        if kernel_type == 'default':
            kernel_type = get_default_kernel_type(dom.get_type())
        if dom_type in ['euclidean', 'integral', 'prod_discrete_numeric',
                        'discrete_euclidean']:
            curr_kernel_hyperparams = _prep_kernel_hyperparams_for_euc_int_kernels(
                                      kernel_type, dom, dom_prefix, options)
            use_same_bw, _, esp_kernel_type, _ = _get_euc_int_options(dom_type, 'dom', options)
            if hasattr(other_gp_params, 'add_gp_groupings') and \
                other_gp_params.add_gp_groupings is not None:
                add_gp_groupings = other_gp_params.add_gp_groupings[dom_idx]
            else:
                add_gp_groupings = None
            curr_kernel, gp_cts_hps, gp_dscr_hps = \
                get_euclidean_integral_gp_kernel_with_scale(kernel_type, 1.0, \
                curr_kernel_hyperparams, gp_cts_hps, gp_dscr_hps, use_same_bw,
                add_gp_groupings, esp_kernel_type)
        elif dom_type == 'prod_discrete':
            curr_kernel_hyperparams = _prep_kernel_hyperparams_for_discrete_kernels(
                                         kernel_type, dom, dom_prefix, options)
            curr_kernel, gp_cts_hps, gp_dscr_hps = \
                get_discrete_kernel(kernel_type, curr_kernel_hyperparams, gp_cts_hps,
                                    gp_dscr_hps)
        elif dom_type == 'neural_network':
            curr_kernel_hyperparams = _prep_kernel_hyperparams_for_nn_kernels(
                                      kernel_type, dom,
                                      kernel_params_for_each_domain[dom_idx])
            curr_kernel, gp_cts_hps, gp_dscr_hps = \
                get_neural_network_kernel(kernel_type, curr_kernel_hyperparams, gp_cts_hps,
                                      gp_dscr_hps)
        elif dom_type == 'molecule':
            curr_kernel_hyperparams = _prep_kernel_hyperparams_for_molecular_kernels(
                                      kernel_type, dom,
                                      kernel_params_for_each_domain[dom_idx])
            curr_kernel, gp_cts_hps, gp_dscr_hps = \
                get_molecular_kernel(kernel_type, curr_kernel_hyperparams, gp_cts_hps,
                                      gp_dscr_hps)
        else:
          raise NotImplementedError(('Not implemented _child_build_gp for dom_type ' +
                                     '%s yet.')%(dom_type))
        kernel_list.append(curr_kernel)
    return CartesianProductKernel(kernel_scale, kernel_list), gp_cts_hps, gp_dscr_hps

###############################################################################
# Resetting:
cartesian_product_gp._DFLT_DOMAIN_MOL_KERNEL_TYPE = _DFLT_DOMAIN_MOL_KERNEL_TYPE
cartesian_product_gp._prep_kernel_hyperparams_for_molecular_kernels = _prep_kernel_hyperparams_for_molecular_kernels
cartesian_product_gp.get_molecular_kernel = get_molecular_kernel
cartesian_product_gp.get_default_kernel_type.__code__ = get_default_kernel_type.__code__
cartesian_product_gp._set_up_hyperparams_for_domain.__code__ = _set_up_hyperparams_for_domain.__code__
cartesian_product_gp._build_kernel_for_domain.__code__ = _build_kernel_for_domain.__code__
cartesian_product_gp._get_kernel_type_from_options.__code__ = _get_kernel_type_from_options.__code__

###############################################################################
# API classes: using resetted functions

class MolCPGP(cartesian_product_gp.CPGP):
    """ this may not need any modifications at all:
        doesn't use any of the module's functions
    """
    pass


class MolCPGPFitter(cartesian_product_gp.CPGPFitter):
    """also the same functionality as base class"""
    pass


###############################################################################
####                   Old stuff, may be useful to consult                 ####
###############################################################################

# _DFLT_KERNEL_TYPE = 'wl_kernel'

# # dict: name, required, default, help -> these values
# mol_gp_specific_args = [
#     get_option_specs('cont_par', False, '0.1-0.25-0.61-1.5',
#     'Continuous parameter for single-parameter kernels. for If -1, it means we will tune.'),
#     get_option_specs('int_par', False, 3,
#     'Integer parameter for single-parameter kernels. for If -1, it means we will tune.'),
#     # get_option_specs('non_assignment_penalty', False, 1.0,
#     # 'The non-assignment penalty.'),

# ]  # check what these should be

# mol_gp_args = gp_core.mandatory_gp_args + basic_gp_args + mol_gp_specific_args


# # GP implementation for molecules ---------------------------------------------

# class MolGP(gp_core.GP):
#     """ A Gaussian process for Molecules. """
#     def __init__(self, X, Y, kernel_type, kernel_hyperparams, mean_func, noise_var, *args, **kwargs):
#         """ Constructor.
#             kernel_type: Should be one of [TODO: kernel names]
#             kernel_hyperparams: is a python dictionary specifying the hyper-parameters for the
#                                 kernel. Will have parameters [TODO: which hyperparams of kernels]
#             list_of_dists: Is a list of distances for the training set.
#         """
#         kernel = self._get_kernel_from_type(kernel_type, kernel_hyperparams)
#         super(MolGP, self).__init__(X, Y, kernel, mean_func, noise_var,
#                                     handle_non_psd_kernels='project_first',
#                                     *args, **kwargs)

#     def _get_training_kernel_matrix(self):
#         """ Compute the kernel matrix from distances if they are provided. """
#         return self.kernel(self.X, self.X)

#     def _child_str(self):
#         """ Description of the child GP. """
#         return self.kernel.kernel_type

#     @classmethod
#     def _get_kernel_from_type(cls, kernel_type, kernel_hyperparams):
#         """ Returns the kernel with set hyperparams. """
#         return MolKernel(kernel_type, kernel_hyperparams)


# # GP fitter for molecules -----------------------------------------------------

# class MolGPFitter(gp_core.GPFitter):
#     """
#     Fits a GP by tuning the kernel hyper-params.
#     This is the interface for MolGP.
#     """
#     def __init__(self, X, Y, options=None, reporter=None, *args, **kwargs):
#         """ Constructor. """
#         self.X = X
#         self.Y = Y
#         self.reporter = get_reporter(reporter)
#         self.num_data = len(X)
#         if options is None:
#             options = load_options(mol_gp_args, 'GPFitter', reporter=reporter)
#         super(MolGPFitter, self).__init__(options, *args, **kwargs)

#     def _child_set_up(self):
#         """
#         Technical method that sets up attributes,
#         such as hyperparameter bounds etc.
#         """
#         if self.options.kernel_type == 'default':
#             self.kernel_type = _DFLT_KERNEL_TYPE
#         else:
#             self.kernel_type = self.options.kernel_type
#         # Some parameters we will be using often
#         if len(self.Y) > 1:
#             self.Y_var = self.Y.std() ** 2
#         else:
#             self.Y_var = 1.0
#         # Compute the pairwise distances here
#         # Set up based on whether we are doing maximum likelihood or post_sampling
#         if self.options.hp_tune_criterion == 'ml':
#             self.hp_bounds = []
#             self._child_set_up_ml_tune()

#     def _child_set_up_ml_tune(self):
#         """
#         Sets up tuning for Maximum likelihood.
#         TODO: choose boundaries for kernels
#         """
        
#         """ Sets up tuning for Maximum likelihood. """
#         # Order of hyper-parameters (this is critical):
#         # noise variance, int_par/cont_par
        
#         # 1. Noise variance ---------------------------------------
#         if self.options.noise_var_type == 'tune':
#             self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
#             self.hp_bounds.append(self.noise_var_log_bounds)
        
#         # 2. int_par/cont_par ---------------------------------------
#         if self.kernel_type == "wl_kernel":
#             int_bounds = [2, 3]
#             self.hp_bounds.append(int_bounds)
#         elif self.kernel_type == "edgehist_kernel":
#             cont_bounds = [0.1, 5.]
#             self.hp_bounds.append(cont_bounds)

#     def _child_set_up_post_sampling(self):
#         raise NotImplementedError("Not implemented post sampling yet.")

#     def _child_build_gp(self, gp_hyperparams, build_posterior=True):
#         """ Builds the GP from the hyper-parameters. """
#         gp_hyperparams = gp_hyperparams[:]  # create a copy of the list
#         kernel_hyperparams = {}
#         # mean functions
#         mean_func = gp_core.get_mean_func_from_options(self.options, self.Y)
#         # extract GP hyper-parameters
#         # 1. Noise variance ------------------------------------------------------
#         noise_var = gp_core.get_noise_var_from_options_and_hyperparams(self.options,
#                                                           gp_hyperparams, self.Y, 0)
#         if self.options.noise_var_type == 'tune':
#             gp_hyperparams = gp_hyperparams[1:]

#         # SETTING kernel_hyperparams BASED ON KERNEL TYPE--------------------------
#         if self.kernel_type == "wl_kernel":
#             kernel_hyperparams["int_par"] = gp_hyperparams[:1][0]
#             gp_hyperparams = gp_hyperparams[1:]
#         elif self.kernel_type == "edgehist_kernel":
#             kernel_hyperparams["cont_par"] = gp_hyperparams[:1][0]
#             gp_hyperparams = gp_hyperparams[1:]

#         return MolGP(self.X, self.Y, self.kernel_type, 
#                      kernel_hyperparams, mean_func, noise_var,
#                      build_posterior=build_posterior)

