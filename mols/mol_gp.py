"""
GP implementation working on molecular domains.
Kernels are in mol_kernels.py
@author: kkorovin@cs.cmu.edu

TODO:
* complete all places with ImplementMe
* start with _child_set_up in GPFitter
* look into what the mol_gp_specific_args should be
  (see get_option_specs)
* after that, write unittests for both classes
NOTES:
* self.tp_comp is an object that has .evaluate_single(x1, x2) method
it is a component of kernels in nasbot, so I shouldn't need it
* mol_gp_args are passed to superconstructor (GPFitter)
"""

import numpy as np
from mols.mol_kernels import MolKernel

# dragonfly imports
from dragonfly.gp import gp_core
from dragonfly.gp.gp_instances import basic_gp_args
from dragonfly.utils.ancillary_utils import get_list_of_floats_as_str
from dragonfly.utils.reporters import get_reporter
from dragonfly.utils.option_handler import get_option_specs, load_options


_DFLT_KERNEL_TYPE = 'wl_kernel'

# dict: name, required, default, help -> these values
mol_gp_specific_args = [
    get_option_specs('cont_par', False, '0.1-0.25-0.61-1.5',
    'Continuous parameter for single-parameter kernels. for If -1, it means we will tune.'),
    get_option_specs('int_par', False, 3,
    'Integer parameter for single-parameter kernels. for If -1, it means we will tune.'),
    # get_option_specs('non_assignment_penalty', False, 1.0,
    # 'The non-assignment penalty.'),

]  # check what these should be

mol_gp_args = gp_core.mandatory_gp_args + basic_gp_args + mol_gp_specific_args


# GP implementation for molecules ---------------------------------------------

class MolGP(gp_core.GP):
    """ A Gaussian process for Molecules. """
    def __init__(self, X, Y, kernel_type, kernel_hyperparams, mean_func, noise_var, *args, **kwargs):
        """ Constructor.
            kernel_type: Should be one of [TODO: kernel names]
            kernel_hyperparams: is a python dictionary specifying the hyper-parameters for the
                                kernel. Will have parameters [TODO: which hyperparams of kernels]
            list_of_dists: Is a list of distances for the training set.
        """
        kernel = self._get_kernel_from_type(kernel_type, kernel_hyperparams)
        super(MolGP, self).__init__(X, Y, kernel, mean_func, noise_var,
                                    handle_non_psd_kernels='project_first',
                                    *args, **kwargs)

    def _get_training_kernel_matrix(self):
        """ Compute the kernel matrix from distances if they are provided. """
        return self.kernel(self.X, self.X)

    def _child_str(self):
        """ Description of the child GP. """
        return self.kernel.kernel_type

    @classmethod
    def _get_kernel_from_type(cls, kernel_type, kernel_hyperparams):
        """ Returns the kernel with set hyperparams. """
        return MolKernel(kernel_type, kernel_hyperparams)


# GP fitter for molecules -----------------------------------------------------

class MolGPFitter(gp_core.GPFitter):
    """
    Fits a GP by tuning the kernel hyper-params.
    This is the interface for MolGP.
    """
    def __init__(self, X, Y, options=None, reporter=None, *args, **kwargs):
        """ Constructor. """
        self.X = X
        self.Y = Y
        self.reporter = get_reporter(reporter)
        self.num_data = len(X)
        if options is None:
            options = load_options(mol_gp_args, 'GPFitter', reporter=reporter)
        super(MolGPFitter, self).__init__(options, *args, **kwargs)

    def _child_set_up(self):
        """
        Technical method that sets up attributes,
        such as hyperparameter bounds etc.
        """
        if self.options.kernel_type == 'default':
            self.kernel_type = _DFLT_KERNEL_TYPE
        else:
            self.kernel_type = self.options.kernel_type
        # Some parameters we will be using often
        if len(self.Y) > 1:
            self.Y_var = self.Y.std() ** 2
        else:
            self.Y_var = 1.0
        # Compute the pairwise distances here
        # Set up based on whether we are doing maximum likelihood or post_sampling
        if self.options.hp_tune_criterion == 'ml':
            self.hp_bounds = []
            self._child_set_up_ml_tune()

    def _child_set_up_ml_tune(self):
        """
        Sets up tuning for Maximum likelihood.
        TODO: choose boundaries for kernels
        """
        
        """ Sets up tuning for Maximum likelihood. """
        # Order of hyper-parameters (this is critical):
        # noise variance, int_par/cont_par
        
        # 1. Noise variance ---------------------------------------
        if self.options.noise_var_type == 'tune':
            self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
            self.hp_bounds.append(self.noise_var_log_bounds)
        
        # 2. int_par/cont_par ---------------------------------------
        if self.kernel_type == "wl_kernel":
            int_bounds = [2, 3]
            self.hp_bounds.append(int_bounds)
        elif self.kernel_type == "edgehist_kernel":
            cont_bounds = [0.1, 5.]
            self.hp_bounds.append(cont_bounds)

    def _child_set_up_post_sampling(self):
        raise NotImplementedError("Not implemented post sampling yet.")

    def _child_build_gp(self, gp_hyperparams, build_posterior=True):
        """ Builds the GP from the hyper-parameters. """
        gp_hyperparams = gp_hyperparams[:]  # create a copy of the list
        kernel_hyperparams = {}
        # mean functions
        mean_func = gp_core.get_mean_func_from_options(self.options, self.Y)
        # extract GP hyper-parameters
        # 1. Noise variance ------------------------------------------------------
        noise_var = gp_core.get_noise_var_from_options_and_hyperparams(self.options,
                                                          gp_hyperparams, self.Y, 0)
        if self.options.noise_var_type == 'tune':
            gp_hyperparams = gp_hyperparams[1:]

        # SETTING kernel_hyperparams BASED ON KERNEL TYPE--------------------------
        if self.kernel_type == "wl_kernel":
            kernel_hyperparams["int_par"] = gp_hyperparams[:1][0]
            gp_hyperparams = gp_hyperparams[1:]
        elif self.kernel_type == "edgehist_kernel":
            kernel_hyperparams["cont_par"] = gp_hyperparams[:1][0]
            gp_hyperparams = gp_hyperparams[1:]

        return MolGP(self.X, self.Y, self.kernel_type, 
                     kernel_hyperparams, mean_func, noise_var,
                     build_posterior=build_posterior)

