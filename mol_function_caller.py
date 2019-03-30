"""

Function caller working on Molecular domains.
@author: kkorovin@cs.cmu.edu

TODO:
* Think whether Cart product can be used,
  and if not, look at exd.experiment_callers
  and find one to inherit from to make
  a MolFunctionCaller

"""

from copy import deepcopy
import numpy as np
from time import sleep

# Local imports
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.exd.exd_core import EVAL_ERROR_CODE
from dragonfly.utils.reporters import get_reporter


class MolFunctionCaller(CPFunctionCaller):
    def __init__(self, config, train_params, descr='',
                 debug_mode=False, reporter='silent'):
        pass
    @classmethod
    def is_mf(cls):
        """ Returns True if Multi-fidelity. """
        return False
    def _fidel_cost(self, fidel):
        pass
    @classmethod
    def _raw_fidel_cost(cls, raw_fidel):
        pass
    def eval_at_fidel_single(self, fidel, point, qinfo, noisy=False):
        pass
    def _func_wrapper(self, raw_fidel, raw_point, qinfo):
        pass
    @classmethod
    def _eval_synthetic_function(cls, raw_point):
        pass

    # The score (function evaluation)
    # def _eval_validation_score(self, raw_fidel, raw_point, qinfo):
    #     #This is a function that may need
    #     #to be overriden for the child class
    #     pass
