"""

Molecular function callers.
@author: kkorovin@cs.cmu.edu

A harness for calling functions defined over Molecules.
Makes use of the mols/mol_functions.py

"""

from copy import deepcopy
import numpy as np
from time import sleep

# Local imports
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.exd.exd_core import EVAL_ERROR_CODE
from dragonfly.utils.reporters import get_reporter

class MolFunctionCaller(CPFunctionCaller):
    pass