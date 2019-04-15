"""
  Harness for matching based distance computation among chemical molecules.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error

import numpy as np
from rdkit.Chem.rdchem import BondType
from mols.molecule import Molecule

# Define bond types
SINGLE_BOND = BondType.SINGLE
DOUBLE_BOND = BondType.DOUBLE
TRIPLE_BOND = BondType.TRIPLE
AROMATIC_BOND = BondType.AROMATIC
IONIC_BOND = BondType.IONIC


def get_atom_type_dissimilarity_matrix(list_of_atoms_1, list_of_atoms_2):
  """ Returns the dissimilarity matrix. """
  n1 = len(list_of_atoms_1)
  n2 = len(list_of_atoms_2)
  ret = np.zeros((n1, n2))
  for i, elem_1 in enumerate(list_of_atoms_1):
    for j, elem_2 in enumerate(list_of_atoms_2):
      ret[i, j] = 1 - float(elem_1 == elem_2)
  ret[ret >= 0.9] = np.inf
  return ret


def get_dissimiliary_matrix_with_non_assignment(orig_dissim_matrix,
                                                non_assignment_penalty_vals=1.0):
  """ Returns a dissimilarity matrix augmented with non-assignment. """
  if not hasattr(non_assignment_penalty_vals, '__iter__'):
    non_assignment_penalty_vals = [non_assignment_penalty_vals]
  n1, n2 = orig_dissim_matrix.shape
  hor_stack = np.repeat([non_assignment_penalty_vals], n1, axis=0)
  ret = np.hstack((orig_dissim_matrix, hor_stack))
  ver_stack = np.repeat(np.transpose([non_assignment_penalty_vals]),
                        n2 + len(non_assignment_penalty_vals), axis=1)
  ret = np.vstack((ret, ver_stack))
  return ret



class ChemDistComputer(object):
  """ An abstract class for distance computation among chemical molecules. Adapted
      from NNDistanceComputer class in
      github.com/kirthevasank/nasbot/blob/master/nn/nn_comparators.py
  """

  def __init__(self):
    """ Constructor. """
    super(ChemDistanceComputer, self).__init__()

  def __call__(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances by calling evaluate. """
    return self.evaluate(X1, X2, *args, **kwargs)

  def evaluate(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances between X1 and X2 and returns an n1 x n2 distance matrix.
        If X1 and X2 are single chemical molecules, returns a scalar. """
    if isinstance(X1, Molecule) and isinstance(X2, Molecule):
      return self.evaluate_single(X1, X2, *args, **kwargs)
    else:
      n1 = len(X1)
      n2 = len(X2)
      X2 = X2 if X2 is not None else X1
      x1_is_x2 = X1 is X2

      all_ret = None
      es_is_iterable = None
      for i, x1 in enumerate(X1):
        X2_idxs = range(i, n2) if x1_is_x2 else range(n2)
        for j in X2_idxs:
          x2 = X2[j]
          # Compute the distances
          curr_ret = self.evaluate_single(x1, x2, *args, **kwargs)
          all_ret, es_is_iterable = self._add_to_all_ret(curr_ret, i, j, n1, n2,
                                                         all_ret, es_is_iterable)
          # Check if we need to do j and i as well.
          if x1_is_x2:
            all_ret, es_is_iterable = self._add_to_all_ret(curr_ret, j, i, n1, n2,
                                                           all_ret, es_is_iterable)
      return all_ret

  @classmethod
  def _add_to_all_ret(cls, curr_ret, i, j, n1, n2, all_ret=None, es_is_iterable=None):
    """ Adds the current result to all results. """
    if all_ret is None:
      if hasattr(curr_ret, '__iter__'):
        es_is_iterable = True
        all_ret = [np.zeros((n1, n2)) for _ in range(len(curr_ret))]
      else:
        es_is_iterable = False
        all_ret = np.zeros((n1, n2))
    if es_is_iterable:
      for k in range(len(curr_ret)):
        all_ret[k][i, j] = curr_ret[k]
    else:
      all_ret[i, j] = curr_ret
    return all_ret, es_is_iterable

  def evaluate_single(self, x1, x2, *args, **kwargs):
    """ Evaluates the distance between the two networks x1 and x2. """
    raise NotImplementedError('Implement in a child class.')


def QPChemDistanceComputer(ChemDistComputer)
  """ A distance between chemical molecules based on Quadratic Programming. """

  def __init__(self, struct_pen_coeffs, mislabel_pen_coeffs=None,
               non_assignment_penalty=1.0, nonexist_non_assignment_penalty=1.0):
    """ Constructor.
        struct_pen_coeffs: A list of coefficients for the structural penalty term.
        mislabel_pen_coeffs: A list of coefficients for the mislabel penalty.
        non_assignment_penalty: The non-assignment penalty.
        nonexist_non_assignment_penalty: The non-assignment penalty if the particular
                                         atom does not exist in the other molecule.
    """
    if mislabel_pen_coeffs is None:
      mislabel_pen_coeffs = 1.0
    if not hasattr(mislabel_pen_coeffs, '__iter__'):
      mislabel_pen_coeffs = [mislabel_pen_coeffs] * len(struct_pen_coeffs)
    self.struct_pen_coeffs = struct_pen_coeffs
    self.mislabel_pen_coeffs = mislabel_pen_coeffs
    self.non_assignment_penalty = non_assignment_penalty
    self.nonexist_non_assignment_penalty = nonexist_non_assignment_penalty

