"""
  Harness for matching based distance computation among chemical molecules.
  -- kirthevasank
"""

import numpy as np


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
  ver_stack = np.repeat(np.transpose([non_assignment_penalty_vals],
                        n2 + len(non_assignment_penalty_vals), axis=1)
  ret = np.vstack((ret, ver_stack))
  return ret



