"""
  Base class for distance computation among chemical molecules.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error

import numpy as np
from mols.molecule import Molecule


class ChemDistanceComputer(object):
  """ An abstract class for distance computation among chemical molecules. Adapted
      from NNDistanceComputer class in
      github.com/kirthevasank/nasbot/blob/master/nn/nn_comparators.py
  """

  def __call__(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances by calling evaluate. """
    return self.evaluate(X1, X2, *args, **kwargs)

  def evaluate(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances between X1 and X2 and returns an n1 x n2 distance matrix.
        If X1 and X2 are single chemical molecules, returns a scalar. """
    if isinstance(X1, Molecule) and isinstance(X2, Molecule):
      return self.evaluate_single(X1, X2, *args, **kwargs)
    # Otherwise, compute a matrix of dissimilarities --------------------
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
      for k, curr_ret_elem in enumerate(curr_ret):
        all_ret[k][i, j] = curr_ret_elem
    else:
      all_ret[i, j] = curr_ret
    return all_ret, es_is_iterable

  def get_num_distances(self):
    """ Return the number of distances. """
    raise NotImplementedError('Implement in a child class.')

  def evaluate_single(self, x1, x2, *args, **kwargs):
    """ Evaluates the distance between the two networks x1 and x2. """
    raise NotImplementedError('Implement in a child class.')

