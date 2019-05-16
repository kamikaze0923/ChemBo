"""
  Harness for matching based distance computation among chemical molecules.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error
# pylint: disable=no-member

from copy import copy
import numpy as np
from myrdkit import Chem
from dragonfly.utils.oper_utils import opt_transport
# Local
from dist.dist_computer import ChemDistanceComputer
from dist.dist_utils import get_graph_data_for_distance_computation

# Define bond types
SINGLE_BOND = Chem.rdchem.BondType.SINGLE
DOUBLE_BOND = Chem.rdchem.BondType.DOUBLE
TRIPLE_BOND = Chem.rdchem.BondType.TRIPLE
AROMATIC_BOND = Chem.rdchem.BondType.AROMATIC
IONIC_BOND = Chem.rdchem.BondType.IONIC
REPLACE_COST_INF_WITH = 7.65432198e6


# Some utilities we will need below.
def get_unique_elements(list_of_items):
  """ Returns unique elements. """
  return list(set(list_of_items))


# Utilities for computing the dissimilarity matrices ----------------------------------
def get_atom_type_similarity_matrix(list_of_atoms_1, list_of_atoms_2):
  """ Returns a Boolean matrix where ret(i,j) = True if the list_of_atoms_1[i] and
      list_of_atoms_2[j] are the same and False otherwise. """
  n1 = len(list_of_atoms_1)
  n2 = len(list_of_atoms_2)
  ret = np.full((n1, n2), False)
  for i, elem_1 in enumerate(list_of_atoms_1):
    for j, elem_2 in enumerate(list_of_atoms_2):
      ret[i, j] = float(elem_1 == elem_2)
  return ret

def get_bond_similarity_matrix(bonds_of_each_atom_1, bonds_of_each_atom_2):
  """ Structural Dissimilarity Matrices. """
  n1 = len(bonds_of_each_atom_1)
  n2 = len(bonds_of_each_atom_2)
  ret = np.full((n1, n2), False)
  for i, bonds_1 in enumerate(bonds_of_each_atom_1):
    for j, bonds_2 in enumerate(bonds_of_each_atom_2):
      ret[i, j] = float(bonds_1 == bonds_2)
  return ret

def get_mismatching_bond_frac_matrix(bond_type_counts_1, bond_type_counts_2):
  """ Structural Dissimilarity Matrices. """
  n1 = len(bond_type_counts_1)
  n2 = len(bond_type_counts_2)
  ret = np.zeros((n1, n2))
  for i, bonds_1 in enumerate(bond_type_counts_1):
    for j, bonds_2 in enumerate(bond_type_counts_2):
      merged_bond_counts = copy(bonds_1)
      for key, val in bonds_2.items():
        if key in merged_bond_counts.keys():
          merged_bond_counts[key] = max(merged_bond_counts[key], val)
        else:
          merged_bond_counts[key] = val
      # Compute a difference dictionary
      diff_bond_counts = {}
      for key, val in merged_bond_counts.items():
        if key in bonds_1 and key in bonds_2:
          diff_bond_counts[key] = val - min(bonds_1[key], bonds_2[key])
        else:
          diff_bond_counts[key] = val
      # Compute sum of differences and union
      num_diffs = sum([val for _, val in diff_bond_counts.items()])
      normaliser = sum([val for _, val in merged_bond_counts.items()])
      if normaliser == 0:
        assert num_diffs == 0
        ret[i, j] = 1.0
      else:
        ret[i, j] = num_diffs / float(normaliser)
  return ret

def get_matching_matrix_from_similarity_matrix(similarity_matrix,
                                               similar_coeff, dissimilar_coeff):
  """ Returns a matching matrix.
      ret(i,j) = similar_coeff if the list_of_atoms_1[i] and list_of_atoms_2[j]
                 are the same and dissimilar_coeff otherwise.
  """
  ret = np.zeros(similarity_matrix.shape)
  ret[similarity_matrix] = similar_coeff
  ret[np.logical_not(similarity_matrix)] = dissimilar_coeff
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


# Now define the distance computer =======================================================
class OTChemDistanceComputer(ChemDistanceComputer):
  """ A distance between chemical molecules based on Quadratic Programming. """

  def __init__(self,
               mass_assignment_method='equal-molecular_mass',
               normalisation_method='none-total_mass',
               struct_pen_method='bond_frac',
               struct_pen_coeffs=1.0,
               non_assignment_penalty=1.0,
               nonexist_non_assignment_penalty_vals=1.0,
              ):
    """ Constructor.
        struct_pen_coeffs: A list of coefficients for the structural penalty term.
        mass_assignment_method: A string indicating how the masses should be assigned
                                to each atom. If equal, we will use equal (unit) mass
                                for all atoms. If atomic_mass, we will use atomic mass.
        normalisation_method: How to normalise for the different sizes of the molecules.
        non_assignment_penalty: The non-assignment penalty.
        nonexist_non_assignment_penalty: The non-assignment penalty if the particular
                                         atom does not exist in the other molecule.
    """
    if not hasattr(struct_pen_coeffs, '__iter__'):
      struct_pen_coeffs = [struct_pen_coeffs]
    if not hasattr(nonexist_non_assignment_penalty_vals, '__iter__'):
      nonexist_non_assignment_penalty_vals = [nonexist_non_assignment_penalty_vals]
    # Assign attributes
    self.mass_assignment_methods = mass_assignment_method.split('-')
    self.normalisation_methods = normalisation_method.split('-')
    self.struct_pen_methods = struct_pen_method.split('-')
    self.non_assignment_penalty = non_assignment_penalty
    self.struct_pen_coeffs = struct_pen_coeffs
    self.nonexist_non_assignment_penalty_vals = nonexist_non_assignment_penalty_vals
    self._num_distances = None
    self.str_params = self.format_params(mass_assignment_method, normalisation_method,
                                         struct_pen_method, struct_pen_coeffs,
                                         non_assignment_penalty,
                                         nonexist_non_assignment_penalty_vals)
    super(OTChemDistanceComputer, self).__init__()

  @classmethod
  def format_params(cls, mass_assignment_method, normalisation_method,
                    struct_pen_method, struct_pen_coeffs,
                    non_assignment_penalty, nonexist_non_assignment_penalty_vals):
    struct_pen_coeffs_ = str([str(s).replace('.', ',') for s in struct_pen_coeffs])
    non_assignment_penalty_ = str(non_assignment_penalty).replace('.', ',')
    nonexist_non_assignment_penalty_vals_ = str([str(s).replace('.', ',') for s in
                                                 nonexist_non_assignment_penalty_vals])
    return '--'.join([mass_assignment_method, normalisation_method, struct_pen_method,
                      struct_pen_coeffs_, non_assignment_penalty_,
                      nonexist_non_assignment_penalty_vals_])

  def evaluate_single(self, x1, x2, *args, **kwargs):
    """ Evaluates the distance between two chemical molecules x1 and x2. """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-nested-blocks
    # The following are the same to compute for all different options
    x1_graph_data = get_graph_data_for_distance_computation(x1)
    x2_graph_data = get_graph_data_for_distance_computation(x2)
    unique_atoms = get_unique_elements(x1_graph_data.atomic_symbols +
                                       x2_graph_data.atomic_symbols)
    atom_dissimilarity_matrices = self._get_atom_dissimilarity_matrices(
        unique_atoms, x1_graph_data, x2_graph_data)
    ret = []
    # structural penalty types -----------------------------------------------------------
    for stru_pen_meth in self.struct_pen_methods:
      struct_pen_matrix = self._get_struct_penalty_matrices(
          unique_atoms, x1_graph_data, x2_graph_data, stru_pen_meth)
      # nonexist_non_assignment_penalty_vals ---------------------------------------------
      for nonexist_nas_pen in self.nonexist_non_assignment_penalty_vals:
        # structural penalty coefficient -------------------------------------------------
        for stru_coef in self.struct_pen_coeffs:
          matching_matrix = \
              self._get_matching_matrix_from_dissimilarity_matrices_and_coeffs(
                  self.non_assignment_penalty, nonexist_nas_pen, stru_coef,
                  struct_pen_matrix, *atom_dissimilarity_matrices)
          matching_matrix[np.logical_not(np.isfinite(matching_matrix))] = \
              REPLACE_COST_INF_WITH
          # mass_assignment_methods ------------------------------------------------------
          for mass_asgn_meth in self.mass_assignment_methods:
            x1_masses = self._get_mass_vector(x1_graph_data, mass_asgn_meth)
            x2_masses = self._get_mass_vector(x2_graph_data, mass_asgn_meth)
            x1_sink_masses = [sum([mass for idx, mass in enumerate(x2_masses)
                                   if x2_graph_data.atomic_symbols[idx] == curr_symbol])
                              for curr_symbol in unique_atoms]
            x2_sink_masses = [sum([mass for idx, mass in enumerate(x1_masses)
                                   if x1_graph_data.atomic_symbols[idx] == curr_symbol])
                              for curr_symbol in unique_atoms]
            x1_masses_aug = np.array(x1_masses + x1_sink_masses)
            x2_masses_aug = np.array(x2_masses + x2_sink_masses)
            _, soln, emd = opt_transport(x1_masses_aug, x2_masses_aug, matching_matrix)
            if 'none' in self.normalisation_methods:
              ret.append(soln)
            if 'total_mass' in self.normalisation_methods:
              ret.append(emd)
    return ret

  @classmethod
  def _get_struct_penalty_matrices(cls, unique_atoms, graph_data_1, graph_data_2,
                                   struct_pen_method):
    """ Returns the dissimilarity matrices. """
    # pylint: disable=unused-argument
    if struct_pen_method == 'all_bonds':
      mol1_mol2_bond_sim_mat = get_bond_similarity_matrix(graph_data_1.bonds_of_each_atom,
                                                          graph_data_2.bonds_of_each_atom)
      mol1_mol2_bond_dissim_mat = get_matching_matrix_from_similarity_matrix(
          mol1_mol2_bond_sim_mat, 0.0, np.inf)
    elif struct_pen_method == 'bond_frac':
      mol1_mol2_bond_dissim_mat = get_mismatching_bond_frac_matrix(
          graph_data_1.bond_type_counts_of_each_atom,
          graph_data_2.bond_type_counts_of_each_atom)
    else:
      raise ValueError('Unknown struct_pen_method \'%s\'.'%(struct_pen_method))
    return mol1_mol2_bond_dissim_mat

  def _get_atom_dissimilarity_matrices(self, unique_atoms, graph_data_1, graph_data_2):
    """ Returns dissimilarity matrices based on atom type. """
    mol1_mol2_atom_sim_mat, mol1_unique_atom_sim_mat, mol2_unique_atom_sim_mat = \
        self._get_atom_similarity_matrices(unique_atoms, graph_data_1, graph_data_2)
    mol1_mol2_atom_dissim_mat = get_matching_matrix_from_similarity_matrix(
        mol1_mol2_atom_sim_mat, 0.0, np.inf)
    raw_mol1_unique_atom_dissim_mat = get_matching_matrix_from_similarity_matrix(
        mol1_unique_atom_sim_mat, 1.0, np.inf)
    raw_mol2_unique_atom_dissim_mat = get_matching_matrix_from_similarity_matrix(
        mol2_unique_atom_sim_mat, 1.0, np.inf)
    # Checking for atoms in one that do not exist in another.
    mol1_unique_atoms = get_unique_elements(graph_data_1.atomic_symbols)
    mol2_unique_atoms = get_unique_elements(graph_data_2.atomic_symbols)
    mol1_unique_nonexist_multiplier = [False] * len(unique_atoms)
    mol2_unique_nonexist_multiplier = [False] * len(unique_atoms)
    for idx, atom in enumerate(unique_atoms):
      if atom not in mol2_unique_atoms:
        mol1_unique_nonexist_multiplier[idx] = True
      if atom not in mol1_unique_atoms:
        mol2_unique_nonexist_multiplier[idx] = True
    # Construct last block of the matching matrix
    num_unique = len(unique_atoms)
    last_block = np.inf * np.ones((num_unique, num_unique))
    np.fill_diagonal(last_block, 0.0)
    return (mol1_mol2_atom_dissim_mat,
            raw_mol1_unique_atom_dissim_mat, raw_mol2_unique_atom_dissim_mat,
            mol1_unique_nonexist_multiplier, mol2_unique_nonexist_multiplier,
            last_block)

  @classmethod
  def _get_matching_matrix_from_dissimilarity_matrices_and_coeffs(
      cls, non_assignment_penalty, nonexist_non_assignment_penalty,
      struct_pen_coeff, struct_pen_matrix, mol1_mol2_atom_dissim_mat,
      raw_mol1_unique_atom_dissim_mat, raw_mol2_unique_atom_dissim_mat,
      mol1_unique_nonexist_multiplier, mol2_unique_nonexist_multiplier, last_block):
    # pylint: disable=too-many-arguments
    """ Return the matching matrices from the dissimilarity matrices. """
    mol1_unique_multiplier = np.array([
        nonexist_non_assignment_penalty if elem else non_assignment_penalty
        for elem in mol1_unique_nonexist_multiplier])
    mol2_unique_multiplier = np.array([
        nonexist_non_assignment_penalty if elem else non_assignment_penalty
        for elem in mol2_unique_nonexist_multiplier])
    mol1_unique_dissim_mat = raw_mol1_unique_atom_dissim_mat * mol1_unique_multiplier
    mol2_unique_dissim_mat = raw_mol2_unique_atom_dissim_mat * mol2_unique_multiplier
    first_block = mol1_mol2_atom_dissim_mat + \
                  struct_pen_coeff * struct_pen_matrix
    # Stack them together
    matching_matrix = np.vstack(
        (np.hstack((first_block, mol1_unique_dissim_mat)),
         np.hstack((mol2_unique_dissim_mat.T, last_block)))
        )
    return matching_matrix

  @classmethod
  def _get_atom_similarity_matrices(cls, unique_atoms, graph_data_1, graph_data_2):
    """ Return similrity matrices. """
    x1_x2_atom_sim_mat = get_atom_type_similarity_matrix(graph_data_1.atomic_symbols,
                                                         graph_data_2.atomic_symbols)
    x1_unique_atom_sim_mat = get_atom_type_similarity_matrix(graph_data_1.atomic_symbols,
                                                             unique_atoms)
    x2_unique_atom_sim_mat = get_atom_type_similarity_matrix(graph_data_2.atomic_symbols,
                                                             unique_atoms)
    return (x1_x2_atom_sim_mat, x1_unique_atom_sim_mat, x2_unique_atom_sim_mat)

  @classmethod
  def _get_mass_vector(cls, graph_data, mass_assignment_method):
    """ Returns the mass vector. """
    if mass_assignment_method == 'equal':
      ret = [1.0] * graph_data.num_atoms
    elif mass_assignment_method == 'molecular_mass':
      ret = graph_data.atomic_masses
    elif mass_assignment_method == 'sqrt_molecular_mass':
      ret = [np.sqrt(x) for x in graph_data.atomic_masses]
    else:
      raise ValueError('Unknown mass_assignment_method %s.'%(mass_assignment_method))
    return ret

#   @classmethod
#   def _get_mass_vector(cls, graph_data, mass_assignment_method, normalisation_method):
#     """ Returns mass vector. """
#     # pylint: disable=no-else-return
#     # Compute masses
#     if mass_assignment_method == 'equal':
#       ret = [1.0] * graph_data.num_atoms
#     elif mass_assignment_method == 'atomic_mass':
#       ret = graph_data.atomic_masses
#     elif mass_assignment_method == 'sqrt_atomic_mass':
#       ret = [np.sqrt(x) for x in graph_data.atomic_masses]
#     else:
#       raise ValueError('Unknown mass_assignment_method %s.'%(mass_assignment_method))
#     # Normalise
#     if normalisation_method == 'none':
#       pass
#     elif normalisation_method == 'num_carbon_atoms':
#       num_carbon_atoms = sum([elem == 'C' for elem in graph_data.atomic_symbols])
#       -- N.B: added a +1 here to avoid zero division
#       ret = [x/float(1 + num_carbon_atoms) for x in ret]
#     elif normalisation_method == 'molecular_mass':
#       tot_molecular_mass = sum(graph_data.atomic_masses)
#       ret = [x/float(tot_molecular_mass) for x in ret]
#     return ret
#
  def get_num_distances(self):
    """ Return the number of distances. """
    if self._num_distances is None:
      self._num_distances = len(self.struct_pen_methods) * \
                            len(self.nonexist_non_assignment_penalty_vals) * \
                            len(self.struct_pen_coeffs) * \
                            len(self.mass_assignment_methods) * \
                            len(self.normalisation_methods)
    return self._num_distances

  def __repr__(self):
    return 'OTChemDistanceComputer: %s' % self.str_params

