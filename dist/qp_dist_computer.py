"""
  Harness for matching based distance computation among chemical molecules.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error
# pylint: disable=no-member

from argparse import Namespace
import numpy as np
from rdkit import Chem
from mols.molecule import Molecule
# Local
from dist.dist_computer import ChemDistanceComputer

# Define bond types
SINGLE_BOND = Chem.rdchem.BondType.SINGLE
DOUBLE_BOND = Chem.rdchem.BondType.DOUBLE
TRIPLE_BOND = Chem.rdchem.BondType.TRIPLE
AROMATIC_BOND = Chem.rdchem.BondType.AROMATIC
IONIC_BOND = Chem.rdchem.BondType.IONIC


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

def get_graph_data_for_distance_computation(mol):
  """ Returns graph representation for a molecule. """
  if isinstance(mol, str):
    mol = Molecule(mol)
  rdk_mol = mol.to_rdkit()
  rdk_mol = Chem.AddHs(rdk_mol)
  adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(rdk_mol)
  bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rdk_mol.GetBonds()]
  bond_types = [rdk_mol.GetBondBetweenAtoms(b[0], b[1]).GetBondType() for b in bonds]
  atom_idxs = list(range(len(rdk_mol.GetAtoms())))
  atomic_numbers = [rdk_mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in atom_idxs]
  atomic_symbols = [rdk_mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_idxs]
  atomic_masses = [rdk_mol.GetAtomWithIdx(idx).GetMass() for idx in atom_idxs]
  num_atoms = len(atom_idxs)
  # Return
  graph_data = Namespace(rdk_mol=rdk_mol,
                         adj_matrix=adj_matrix,
                         bonds=bonds,
                         bond_types=bond_types,
                         atom_idxs=atom_idxs,
                         atomic_numbers=atomic_numbers,
                         atomic_symbols=atomic_symbols,
                         atomic_masses=atomic_masses,
                         num_atoms=num_atoms,
                        )
  return graph_data


# Now define the distance computer =======================================================
class QPChemDistanceComputer(ChemDistanceComputer):
  """ A distance between chemical molecules based on Quadratic Programming. """

  def __init__(self, struct_pen_coeffs, mislabel_pen_coeffs=None,
               mass_assignment_methods='equal-atomic_mass-sqrt_atomic_mass',
               non_assignment_penalty=1.0, nonexist_non_assignment_penalty_vals=1.0):
    """ Constructor.
        struct_pen_coeffs: A list of coefficients for the structural penalty term.
        mislabel_pen_coeffs: A list of coefficients for the mislabel penalty.
        mass_assignment_method: A string indicating how the masses should be assigned
                                to each atom. If equal, we will use equal (unit) mass
                                for all atoms. If atomic_mass, we will use atomic mass.
        non_assignment_penalty: The non-assignment penalty.
        nonexist_non_assignment_penalty: The non-assignment penalty if the particular
                                         atom does not exist in the other molecule.
    """
    if mislabel_pen_coeffs is None:
      mislabel_pen_coeffs = 1.0
    if not hasattr(mislabel_pen_coeffs, '__iter__'):
      mislabel_pen_coeffs = [mislabel_pen_coeffs] * len(struct_pen_coeffs)
    if len(struct_pen_coeffs) != len(mislabel_pen_coeffs):
      raise ValueError(('struct_pen_coeffs(%d) and mislabel_pen_coeffs(%d) should be' +
                        'of same length.')%(
                            len(struct_pen_coeffs), len(mislabel_pen_coeffs)))
    if not hasattr(nonexist_non_assignment_penalty_vals, '__iter__'):
      nonexist_non_assignment_penalty_vals = [nonexist_non_assignment_penalty_vals]
    # Assign attributes
    self.struct_pen_coeffs = struct_pen_coeffs
    self.mislabel_pen_coeffs = mislabel_pen_coeffs
    self.mass_assignment_methods = mass_assignment_methods.split('-')
    self.non_assignment_penalty = non_assignment_penalty
    self.nonexist_non_assignment_penalty_vals = nonexist_non_assignment_penalty_vals
    super(QPChemDistanceComputer, self).__init__()

  def evaluate_single(self, x1, x2, *args, **kwargs):
    """ Evaluates the distance between two chemical molecules x1 and x2. """
    x1_graph_data = get_graph_data_for_distance_computation(x1)
    x2_graph_data = get_graph_data_for_distance_computation(x2)
    dissimilarity_matrices = self._get_dissimilarity_matrices(
        x1_graph_data.atomic_symbols, x2_graph_data.atomic_symbols)
    # First iterate through each nonexist_non_assignment_penalty_vals value
    ret = []
    print(self.nonexist_non_assignment_penalty_vals)
    for nonexist_nas_pen in self.nonexist_non_assignment_penalty_vals:
      matching_matrix = \
          self._get_matching_matrix_from_dissimilarity_matrices_and_non_assignment_coeffs(
              self.non_assignment_penalty, nonexist_nas_pen, *dissimilarity_matrices)
      print(x1, x2)
      print(matching_matrix)
      import pdb; pdb.set_trace()
#       for mass_assignment_method in self.mass_assignment_methods:
#         # Total mass vectors
#         x1_masses = self._get_mass_vector(x1_graph_data, mass_assignment_method)
#         x2_masses = self._get_mass_vector(x2_graph_data, mass_assignment_method)
#         for stru_coef, misl_coef in zip(self.struct_pen_coeffs,
#                                         self.mislabel_pen_coeffs):
#           curr_dist = 0.0
#           ret.append(curr_dist)
      return ret

  def _get_dissimilarity_matrices(self, list_of_atoms_1, list_of_atoms_2):
    """ Returns the dissimilarity matrices. """
    x1_x2_sim_mat, unique_atoms, x1_unique_sim_mat, x2_unique_sim_mat = \
        self._get_similarity_matrices(list_of_atoms_1, list_of_atoms_2)
    print('unique_atoms', unique_atoms)
    mol_1_mol_2_dissim_mat = get_matching_matrix_from_similarity_matrix(
        x1_x2_sim_mat, 0.0, np.inf)
    raw_mol_1_unique_dissim_mat = get_matching_matrix_from_similarity_matrix(
        x1_unique_sim_mat, 1.0, np.inf)
    raw_mol_2_unique_dissim_mat = get_matching_matrix_from_similarity_matrix(
        x2_unique_sim_mat, 1.0, np.inf)
    # Checking for atoms in one that do not exist in another.
    x1_unique_atoms = get_unique_elements(list_of_atoms_1)
    x2_unique_atoms = get_unique_elements(list_of_atoms_2)
    mol_1_unique_nonexist_multiplier = [False] * len(unique_atoms)
    mol_2_unique_nonexist_multiplier = [False] * len(unique_atoms)
    for idx, atom in enumerate(unique_atoms):
      if atom not in x2_unique_atoms:
        mol_1_unique_nonexist_multiplier[idx] = True
      if atom not in x1_unique_atoms:
        mol_2_unique_nonexist_multiplier[idx] = True
    # Construct last block of the matching matrix
    num_unique = len(unique_atoms)
    last_block = np.inf * np.ones((num_unique, num_unique))
    np.fill_diagonal(last_block, 0.0)
    return (mol_1_mol_2_dissim_mat,
            raw_mol_1_unique_dissim_mat, raw_mol_2_unique_dissim_mat,
            mol_1_unique_nonexist_multiplier, mol_2_unique_nonexist_multiplier,
            last_block)

  @classmethod
  def _get_matching_matrix_from_dissimilarity_matrices_and_non_assignment_coeffs(
      cls, non_assignment_penalty, nonexist_non_assignment_penalty,
      mol_1_mol_2_dissim_mat,
      raw_mol_1_unique_dissim_mat, raw_mol_2_unique_dissim_mat,
      mol_1_unique_nonexist_multiplier, mol_2_unique_nonexist_multiplier, last_block):
    """ Return the matching matrices from the dissimilarity matrices. """
    mol_1_unique_multiplier = np.array([
        nonexist_non_assignment_penalty if elem else non_assignment_penalty
        for elem in mol_1_unique_nonexist_multiplier])
    mol_2_unique_multiplier = np.array([
        nonexist_non_assignment_penalty if elem else non_assignment_penalty
        for elem in mol_2_unique_nonexist_multiplier])
    mol_1_unique_dissim_mat = raw_mol_1_unique_dissim_mat * mol_1_unique_multiplier
    mol_2_unique_dissim_mat = raw_mol_2_unique_dissim_mat * mol_2_unique_multiplier
    # Stack them together
    matching_matrix = np.vstack(
        (np.hstack((mol_1_mol_2_dissim_mat, mol_1_unique_dissim_mat)),
         np.hstack((mol_2_unique_dissim_mat.T, last_block)))
        )
    return matching_matrix

  @classmethod
  def _get_similarity_matrices(cls, list_of_atoms_1, list_of_atoms_2):
    """ Return similrity matrices. """
    x1_x2_sim_mat = get_atom_type_similarity_matrix(list_of_atoms_1, list_of_atoms_2)
    unique_atoms = get_unique_elements(list_of_atoms_1 + list_of_atoms_2)
    x1_unique_sim_mat = get_atom_type_similarity_matrix(list_of_atoms_1, unique_atoms)
    x2_unique_sim_mat = get_atom_type_similarity_matrix(list_of_atoms_2, unique_atoms)
    return x1_x2_sim_mat, unique_atoms, x1_unique_sim_mat, x2_unique_sim_mat


  @classmethod
  def _get_mass_vector(cls, graph_data, mass_assignment_method):
    """ Returns mass vector. """
    # pylint: disable=no-else-return
    if mass_assignment_method == 'equal':
      return [1.0] * graph_data.num_atoms
    elif mass_assignment_method == 'atomic_mass':
      return graph_data.atomic_numbers
    elif mass_assignment_method == 'sqrt_atomic_mass':
      return [np.sqrt(x) for x in graph_data.atomic_numbers]
    else:
      raise ValueError('Unknown mass_assignment_method %s.'%(mass_assignment_method))

