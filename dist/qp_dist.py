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
  # Return
  graph_data = Namespace(rdk_mol=rdk_mol,
                         adj_matrix=adj_matrix,
                         bonds=bonds,
                         bond_types=bond_types,
                         atom_idxs=atom_idxs,
                         atomic_numbers=atomic_numbers,
                         atomic_symbols=atomic_symbols,
                        )
  return graph_data


def QPChemDistanceComputer(ChemDistComputer)
  """ A distance between chemical molecules based on Quadratic Programming. """

  def __init__(self, struct_pen_coeffs, mislabel_pen_coeffs=None,
               mass_assignment_method='equal',
               non_assignment_penalty=1.0, nonexist_non_assignment_penalty=1.0):
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
    self.struct_pen_coeffs = struct_pen_coeffs
    self.mislabel_pen_coeffs = mislabel_pen_coeffs
    self.non_assignment_penalty = non_assignment_penalty
    self.nonexist_non_assignment_penalty = nonexist_non_assignment_penalty

