"""
  Utilities for distance computation.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error
# pylint: disable=no-member

from argparse import Namespace
from myrdkit import Chem

def get_neighbors_and_bond_types(atom_idx, list_of_bonds, atomic_symbols, bond_types):
  """ Returns the bonds for the current atom. """
  if not len(list_of_bonds) == len(bond_types):
    raise ValueError(('list_of_bonds(%d) and bond_types(%d) should be of the same ' +
                      'length.')%(len(list_of_bonds), len(bond_types)))
  ret = []
  for bond, btype in zip(list_of_bonds, bond_types):
    if bond[0] == atom_idx:
      ret.append((atomic_symbols[bond[1]], btype))
    elif bond[1] == atom_idx:
      ret.append((atomic_symbols[bond[0]], btype))
  return ret

def get_bond_type_counts(bond_types):
  """ Returns a count on the number of bonds of each type. """
  count = {}
  for bt in bond_types:
    if bt in count.keys():
      count[bt] += 1
    else:
      count[bt] = 1
  return count

def get_graph_data_for_distance_computation(mol):
  """ Returns graph representation for a molecule. """
  if isinstance(mol, str):
    from mols.molecule import Molecule
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
  bonds_of_each_atom = [
      get_neighbors_and_bond_types(idx, bonds, atomic_symbols, bond_types)
      for idx in range(num_atoms)]
  bond_type_counts_of_each_atom = [
      get_bond_type_counts(bt) for bt in bonds_of_each_atom]
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
                         bonds_of_each_atom=bonds_of_each_atom,
                         bond_type_counts_of_each_atom=bond_type_counts_of_each_atom,
                        )
  return graph_data

