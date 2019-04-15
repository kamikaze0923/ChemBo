"""
  From utilities for computing the distance.
  -- kirthevasank
"""

# pylint: disable=invalid-name
# pylint: disable=import-error

from argparse import Namespace
import numpy as np
# Local
from rdkit import Chem


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

