"""
Functions defined on Molecules.

@author: kkorovin@cs.cmu.edu

A list of examples of how add objective functions.
(Some are from rdkit, but new ones should be user-definable.)

"""

from rdkit import Chem
from rdkit_contrib.sascorer import calculateScore as calculateSAScore


def get_objective_by_name(name):
	if name == "sascore":
		return SAScore
	else:
		raise NotImplementedError

def SAScore(mol):
    """ Synthetic accessibility score """
    if isinstance(mol, list):
    	rdkit_mol = Chem.MolFromSmiles(mol[0].smiles)
    else:
    	rdkit_mol = Chem.MolFromSmiles(mol.smiles)
    return calculateSAScore(rdkit_mol)

def SMILES_len(mol):
	return len(mol.smiles)