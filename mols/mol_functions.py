"""
Functions defined on Molecules.

@author: kkorovin@cs.cmu.edu

A list of examples of how add objective functions.
(Some are from rdkit, but new ones should be user-definable.)

NOTES:
* List of RDkit mol descriptors:
  https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors

"""

from rdkit import Chem
from rdkit_contrib.sascorer import calculateScore as calculateSAScore
from rdkit.Chem import Descriptors

def get_objective_by_name(name):
    if name == "sascore":
        return SAScore
    elif name == "logp":
        return LogP
    else:
        raise NotImplementedError

def SAScore(mol):
    """ Synthetic accessibility score """
    # if isinstance(mol, list):
    #   rdkit_mol = Chem.MolFromSmiles(mol[0].smiles)
    # else:
    #   rdkit_mol = Chem.MolFromSmiles(mol.smiles)
    if isinstance(mol, list):
        rdkit_mol = mol[0].to_rdkit()
    else:
        rdkit_mol = mol.to_rdkit()
    return calculateSAScore(rdkit_mol)

def LogP(mol):
    """ Synthetic accessibility score """
    if isinstance(mol, list):
        rdkit_mol = mol[0].to_rdkit()
    else:
        rdkit_mol = mol.to_rdkit()
    return Descriptors.MolLogP(rdkit_mol)

def SMILES_len(mol):
    return len(mol.smiles)