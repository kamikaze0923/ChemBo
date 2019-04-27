"""
Molecular domain
@author: kkorovin@cs.cmu.edu

An "optimization analog" of Molecule class
that is ties together Molecules and 
optimization over them.

TODO:
* Make the MolDomain class parametrized,
  with a MolSampler that can sample from variable sources.
  Maybe pass an argument to MolDomain constructor from function caller,
  which has access to 'training data' parameters
  (see chemist_opt.mol_function_caller)

"""

import numpy as np
from dragonfly.exd.domains import Domain
from datasets.loaders import MolSampler
import logging


# Function to be called on CP domain to sample molecules
def sample_mols_from_cartesian_domain(domain, n_samples):
    logging.info(f"Sampling {n_samples} molecules.")
    for dom in domain.list_of_domains:
        if isinstance(dom, MolDomain):
            samples = dom.sample(n_samples)
            return samples
    raise ValueError("MolDomain not in list of domains.")

def get_constraint_checker_from_name(name):
    logging.info(f'Using a constraint checker {name}')
    if name is None:
        # no constraints
        return lambda mol: True
    elif name == 'organic':
        return has_carbon
    else:
        raise NotImplementedError(f'{name}')

class MolDomain(Domain):
    """ Domain for Molecules. """
    def __init__(self, mol_type=None,
                 constraint_checker=None,
                 data_source='chembl', sampling_seed=None):
        """ Constructor. The arguments are all kwd and come from
            domain_config in mol_function_caller.

            mol_type -- [TODO] e.g. can be 'drug-like'
            constraint_checker -- [TODO] check it gets used in a reasonable way
            data_source, sampling_seed -- MolSampler parameters
        """
        self.mol_type = mol_type
        self.constraint_checker = get_constraint_checker_from_name(constraint_checker)
        self.data_source = MolSampler(data_source, sampling_seed)
        super(MolDomain, self).__init__()

    def get_type(self):
        """ Returns type of the domain. """
        return "molecule"

    def get_dim(self):
        """ Return dimension. """
        return 1

    def is_a_member(self, molecule):
        """ Returns true if point is in the domain. """
        return self.constraint_checker(molecule)
        # TODO: add mol_type?
        # if not self.mol_type == point.mol_class:
        #     return False
        # else:
        #     return self.constraint_checker(point)

    def sample(self, n_samples):
        return self.data_source(n_samples)

    @classmethod
    def members_are_equal(cls, point_1, point_2):
        """ Technically, because SMILES are not unique,
            this may sometimes give false negatives.
            TODO: graph structure matching?
        """
        return mol1.to_smiles() == mol2.to_smiles()

    def __str__(self):
        """ Returns a string representation. """
        cc_attrs = ""
        if hasattr(self, "constraint_checker") and self.constraint_checker is not None:
            cc_attrs = {key:getattr(self.constraint_checker, key)
                        for key in self.constraint_checker.constraint_names}
        return 'Mol(%s):%s'%(self.mol_type, cc_attrs)


# Different constraint checker functions(Molecule -> bool) --------------------

def has_carbon(mol):
    rdk = mol.to_rdkit()
    atomic_symbols = [rdk.GetAtomWithIdx(idx).GetSymbol() for idx in range(len(rdk.GetAtoms()))]
    return ('C' in atomic_symbols)

