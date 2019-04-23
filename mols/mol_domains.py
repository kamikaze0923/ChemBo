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

class MolConstraintChecker:
    pass

class MolDomain(Domain):
    """ Domain for Molecules. """
    def __init__(self, mol_type=None,
                 constraint_checker=None,
                 data_source='chembl'):
        """ Constructor. The arguments are all kwd and come from
            domain_config in mol_function_caller.
        """
        self.mol_type = mol_type  # e.g. can be 'drug-like'
        self.constraint_checker = constraint_checker  # TODO: make a from-string constructor
        self.data_source = MolSampler(data_source)
        super(MolDomain, self).__init__()

    def get_type(self):
        """ Returns type of the domain. """
        return "molecule"

    def get_dim(self):
        """ Return dimension. """
        return 1

    def is_a_member(self, point):
        """ Returns true if point is in the domain. """
        return True

        # TODO:
        # if not self.mol_type == point.mol_class:
        #     return False
        # else:
        #     return self.constraint_checker(point)

    def sample(self, n_samples):
        return self.data_source(n_samples)

    @classmethod
    def members_are_equal(cls, point_1, point_2):
        """ Returns true if they are equal. """
        return molecules_are_equal(point_1, point_2)

    def __str__(self):
        """ Returns a string representation. """
        cc_attrs = ""
        if hasattr(self, "constraint_checker") and self.constraint_checker is not None:
            cc_attrs = {key:getattr(self.constraint_checker, key)
                        for key in self.constraint_checker.constraint_names}
        return 'Mol(%s):%s'%(self.mol_type, cc_attrs)


def molecules_are_equal(mol1, mol2):
    # TODO: implement a comparator method
    pass


# API -------------------------------------------------------------------------

def get_mol_domain_from_constraints(mol_type, constraint_dict):
    """ mol_type is the type of the molecule.
      See MolConstraintChecker constructors for args and kwargs.
    """

    #--- TODO constructing constraint_checker ---#
    # .......................................... #
    #--- TODO constructing constraint_checker ---#

    return MolDomain(mol_type, constraint_checker)

