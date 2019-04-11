"""
Molecule class definition.
@author: kkorovin@cs.cmu.edu

Binds in the rdkit molecule definition.

"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols

class Molecule:
    """
    Class to hold both representations,
    as well as synthesis path, of a molecule.
    """
    def __init__(self, smiles=None, graph=None, conv_enabled=False):
        """Constructor
        Keyword Arguments:
            smiles {[type]} -- [description] (default: {None})
            graph {[type]} -- [description] (default: {None})
            conv_enabled {bool} -- whether to set both smiles and graph
                                     arguments here or lazily defer until called
                                     (default: {False})
        
        Raises:
            ValueError -- if neither a correct smiles string
                            or a rdkit mol are provided
        """
        if conv_enabled:
            if isinstance(smiles, str):
                # also checks if smiles can be parsed
                graph = Chem.MolFromSmiles(smiles)
                assert graph is not None
            elif graph is not None:
                smiles = Chem.MolToSmiles(graph)
            else:
                raise ValueError("Invalid arguments")

        self.smiles = smiles
        self.graph = graph
        self.synthesis_path = []  # list of Reactions
        self.begin_flag = True

    def to_smiles(self):
        if self.smiles is None:
            smiles = Chem.MolToSmiles(self.graph)
        return smiles

    def to_rdkit(self):
        if self.graph is None:
            self.graph = Chem.MolFromSmiles(self.smiles)
        return self.graph

    def to_fingerprint(self, ftype='default'):
        """ Get numeric vectors representing a molecule.
            Can be used in some kernels.
        """
        mol = self.to_rdkit()
        if ftype == 'default':
            """ binary vectors of length 64 """
            fp = FingerprintMols.FingerprintMol(mol)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

    def set_synthesis(self, inputs):
        self.begin_flag = False
        self.inputs = inputs  # list of Molecules

    def get_synthesis_path(self):
        """
        Unwind the synthesis graph until all the inputs have True flags.
        """
        if self.begin_flag:
            return self
        return {inp: inp.get_synthesis_path() for inp in self.inputs}

    def __str__(self):
        return self.smiles

    def __repr__(self):
        return self.smiles


# class Reaction:
#     def __init__(self, inputs, outputs):
#         self.inputs = inputs  # list of Molecules
#         self.outputs = outputs  # list of Molecules


