"""
Molecule class definition.
@author: kkorovin@cs.cmu.edu

Binds in the rdkit molecule definition.

TODO:
* fix conversion to same-sized molecular fingerprints

"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

GRAPH_LIB = "igraph"  # depending on package for graph kernels
if GRAPH_LIB == "igraph":
    import igraph
else:
    import networkx


class Molecule:
    """
    Class to hold both representations,
    as well as synthesis path, of a molecule.
    """
    def __init__(self, smiles=None, rdk=None, conv_enabled=False):
        """Constructor
        Keyword Arguments:
            smiles {str} -- SMILES representation of a molecule (default: {None})
            rdk {rdkit Mol} -- molecule as an RDKit object (default: {None})
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
                rdk = Chem.MolFromSmiles(smiles)
                assert rdk is not None
            elif rdk is not None:
                smiles = Chem.MolToSmiles(rdk)
            else:
                raise ValueError("Invalid arguments")

        self.smiles = smiles
        self.rdk = rdk
        self.graph = None  # should be obtained from rdk when needed
        self.synthesis_path = []  # list of Reactions
        self.begin_flag = True

    def to_smiles(self):
        if self.smiles is None:
            smiles = Chem.MolToSmiles(self.rdk)
        return smiles

    def to_rdkit(self):
        if self.rdk is None:
            self.rdk = Chem.MolFromSmiles(self.smiles)
        return self.rdk

    def to_graph(self, gformat="igraph", set_properties=False):
        if self.graph is None:
            if gformat == "igraph":
                self.graph = mol2graph_igraph(self, set_properties)
            elif gformat == "networkx":
                self.graph = mol2graph_networkx(self, set_properties)
            else:
                raise ValueError(f"Graph format {gformat} not supported")
        return self.graph

    def to_fingerprint(self, ftype='default'):
        """ Get numeric vectors representing a molecule.
            Can be used in some kernels.
        """
        mol = self.to_rdkit()
        if ftype == 'default':
            """ binary vectors of length 64
            >>> TODO: is there a better way to get fixed-size vectors?
                (e.g. below, arr may be of size 64 or 2048 for different mols)
            """
            fp = FingerprintMols.FingerprintMol(mol)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr[:64]

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


def mol2graph_igraph(mol, set_properties=False):
    """
    Convert molecule to nx.Graph
    Adapted from
    https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
    """
    mol = mol.to_rdkit()
    admatrix = rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds()]
    adlist = np.ndarray.tolist(admatrix)
    graph = igraph.Graph()
    g = graph.Adjacency(adlist).as_undirected()

    if set_properties:
        for idx in g.vs.indices:
            g.vs[idx][ "AtomicNum" ] = mol.GetAtomWithIdx(idx).GetAtomicNum()
            g.vs[idx][ "AtomicSymbole" ] = mol.GetAtomWithIdx(idx).GetSymbol()
        
        for bd in bondidxs:
            btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
            g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
            # print( bd, mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble() )
    return g


def mol2graph_networkx(mol, set_properties=False):
    """
    Convert molecule to nx.Graph
    Adapted from
    https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
    """
    mol = mol.to_rdkit()
    admatrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds()]
    graph = nx.Graph(admatrix)

    if set_properties:
        for idx in graph.nodes:
            graph.nodes[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
            graph.nodes[idx]["AtomicSymbol"] = mol.GetAtomWithIdx(idx).GetSymbol()

        for bd in bondidxs:
            btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
            graph.edges[bd[0], bd[1]]["BondType"] = str(int(btype))
            # print(bd, m1.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble())
    return graph


# class Reaction:
#     def __init__(self, inputs, outputs):
#         self.inputs = inputs  # list of Molecules
#         self.outputs = outputs  # list of Molecules


