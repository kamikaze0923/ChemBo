"""
Molecule and Reaction classes definitions.
@author: kkorovin@cs.cmu.edu

Molecule provides conversions between string, graph-based
and rdkit.Mol representations of a molecule.
Reaction wraps together reagents, conditions (e.g. reactants),
and products of the reaction (need to be set).

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


class Molecule(object):
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
        """
        Converter to rdkit library format, which is
        used for computation of molecular properties
        and for synthesis. Performs a validity check.

        Returns:
            rdkit.Mol -- molecule in RDKit format
        
        Raises:
            ValueError -- if SMILES cannot be decoded
                        into a chemically valid molecule.
        """
        if self.rdk is None:
            self.rdk = Chem.MolFromSmiles(self.smiles)
        if self.rdk is None:
            raise ValueError(f"Molecule {self.smiles} is not valid.")
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


class Reaction(object):
    def __init__(self, inputs, products=None, conditions=None):
        """Class to represent a chemical reaction.

        Reactants vs reagents: former contribute atoms, latter don't.

        Arguments:
            inputs {list[Molecule]} -- list of reactants

        Keyword Arguments:
            products {[type]} -- predicted results of reaction,
                ranked by likelihood (default: {None})
            conditions {[type]} -- reagents (default: {None})
        """
        self.inputs = inputs
        self.products = products
        self.conditions = conditions

    def set_products(self, ranked_outcomes):
        if self.products is None:
            self.products = ranked_outcomes


