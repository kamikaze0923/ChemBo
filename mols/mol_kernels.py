"""

Molecular kernels.
To be used as part of CartesianProductKernel

Kernels to be implemented:
* Graph-based
* Fingerprints as vectors
* Fingerprints for molecular similarity
* String-based

@author: kkorovin@cs.cmu.edu

TODO:
* Implement the remaining graph-based kernels

"""

GRAPH_LIB = "igraph"  # depending on package for graph kernels

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
if GRAPH_LIB == "igraph":
    import igraph
else:
    import networkx
import graphkernels.kernels as gk

from dragonfly.gp.kernel import Kernel, MaternKernel
from mols.molecule import Molecule

# Main Kernel class ---------------------------------------------------------

class MolKernel(Kernel):
    def __init__(self, kernel_type, kernel_hyperparams):
        """ cont_par, int_par """
        self.kernel_type = kernel_type
        if kernel_type == "edgehist_kernel":
            self.kernel_func = compute_edgehist_kernel
        elif kernel_type == "wl_kernel":
            self.kernel_func = compute_wl_kernel
        elif kernel_type == "graphlet":
            self.kernel_func = compute_graphlet_kernel
        else:
            raise ValueError('Unknown kernel_type %s.'%kernel_type)
        self.hyperparams = kernel_hyperparams

    def is_guaranteed_psd(self):
        return False

    def _child_evaluate(self, X1, X2):
        """ X1, X2 - lists of Molecules """
        assert isinstance(X1, list)
        assert isinstance(X2, list)
        assert isinstance(X1[0], Molecule)
        assert isinstance(X2[0], Molecule)

        return self.compute_dists(X1, X2)

    def compute_dists(self, X1, X2):
        """
        Given two lists of mols, computes
        all pairwise distances between them
        (of size n1 x n2)
        """
        bigmat = self.kernel_func(X1 + X2, self.hyperparams)
        n1 = len(X1)
        return bigmat[:n1, n1:]


class FingerprintKernel(MaternKernel):
    def __init__(self, nu=None, scale=None, dim_bandwidths=None,
                 kernel_dim=64):
        super(FingerprintKernel, self).__init__(kernel_dim, nu, scale, dim_bandwidths)

    def _get_fps(self, X):
        """
        turn each molecule to its fingerprint representation
        """
        return [mol.to_fingerprint() for mol in X]

    def _child_evaluate(self, X1, X2):
        X1 = self._get_fps(X1)
        X2 = self._get_fps(X2)
        return super(FingerprintKernel, self)._child_evaluate(X1, X2)

    def __str__(self):
        return "FingerprintKernel: " + super(FingerprintKernel, self).__str__()


class MolSimilarityKernel(Kernel):
    def _get_fps(self, X):
        """
        turn each molecule to its fingerprint representation
        """
        return [mol.to_fingerprint() for mol in X]

    def _child_evaluate(self, X1, X2):
        # first generate the distance matrix:
        dists = []
        nfps = len(fps)
        for i in range(1,nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            dists.extend([1-x for x in sims])

# Graph-based kernels ---------------------------------------------------------

def mol2graph_igraph(mol):
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

    ## set properties
    # for idx in g.vs.indices:
    #     g.vs[idx][ "AtomicNum" ] = mol.GetAtomWithIdx(idx).GetAtomicNum()
    #     g.vs[idx][ "AtomicSymbole" ] = mol.GetAtomWithIdx(idx).GetSymbol()
    
    # for bd in bondidxs:
    #     btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
    #     g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
    #     print( bd, mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble() )
    return g


def mol2graph_networkx(mol):
    """
    Convert molecule to nx.Graph
    Adapted from
    https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
    """
    mol = mol.to_rdkit()
    admatrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds()]
    graph = nx.Graph(admatrix)

    for idx in graph.nodes:
        graph.nodes[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
        graph.nodes[idx]["AtomicSymbol"] = mol.GetAtomWithIdx(idx).GetSymbol()

    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
        graph.edges[bd[0], bd[1]]["BondType"] = str(int(btype))
        # print(bd, m1.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble())
    return graph


"""
Kernels available in graphkernels: TODO into functions
    K1 = gk.CalculateEdgeHistKernel(graph_list)
    K2 = gk.CalculateVertexHistKernel(graph_list) 
    K3 = gk.CalculateVertexEdgeHistKernel(graph_list)
    K4 = gk.CalculateVertexVertexEdgeHistKernel(graph_list)
    K5 = gk.CalculateEdgeHistGaussKernel(graph_list)
    K6 = gk.CalculateVertexHistGaussKernel(graph_list)
    K7 = gk.CalculateVertexEdgeHistGaussKernel(graph_list)
    K8 = gk.CalculateGeometricRandomWalkKernel(graph_list)
    K9 = gk.CalculateExponentialRandomWalkKernel(graph_list)
    K10 = gk.CalculateKStepRandomWalkKernel(graph_list)
    K11 = gk.CalculateWLKernel(graph_list)
    K12 = gk.CalculateConnectedGraphletKernel(graph_list, 4)
    K13 = gk.CalculateGraphletKernel(graph_list, 4)
    K14 = gk.CalculateShortestPathKernel(graph_list)
"""

"""
Base class Kernel has a call method
most kernels from graphkernels have only one parameter:
it is either an integer or a continuous quantity
"""

def compute_edgehist_kernel(mols, params):
    """
    Compute edge hist kernel
    Arguments:
            mols {list[Molecule]} -- [description]
    """
    par = params["cont_par"]
    mol_graphs_list = [mol2graph_igraph(m) for m in mols]
    return gk.CalculateEdgeHistKernel(mol_graphs_list,
                                      par=par)

def compute_vertexedgehist_kernel(mols, params):
    """
    Compute edge hist kernel
    Arguments:
            mols {list[Molecule]} -- [description]
    """
    par = params["cont_par"]
    mol_graphs_list = [mol2graph_igraph(m) for m in mols]
    return gk.CalculateVertexEdgeHistKernel(mol_graphs_list,
                                            par=par)

def compute_wl_kernel(mols, params):
    """
    Compute edge hist kernel
    Arguments:
            mols {list[Molecule]} -- [description]
    """
    par = int(params["int_par"])
    mol_graphs_list = [mol2graph_igraph(m) for m in mols]
    return gk.CalculateWLKernel(mol_graphs_list,
                                par=par)

def compute_graphlet_kernel(mols, params):
    par = int(params["int_par"])  # set default to 4
    mol_graphs_list = [mol2graph_igraph(m) for m in mols]
    return gk.CalculateGraphletKernel(mol_graphs_list,
                                      par=par)



# String-based kernels ---------------------------------------------------------



