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
* Graphlets do not work
* For fingerprints, do projection

"""

import numpy as np
from typing import List, Union

import graphkernels.kernels as gk

from dragonfly.gp.kernel import Kernel
from mols.molecule import Molecule


MOL_GRAPH_CONT_KERNEL_TYPES = [
    "edgehist_kernel", "vertexhist_kernel", "vehist_kernel"
]
MOL_GRAPH_INT_KERNEL_TYPES = [
    "vvehist_kernel", "edgehistgauss_kerenl",
    "vertexhistgauss_kernel", "vehistgauss_kernel", "georandwalk_kernel", "exprandwalk_kernel",
    "steprandwalk_kernel", "wl_kernel", "graphlet_kernel", "conngraphlet_kernel", "shortestpath_kernel"
]
MOL_FINGERPRINT_KERNEL_TYPES = ["fingerprint_kernel"]
MOL_SIMILARITY_KERNEL_TYPES = ["similarity_kernel"]
MOL_DISTANCE_KERNEL_TYPES = ["distance_kernel"]


def mol_kern_factory(kernel_type:str, *args, **kwargs):
    """
    factory method for generate a proper kernel
    :param kernel_type:
    :return: a proper kernel with `args` and `kwargs` that matches `kernel_type`
    """
    kernel_to_kernel_type = {
        MolGraphKernel: MOL_GRAPH_CONT_KERNEL_TYPES + MOL_GRAPH_INT_KERNEL_TYPES,
        MolFingerprintKernel: MOL_FINGERPRINT_KERNEL_TYPES,
        MolDistanceKernel: MOL_DISTANCE_KERNEL_TYPES
    }
    kernel_type_to_kernel = {
        kernel_type: kernel
        for kernel, kernel_type_list in kernel_to_kernel_type.items()
        for kernel_type in kernel_type_list
    }
    if kernel_type not in kernel_type_to_kernel:
        raise ValueError("Not recognized kernel type: {}".format(kernel_type))
    kernel = kernel_type_to_kernel[kernel_type]
    return kernel(kernel_type, *args, **kwargs)


class MolKernel(Kernel):
    def __init__(self, kernel_type: str, **kwargs):
        self.kernel_type = kernel_type
        super(MolKernel, self).__init__()

    def is_guaranteed_psd(self):
        return False


class MolGraphKernel(MolKernel):
    _kernel_calculator = {
        "edgehist_kernel": gk.CalculateEdgeHistKernel,
        "vertexhist_kernel": gk.CalculateVertexHistKernel,
        "vehist_kernel": gk.CalculateVertexEdgeHistKernel,
        "vvehist_kernel": gk.CalculateVertexVertexEdgeHistKernel,
        "vertexhistgauss_kernel": gk.CalculateVertexHistGaussKernel,
        "vehistgauss_kernel": gk.CalculateVertexEdgeHistGaussKernel,
        "georandwalk_kernel": gk.CalculateGeometricRandomWalkKernel,
        "exprandwalk_kernel": gk.CalculateExponentialRandomWalkKernel,
        "steprandwalk_kernel": gk.CalculateKStepRandomWalkKernel,
        "wl_kernel": gk.CalculateWLKernel,
        "graphlet_kernel": gk.CalculateGraphletKernel,
        "conngraphlet_kernel": gk.CalculateConnectedGraphletKernel,
        "shorestpath_kernel": gk.CalculateShortestPathKernel
    }

    def __init__(self, kernel_type: str, par: Union[int, float], **kwargs):
        """
        :param kernel_type: graph kernel type, refer to "https://github.com/BorgwardtLab/GraphKernels"
        :param par: `int` for integer parametrized graph kernels
                    `float` for float parametrized graph kernels
        """
        super(MolGraphKernel, self).__init__(kernel_type, **kwargs)
        self.set_hyperparams(par=par)
        if kernel_type not in self._kernel_calculator:
            raise ValueError("Unknown kernel_type {}".format(kernel_type))
        self.kernel_calculator = self._kernel_calculator[kernel_type]

    def _child_evaluate(self, X1: List[Molecule], X2: List[Molecule]) -> np.array:
        complete_graph_list = [m.to_graph() for m in X1 + X2]
        if self.kernel_type in MOL_GRAPH_INT_KERNEL_TYPES:
            par = int(self.hyperparams["par"])
        else:
            par = self.hyperparams["par"]
        complete_ker = self.kernel_calculator(complete_graph_list, par=par)
        n1 = len(X1)
        return complete_ker[:n1, n1:]


class MolFingerprintKernel(MolKernel):
    def __init__(self, kernel_type: str, base_kernel: Kernel, **kwargs):
        super(MolFingerprintKernel, self).__init__(kernel_type, **kwargs)
        self.base_kernel = base_kernel

    def is_guaranteed_psd(self):
        return self.base_kernel.is_guaranteed_psd()

    def _get_fps(self, X: List[Molecule]):
        res = np.array([mol.to_fingerprint() for mol in X])
        return res

    def _child_evaluate(self, X1: List[Molecule], X2: List[Molecule]):
        X1 = self._get_fps(X1)
        X2 = self._get_fps(X2)
        return self.base_kernel.evalute(X1, X2)

    def __str__(self):
        return "FingerprintKernel: " + str(self.base_kernel)


class MolDistanceKernel(MolKernel):
    def __init__(self, kernel_type: str, base_kernel: Kernel, **kwargs):
        super(MolDistanceKernel, self).__init__(kernel_type, **kwargs)
        self.base_kernel = base_kernel

    def is_guaranteed_psd(self):
        return self.base_kernel.is_guaranteed_psd()

    def evaluate_from_dists(self, dists: List[np.array]):
        return self.base_kernel.evaluate_from_dists(dists)

    def __str__(self):
        return "MolDistanceKernel: " + str(self.base_kernel)


class MolStringKernel(MolKernel):
    # TODO: implement this
    pass

# class MolFingerprintKernel(MaternKernel):
#     def __init__(self, kernel_type, nu=None, scale=None, dim_bandwidths=None,
#                  kernel_dim=64):
#         super(MolFingerprintKernel, self).__init__(kernel_dim, nu, scale, dim_bandwidths)
#         self.kernel_type = kernel_type
#
#     def _get_fps(self, X):
#         """
#         turn each molecule to its fingerprint representation
#         """
#         res = np.array([mol.to_fingerprint() for mol in X])
#         return res
#
#     def _child_evaluate(self, X1, X2):
#         X1 = self._get_fps(X1)
#         X2 = self._get_fps(X2)
#         return super(MolFingerprintKernel, self)._child_evaluate(X1, X2)
#
#     def __str__(self):
#         return "FingerprintKernel: " + super(FingerprintKernel, self).__str__()


# class MolSimilarityKernel(Kernel):
#     def _get_fps(self, X):
#         """
#         turn each molecule to its fingerprint representation
#         """
#         return [mol.to_fingerprint() for mol in X]
#
#     def _child_evaluate(self, X1, X2):
#         # first generate the distance matrix:
#         dists = []
#         nfps = len(fps)
#         for i in range(1, nfps):
#             sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
#             dists.extend([1-x for x in sims])

