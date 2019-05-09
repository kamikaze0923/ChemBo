"""
Build t-SNE visualization for molecular distance
@author: kkorovin@cs.cmu.edu
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dist.ot_dist_computer import OTChemDistanceComputer
from datasets.loaders import get_chembl_prop

import os

VIS_DIR = 'experiments/visualizations'

if __name__ == "__main__":
    os.makedirs(VIS_DIR, exist_ok=True)
    dist_computers = [
        OTChemDistanceComputer(), # default parameters
        OTChemDistanceComputer(mass_assignment_method='equal',
                               nonexist_non_assignment_penalty_vals=[1, 5, 10]),
        OTChemDistanceComputer(normalisation_method='atomic_mass'),
    ]

    smile_strings, smiles_to_prop = get_chembl_prop(n_mols=200)
    for dist_computer in dist_computers:
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
        distances_mat = np.mean(dist_computer(smile_strings, smile_strings), axis=0)
        # plot them
        tsne = TSNE(metric='precomputed')
        points_to_plot = tsne.fit_transform(distances_mat)
        plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral)
        print(str(dist_computer))
        plt.savefig(os.path.join(VIS_DIR, str(dist_computer)))
