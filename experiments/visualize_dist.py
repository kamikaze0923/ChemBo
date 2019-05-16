"""
Build t-SNE visualization for molecular distance
@author: kkorovin@cs.cmu.edu
"""

import numpy as np
import itertools
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dist.ot_dist_computer import OTChemDistanceComputer
from datasets.loaders import get_chembl_prop, get_chembl
from mols.mol_functions import get_objective_by_name
from datetime import datetime

import os

VIS_DIR = 'experiments/visualizations'

def make_tsne():
    """
    Plot TSNE embeddings colored with property
    for several distance computers.
    """
    dist_computers = [
        OTChemDistanceComputer(), # default parameters
        OTChemDistanceComputer(mass_assignment_method='equal',
                               nonexist_non_assignment_penalty_vals=[1, 5, 10]),
        OTChemDistanceComputer(normalisation_method='atomic_mass'),
    ]

    smile_strings, smiles_to_prop = get_chembl_prop(n_mols=200)
    prop_list = [smiles_to_prop[sm] for sm in smile_strings]

    for dist_computer in dist_computers:
        distances_mat = np.mean(dist_computer(smile_strings, smile_strings), axis=0)
        # plot them
        tsne = TSNE(metric='precomputed')
        points_to_plot = tsne.fit_transform(distances_mat)
        plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral)
        print(str(dist_computer))
        plt.savefig(os.path.join(VIS_DIR, str(dist_computer)))


def make_pairwise(func, n_mols, to_randomize=True):


    if func == 'prop':
        smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
    else:
        n_mols_to_get = 5 * n_mols if to_randomize else n_mols
        mols = get_chembl(n_mols=n_mols_to_get)
        np.random.shuffle(mols)
        mols = mols[:n_mols]
        smile_strings = [mol.to_smiles() for mol in mols]
        func_ = get_objective_by_name(func)
        prop_list = [func_(mol) for mol in mols]

    dist_computer = OTChemDistanceComputer()  # <-- default computer
    dists = dist_computer(smile_strings, smile_strings)

    num_rows = max(2, int(np.ceil(dist_computer.get_num_distances() / 4.0)))
    print(num_rows)
    f, ll_ax = plt.subplots(num_rows, 4, figsize=(15, 15))
    axes = itertools.chain.from_iterable(ll_ax)
    for ind, (ax, distmat) in enumerate(zip(axes, dists)):

        xs, ys = [], []
        pairs = []
        for i in range(n_mols):
            for j in range(i, n_mols):
                dist_in_dist = distmat[i, j]
                dist_in_val = np.abs(prop_list[i] - prop_list[j])
                xs.append(dist_in_dist)
                ys.append(dist_in_val)
                pairs.append((i,j))
#                 pairs.append('(%d,%d)'%(i,j))

        ax.set_title(f'Distance {ind}')  # TODO: parameters of distance
        if n_mols > 12:
          ax.scatter(xs, ys, s=1, alpha=0.6)
        else:
          for xval, yval, pval in zip(xs, ys, pairs):
            print(xval, yval, pval)
            if pval[0] == pval[1]:
#               ax.scatter([xval], [yval], s=1, alpha=0.8)
              ax.text(xval, yval, '*', fontsize=14)
            else:
              ax.text(xval, yval, '(%d, %d)'%(pval[0], pval[1]))
          ax.set_xlim((0.0, max(xs) * 1.25))
#         ax.set_xticks([])
#         ax.set_yticks([])

    plt.savefig(os.path.join(VIS_DIR, "dist_vs_value_%d_%s_%s"%(n_mols, func,
                             datetime.now().strftime('%m%d%H%M%S'))))
    print(smile_strings, len(smile_strings))


if __name__ == "__main__":
#     n_mols = 5
    n_mols = 100
    os.makedirs(VIS_DIR, exist_ok=True)
    # make_tsne()

#     make_pairwise('prop', n_mols)
    make_pairwise('qed', n_mols)
#     make_pairwise('sascore', n_mols)

