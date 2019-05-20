"""
Build t-SNE visualization for molecular distance
@author: kkorovin@cs.cmu.edu
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from dist.ot_dist_computer import OTChemDistanceComputer
from datasets.loaders import get_chembl_prop, get_chembl
from mols.mol_functions import get_objective_by_name
from mols.mol_kernels import mol_kern_factory

import itertools
import os

VIS_DIR = 'experiments/visualizations'

def make_tsne(func):
    """
    Plot TSNE embeddings colored with property
    for several distance computers.
    """
    n_mols = 400

    dist_computers = [
                OTChemDistanceComputer(mass_assignment_method='equal',
                                        normalisation_method='none',
                                        struct_pen_method='bond_frac'),
                OTChemDistanceComputer(mass_assignment_method='equal',
                                        normalisation_method='total_mass',
                                        struct_pen_method='bond_frac'),
                OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                        normalisation_method='none',
                                        struct_pen_method='bond_frac'),
                OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                        normalisation_method='total_mass',
                                        struct_pen_method='bond_frac')

    ]
    titles = ['Equal mass assign, no norm', 'Equal mass assign, total mass norm',
              'Mol mass assign, no norm',   'Mol mass assign, total mass norm']

    smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
    if func == 'prop':
        smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
    else:
        mols = get_chembl(n_mols=n_mols)
        smile_strings = [mol.to_smiles() for mol in mols]
        func_ = get_objective_by_name(func)
        prop_list = [func_(mol) for mol in mols]


    f, ll_ax = plt.subplots(2, 2, figsize=(15, 15))
    axes = itertools.chain.from_iterable(ll_ax)
    for ind, (ax, dist_computer, title) in enumerate(zip(axes, dist_computers, titles)):
        distances_mat = dist_computer(smile_strings, smile_strings)[0]

        # plot them
        tsne = TSNE(metric='precomputed')
        points_to_plot = tsne.fit_transform(distances_mat)

        ax.set_title(title)
        ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=prop_list, cmap=plt.cm.Spectral, s=9, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig(os.path.join(VIS_DIR, f'tsne_vis_{func}'))
    plt.clf()


def make_pairwise(func):
    n_mols = 50

    if func == 'prop':
        smile_strings, smiles_to_prop = get_chembl_prop(n_mols=n_mols)
        prop_list = [smiles_to_prop[sm] for sm in smile_strings]
    else:
        mols = get_chembl(n_mols=n_mols)
        smile_strings = [mol.to_smiles() for mol in mols]
        func_ = get_objective_by_name(func)
        prop_list = [func_(mol) for mol in mols]

    dist_computers = [
                    OTChemDistanceComputer(mass_assignment_method='equal',
                                            normalisation_method='none',
                                            struct_pen_method='bond_frac'),
                    OTChemDistanceComputer(mass_assignment_method='equal',
                                            normalisation_method='total_mass',
                                            struct_pen_method='bond_frac'),
                    OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                            normalisation_method='none',
                                            struct_pen_method='bond_frac'),
                    OTChemDistanceComputer(mass_assignment_method='molecular_mass',
                                            normalisation_method='total_mass',
                                            struct_pen_method='bond_frac')

                    ]
    titles = ['Equal mass assign, no norm', 'Equal mass assign, total mass norm',
              'Mol mass assign, no norm',   'Mol mass assign, total mass norm']

    f, ll_ax = plt.subplots(2, 2, figsize=(15, 15))
    axes = itertools.chain.from_iterable(ll_ax)
    for ind, (ax, dist_computer, title) in enumerate(zip(axes, dist_computers, titles)):
        distmat = dist_computer(smile_strings, smile_strings)[0]
        xs, ys = [], []
        for i in range(n_mols):
            for j in range(n_mols):
                dist_in_dist = distmat[i, j]
                dist_in_val = np.abs(prop_list[i] - prop_list[j])
                xs.append(dist_in_dist)
                ys.append(dist_in_val)
        ax.set_title(title)  # TODO: parameters of distance
        ax.scatter(xs, ys, s=2, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(os.path.join(VIS_DIR, "dist_vs_value_"+func))
    plt.clf()


def make_pairwise_kernel(kernel_name, func, **kwargs):
    n_mols = 50

    mols = get_chembl(n_mols=n_mols)
    # smile_strings = [mol.to_smiles() for mol in mols]
    func_ = get_objective_by_name(func)
    kernel = mol_kern_factory(kernel_name, **kwargs)
    kern_mat = kernel(mols, mols)
    prop_list = [func_(mol) for mol in mols]

    xs, ys = [], []
    for i in range(n_mols):
        for j in range(n_mols):
            dist_in_dist = kern_mat[i, j]
            dist_in_val = np.abs(prop_list[i] - prop_list[j])
            xs.append(dist_in_dist)
            ys.append(dist_in_val)
    plt.scatter(xs, ys, s=2, alpha=0.6)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(VIS_DIR, f"{kernel_name}_{func}"))
    plt.clf()


if __name__ == "__main__":
    os.makedirs(VIS_DIR, exist_ok=True)
    make_tsne('qed')
    make_tsne('sascore')

    # make_pairwise('prop')
    # make_pairwise('qed')
    # make_pairwise('sascore')

    # make_pairwise_kernel('similarity_kernel', 'qed')
    # make_pairwise_kernel('edgehist_kernel', 'qed', par=2)
    # make_pairwise_kernel('wl_kernel', 'qed',  par=2)
    # make_pairwise_kernel('distance_kernel_expsum', 'qed', dist_computer=OTChemDistanceComputer(), betas=[1.] * 4)

