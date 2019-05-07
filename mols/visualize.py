"""
Visualization tools for molecules
@author: kkorovin@cs.cmu.edu
"""

import PIL
import matplotlib.pyplot as plt
import numpy as np
from myrdkit import Draw
from mols.molecule import Molecule


def visualize_mol(mol: Molecule, path: str):
    """
    Draw a single molecule and save it to `path`
    :param mol: molecule to draw
    :param path: path to save the drawn molecule to
    """
    img = draw_molecule(mol)
    img.save(path)


def draw_molecule(mol: Molecule) -> PIL.Image.Image:
    """
    Draw a single molecule `mol` (make it `PIL.Image.Image`)
    :param mol: molecule to draw
    :return: corresponding image to `mol`
    """
    img = Draw.MolToImage(mol.to_rdkit())
    return img


def draw_synthesis_path(mol):
    def compute_depth(syn_path):
        depth = 1
        if not mol.begin_flag:
            for inp, inp_syn_path in syn_path:
                inp_depth = compute_depth(inp_syn_path)
                depth = max(depth, inp_depth)
        return depth

    syn_path = mol.get_syn_path()
    depth = compute_depth(syn_path)  # number of rows to allocate for plotting
    imgs_per_row = []
    min_shape = None

    # traverse the synthesis path and append images to imgs_per_row
    # each row should be concatenated: see
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

    # TODO

    imgs_comb = np.vstack([np.asarray(img.resize(min_shape))
                                    for img in imgs_per_row])
    result_img = PIL.Image.fromarray(imgs_comb)
    return result_img


if __name__ == "__main__":
    mol = Molecule("CCCC")
    img = draw_molecule(mol)
    img.save('./experiments/results/test.png')
