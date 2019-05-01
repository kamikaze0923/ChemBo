"""
Visualization tools for molecules
@author: kkorovin@cs.cmu.edu
"""

import PIL
import matplotlib.pyplot as plt
from myrdkit import Draw
from mols.molecule import Molecule


def visualize_mol(mol, path):
    img = draw_molecule(mol)
    img.save(path)

def draw_molecule(mol):
    """ Elementary drawing utility.

    Arguments:
        mol {Molecule} -- molecule to draw
    Returns:
        PIL.Image.Image
    """
    img = Draw.MolToImage(mol.to_rdkit())
    return img

def draw_synthesis_path(mol):
    syn_path = mol.get_syn_path()

    def compute_depth(syn_path):
        depth = 1
        if not mol.begin_flag:
            for inp, inp_syn_path in syn_path:
                inp_depth = compute_depth(inp_syn_path)
                depth = max(depth, inp_depth)
        return depth
    # number of rows to allocate for plotting:
    depth = compute_depth(syn_path)

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
