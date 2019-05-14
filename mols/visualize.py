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


def draw_synthesis_path(mol: Molecule):
    """
    :param mol: an optimal molecule, with the following recursive structure of synthesis path:
               syn_path = {mol: mol, ...} | {mol: syn_path, ...}
    :return: save a graphvz DOT source file corresponding to the structure and the rendered pdf
    """
    from graphviz import Digraph

    def add_node_edge(dot: Digraph, root: Molecule):
        if root.begin_flag:  # base case
            dot.node(name=root.to_smiles(), label=root.to_smiles())
            return
        else:  # recursive case
            for inp in root.inputs:
                add_node_edge(dot, inp)
            dot.node(name=root.to_smiles(), label=root.to_smiles())
            for inp in root.inputs:
                dot.edge(tail_name=inp.to_smiles(), head_name=root.to_smiles())

    dot = Digraph(comment="Synthesis path for {}".format(mol.to_smiles()))
    add_node_edge(dot, mol)
    dot.render("test-output/res.gv", view=True)


def draw_synthesis_path_from_dict(root_mol: Molecule, syn_path: dict):
    from graphviz import Digraph

    def add_node_edge(dot: Digraph, layer: dict):
        for k, v in layer.items():
            if isinstance(v, str):  # base case
                dot.node(name=v, label=v)
            else:  # recursive case
                add_node_edge(dot, v)
                dot.node(name=k, label=k)
                for sub_k in v:
                    dot.edge(tail_name=sub_k, head_name=k)

    dot = Digraph(comment="Synthesis path for {}".format(root_mol.to_smiles()))
    add_node_edge(dot, syn_path)
    dot.node(name=root_mol.to_smiles(), label=root_mol.to_smiles())
    for k in syn_path:
        dot.edge(tail_name=k, head_name=root_mol.to_smiles())
    dot.render("test-output/res.gv", view=True)

# def draw_synthesis_path(mol):
#     def compute_depth(syn_path):
#         depth = 1
#         if not mol.begin_flag:
#             for inp, inp_syn_path in syn_path:
#                 inp_depth = compute_depth(inp_syn_path)
#                 depth = max(depth, inp_depth)
#         return depth
#
#     syn_path = mol.get_syn_path()
#     depth = compute_depth(syn_path)  # number of rows to allocate for plotting
#     imgs_per_row = []
#     min_shape = None
#
#     # traverse the synthesis path and append images to imgs_per_row
#     # each row should be concatenated: see
#     # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
#
#     # TODO
#
#     imgs_comb = np.vstack([np.asarray(img.resize(min_shape))
#                                     for img in imgs_per_row])
#     result_img = PIL.Image.fromarray(imgs_comb)
#     return result_img


if __name__ == "__main__":
    # mol = Molecule("CCCC")
    # img = draw_molecule(mol)
    # img.save('./experiments/results/test.png')
    import pickle
    syn_path = pickle.load(open("test_mols/medium_mol.pkl", "rb"))
    # syn_path = pickle.load(open("test_mols/large_mol.pkl", "rb"))
    root_mol = Molecule(smiles="C")
    draw_synthesis_path_from_dict(root_mol, syn_path)
