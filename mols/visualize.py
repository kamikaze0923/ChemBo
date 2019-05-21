"""
Visualization tools for molecules
@author: kkorovin@cs.cmu.edu

TODO: change to proper saving paths
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


class SynPathDrawer(object):
    def __init__(self, mol: Molecule, draw_mode: str):
        """
        :param mol: the molecule to draw synthesis path for
        :param draw_mode: "smiles" | "formula" | "plot" way of plotting each single molecule

        Examples::

            >>> drawer = SynPathDrawer(root_mol, "smiles")  # or "formula" or "plot"
            >>> drawer.render("some_output_dir/some_file_name")  # please, no file extension
        """
        assert draw_mode in ["smiles", "formula", "plot"]
        from graphviz import Digraph
        self._mol = mol
        self._dot = Digraph(comment="Synthesis path for {}".format(mol.to_smiles()), format="png")
        self._draw_mode = draw_mode
        self._node_counter = 0
        self._sub_dir = None

    def _draw(self, root: Molecule):
        if root.begin_flag:  # base case
            self._draw_node(root)
        else:
            for inp in root.inputs:
                self._draw(inp)
            self._draw_node(root)
            for inp in root.inputs:
                self._draw_edge(tail=inp, head=root)

    def _draw_edge(self, tail: Molecule, head: Molecule):
        self._dot.edge(tail_name=str(id(tail)), head_name=str(id(head)))

    def _draw_node(self, node: Molecule):
        import os
        self._node_counter += 1
        if self._draw_mode == "smiles":
            self._dot.node(name=str(id(node)), label=node.to_smiles())
        elif self._draw_mode == "formula":
            self._dot.node(name=str(id(node)), label=node.to_formula())
        elif self._draw_mode == "plot":
            mol_img_path = os.path.join(self._sub_dir, str(self._node_counter) + ".png")
            visualize_mol(node, path=mol_img_path)
            self._dot.node(name=str(id(node)), label="", image=mol_img_path, shape="plaintext")

    def render(self, out_path: str):
        """
        :param out_path: desired path + filename WITHOUT extension
        """
        import os
        import shutil
        self._sub_dir = os.path.join(os.path.dirname(out_path), ".tmp")
        try:
            os.makedirs(self._sub_dir, exist_ok=False)
            self._draw(self._mol)
            self._dot.render(out_path + ".gv", view=False)
        finally:
            shutil.rmtree(self._sub_dir)


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
    from mols.molecule import smile_synpath_to_mols
    best_mol = pickle.load(open("./mols/best_molecule.pkl", "rb"))
    best_mol = smile_synpath_to_mols(Molecule(smiles="CC(=O)Cc1cc(O)c(C(C)(C)C)c2oc(C)cc(=O)c12"), best_mol)
    drawer = SynPathDrawer(best_mol, "plot")
    drawer.render("plot_plot")  
