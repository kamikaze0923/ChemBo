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
    print("save to: ", path)
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


class SynPathDrawer(object):
    def __init__(self, mol: Molecule, draw_mode: str):
        """
        :param mol: the molecule to draw synthesis path for
        :param draw_mode: "smiles" | "formula" | "plot" way of plotting each single molecule
        """
        assert draw_mode in ["smiles", "formula", "plot"]
        from graphviz import Digraph
        self._mol = mol
        self._dot = Digraph(comment="Synthesis path for {}".format(mol.to_smiles()))
        self._draw_mode = draw_mode
        self._node_counter = 0
        self._out_path = None
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
        self._dot.edge(tail_name=tail.to_smiles(), head_name=head.to_smiles())

    def _draw_node(self, node: Molecule):
        import os
        self._node_counter += 1
        if self._draw_mode == "smiles":
            self._dot.node(name=node.to_smiles(), label=node.to_smiles())
        elif self._draw_mode == "formula":
            self._dot.node(name=node.to_smiles(), label=node.to_formula())
        elif self._draw_mode == "plot":
            mol_img_path = os.path.join(self._sub_dir, str(self._node_counter))
            visualize_mol(node, path=mol_img_path)
            self._dot.node(name=node.to_smiles(), label="", image=mol_img_path, shape="plaintext")

    def render(self, out_path: str):
        import os
        import shutil
        self._out_path = out_path
        self._sub_dir = os.path.join(out_path, ".tmp")
        try:
            os.makedirs(self._sub_dir, exist_ok=False)
            self._draw(self._mol)
            self._dot.render(self._out_path, view=False)
        finally:
            shutil.rmtree(self._sub_dir)


def draw_synthesis_path_from_dict(root_mol: Molecule, syn_path: dict, out_path: str):
    from graphviz import Digraph
    import os
    import shutil
    # make a subdirectory for all the pictures needed
    sub_dir = os.path.join(out_path, ".tmp")
    os.makedirs(sub_dir, exist_ok=True)

    def add_node_edge(dot: Digraph, layer: dict):
        for k, v in layer.items():
            if isinstance(v, str):  # base case
                visualize_mol(Molecule(smiles=v), os.path.join(sub_dir, v))
                dot.node(name=v, label="", image=os.path.join(sub_dir, v), shape="plaintext")
            else:  # recursive case
                add_node_edge(dot, v)
                visualize_mol(Molecule(smiles=k), os.path.join(sub_dir, k))
                dot.node(name=k, label="", image=os.path.join(sub_dir,k), shape="plaintext")
                for sub_k in v:
                    dot.edge(tail_name=sub_k, head_name=k)

    dot = Digraph(comment="Synthesis path for {}".format(root_mol.to_smiles()))
    add_node_edge(dot, syn_path)
    visualize_mol(root_mol, os.path.join(sub_dir, root_mol.to_smiles()))
    dot.node(name=root_mol.to_smiles(), label="", image=os.path.join(sub_dir, root_mol.to_smiles(), shape="plaintext"))
    for k in syn_path:
        dot.edge(tail_name=k, head_name=root_mol.to_smiles())
    dot.render(out_path, view=False)
    shutil.rmtree(sub_dir)

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
    syn_path = pickle.load(open("test_mols/medium_mol.pkl", "rb"))
    root_mol = Molecule(smiles="C")  # a place holder
    root_mol = smile_synpath_to_mols(root_mol, syn_path)
    drawer = SynPathDrawer(root_mol, "plot")
    drawer.render("test-output")
    # from graphviz import Digraph
    # dot = Digraph(comment="a", format="png")
    # dot.node(name="temp",image="test-output/temp.png", label="", shape="plaintext")
    # dot.render("test-output/res.gv", view=False)
