"""
Novelty experiment
@author: kkorovin@cs.cmu.edu
"""

from datasets.loaders import get_chembl, get_zinc250
from tqdm import tqdm

def compute_novel_percentage(mol_list):
    chembl = get_chembl(max_size=-1)  # smiles list
    chembl = [m.smiles for m in chembl]
    zinc = get_zinc250(max_size=-1)  # smiles list
    zinc = [m.smiles for m in zinc]
    # n_total = len(chembl) + len(zinc)
    n_mols = len(mol_list)
    n_in_data = 0.
    for mol in tqdm(mol_list):
        if (mol in chembl) or (mol in zinc):
            n_in_data += 1
    return 1 - n_in_data / n_mols

def get_smiles_list_from_file(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            if 'result:' in line:
                smiles = line.split()[-1]
                res.append(smiles)
    return res

if __name__ == "__main__":
    exp_num = '20190518184219'  # TODO: command line reader
    path = f'./experiments/final/chemist_exp_dir_{exp_num}/run_log'
    mol_list = get_smiles_list_from_file(path)
    perc = compute_novel_percentage(mol_list)
    print('Percentage of proposed molecules that are novel: {:.3f}'.format(perc))
