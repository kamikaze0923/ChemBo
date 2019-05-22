"""
Compute statistics for experiment:
novelty, mean and std achieved value.

@author: kkorovin@cs.cmu.edu
"""

from datasets.loaders import get_chembl, get_zinc250
from tqdm import tqdm
import numpy as np

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

def get_list_from_file(path):
    res = []
    path = path.replace('run_log', 'exp_log')
    with open(path, 'r') as f:
        if path.endswith('exp_log'):
            # Chemist
            for line in f:
                if line.startswith("#"):
                    curr_max = line.split()[3]
                    curr_max = float(curr_max.split("=")[1][:-1])
                    res.append(curr_max)
        elif path.endswith('opt_vals'):
            # Explorer
            line = f.readline()
            res = [float(v) for v in line.split()]
        else:
            raise ValueError
    return res

def compute_novelty(path):
    mol_list = get_smiles_list_from_file(path)
    perc = compute_novel_percentage(mol_list)
    return perc

def get_max(path):
    res = get_list_from_file(path)
    return max(res)

def format_chem(group):
    return [f"./experiments/final/chemist_exp_dir_{exp_num}/run_log" for exp_num in group]

def format_rand(group):
    return [f"./experiments/final/rand_exp_dir_{exp_num}/opt_vals" for exp_num in group]


if __name__ == "__main__":
    qed_rand_exp_paths = format_rand(["20190521154417", "20190522072706", "20190522072835", "20190522073130", "20190522073258"])
    qed_sim_exp_paths = format_chem(["20190518132128", "20190518184219", "20190519053538", "20190519172351"])
    qed_dist_exp_paths = format_chem(["20190518095359", "20190518182118", "20190518182227", "20190520042603"])

    plogp_rand_exp_paths = format_rand(["20190522072622", "20190522072737", "20190522072853", "20190522073012", "20190522073132"])
    plogp_sim_exp_paths = format_chem(["20190519053341", "20190520035241", "20190520051810"])
    plogp_dist_exp_paths = format_chem(["20190520034409", "20190520034422", "20190520041405"])

    exp_paths = plogp_rand_exp_paths
    res = []
    nov = []
    for path in exp_paths:
        # nov.append(compute_novelty(path))
        res.append(get_max(path))
    # print(f"Novelty percentage {np.mean(nov)}")
    print(f"Mean {np.mean(res)}, std {np.std(res)}")

    
