"""
Compute statistics for experiment:
novelty, mean and std achieved value.

@author: kkorovin@cs.cmu.edu
"""

from datasets.loaders import get_chembl, get_zinc250
from mols.molecule import Molecule
from mols.mol_functions import get_objective_by_name
import os
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

def compute_sa_score_datasets():
    sas = get_objective_by_name("sascore")
    chembl = get_chembl(max_size=100)
    res = [sas(m) for m in chembl]
    print("ChEMBL: {:.3f} +- std {:.3f}".format(np.mean(res), np.std(res)))
    zinc = get_zinc250(max_size=100)
    res = [sas(m) for m in zinc]
    print("ZINC: {:.3f} +- std {:.3f}".format(np.mean(res), np.std(res)))
    

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

def compute_synthesizability(exp_path):
    sas = get_objective_by_name("sascore")
    mol = None
    with open(os.path.join(exp_path, 'exp_log'), 'r') as f:
        for line in f:
            if 'Resulting molecule' in line:
                mol = Molecule(smiles=line.split()[2])
    if not mol: return
    sa_score = sas(mol)
    return sa_score

def get_max(path):
    res = get_list_from_file(path)
    return max(res)

def format_chem(group):
    return [f"./experiments/final/chemist_exp_dir_{exp_num}/run_log" for exp_num in group]

def format_rand(group):
    return [f"./experiments/final/rand_exp_dir_{exp_num}/opt_vals" for exp_num in group]


if __name__ == "__main__":
    # from mols.mol_functions import get_objective_by_name
    # from mols.molecule import Molecule
    # exp_num = 20190520051810
    # lst = get_smiles_list_from_file(f"./experiments/final/chemist_exp_dir_{exp_num}/run_log")
    # qed_func = get_objective_by_name('plogp')
    # vals = [qed_func(Molecule(smiles=el)) for el in lst]
    # argmx = np.argmax(vals)
    # print(vals[argmx], lst[argmx])

    # qed_rand_exp_paths = format_rand(["20190522072706", "20190522072835", "20190522073130", "20190522073258",
    #                                   "20190522160909", "20190522161028"])
    # qed_sim_exp_paths = format_chem(["20190518132128", "20190518184219", "20190519053538", "20190519172351"])
    # qed_dist_exp_paths = format_chem(["20190518095359", "20190518182118", "20190518182227", "20190520042603"])

    # plogp_rand_exp_paths = format_rand(["20190522072622", "20190522072737", "20190522072853",
    #                                     "20190522154201", "20190522154310", "20190522154417", "20190522154535"])
    # plogp_sim_exp_paths = format_chem(["20190519053341", "20190520035241", "20190520051810"])
    # plogp_dist_exp_paths = format_chem(["20190520034402", "20190520034422", "20190520041405", "20190518051956_f"]) #"20190520034409", 

    # exp_paths = plogp_dist_exp_paths
    # res = []
    # nov = []
    # for path in exp_paths:
    #     # nov.append(compute_novelty(path))
    #     res.append(get_max(path))
    # # print(f"Novelty percentage {np.mean(nov)}")
    # print("Mean {:.3f} \pm std {:.3f}".format(np.mean(res), np.std(res)/np.sqrt(5)))


    directory = './experiments/results/extra_exps/'
    res = []
    for subdir in os.listdir(directory):
        if subdir.startswith('rand_'):
            sa = compute_synthesizability(os.path.join(directory, subdir))
            if sa: res.append(sa)
    compute_sa_score_datasets()
    print("SA score: {:.3f} +- std {:.3f}".format(np.mean(res), np.std(res)))


    
