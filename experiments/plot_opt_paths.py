"""
Plot optimization paths from logs
@author: kkorovin@cs.cmu.edu
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# VIS_DIR = 'experiments/results/visualizations'
VIS_DIR = 'experiments/visualizations'


def plot_paths(name, paths_list):
    results = []
    for path in paths_list:
        res = get_list_from_file(path)
        results.append(res)

    # preprocessing
    max_len = max([len(res) for res in results])
    for res in results:
        res.extend( [res[-1]] * (max_len - len(res)) )
        plt.plot(range(20, 20 + len(res)), res)

    # plot eps
    plot_path = os.path.join(VIS_DIR, f"{name}.eps")
    plt.savefig(plot_path, format='eps', dpi=1000)
    plt.clf()

def get_list_from_file(path):
    res = []
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
    return res


if __name__ == "__main__":
    # exp_nums = ["20190518184219", "20190518182118"]
    # paths_list = [f"./experiments/results/final/chemist_exp_dir_{exp_num}/exp_log" for exp_num in exp_nums]

    exp_nums = ["20190521154417"]
    paths_list = [f"./experiments/results/final/rand_exp_dir_{exp_num}/opt_vals" for exp_num in exp_nums]
    name = "explorer_test"
    plot_paths(name, paths_list)
