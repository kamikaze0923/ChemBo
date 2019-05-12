#!/bin/bash
# Runner of all experiments

echo -e "\tStarting running Chemist"
python experiments/run_chemist.py -d chembl_small_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl_small_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl_small_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000

python experiments/run_chemist.py -d chembl_large_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl_large_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl_large_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000

python experiments/run_chemist.py -d chembl -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# python experiments/run_chemist.py -d chembl -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000

echo -e "\tStarting running RandomExplorer"

python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000

# echo -e "\tStarting long RandomExplorer runs"
# python experiments/run_explorer.py -d chembl -s 1 -o qed -b 1000 -i 30 -mpl None
# python experiments/run_explorer.py -d chembl -s 3 -o qed -b 1000 -i 30 -mpl None