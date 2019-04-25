# Dragonfly Chemist

*Authors: Ksenia Korovina (kkorovin@cs.cmu.edu), Celsius Xu*

Dragonfly Chemist is library for joint molecular optimization and synthesis. It is based on Dragonfly - a framework for scalable Bayesian optimization.

## Structure of the repo

* `experiments` package contains experiment scripts. In particular, `run_chemist.py` script illustrates usage of the classes.
* `chemist_opt` package isolates the Chemist class which performs joint optimization and synthesis. Contains harnesses for calling molecular functions (`MolFunctionCaller`) and handling optimization over molecular domains (`MolDomain`). Calls for `mols` and `explore`.
* `explorer` implements the exploration of molecular domain. Currently, a `RandomExplorer` is implemented, which explores reactions randoml, starting from a given pool. Calls for `synth`.
* `mols` contains the `Molecule` class, the `Reaction` class, a few examples of objective function definitions, as well as implementations of molecular versions of all components needed for BO to work: `MolCPGP` and `MolCPGPFitter` class and molecular kernels.
* `synth` is responsible for performing forward synthesis.
* `rdkit_contrib` is an extension to rdkit that provides computation of a few molecular scores (for older versions of `rdkit`).
* `baselines` contains wrappers for models we compare against.

## Getting started

**Python packages.** 

First, set up environment for RDKit and Dragonfly:

```bash
conda create -n chemist-env python=3.6
# optionally: export PATH="/opt/miniconda3/bin:$PATH"
conda activate chemist-env
```

<!-- Install basic requirements with conda:

```bash
while read requirement; do conda install --yes $requirement; done < requirements.txt
```
 -->

First, need to install `eigen3`, `pkg-config`: [see instructions here](https://github.com/BorgwardtLab/GraphKernels). Then install basic requirements with pip (graphkernels already installs igraph and other dependencies):

```bash
sudo apt-get install libeigen3-dev; sudo apt-get install pkg-config  # on Linux
brew install eigen; brew install pkg-config  # on MacOS
pip install -r requirements.txt
```

If the above fails on MacOS (see [stackoverflow](https://stackoverflow.com/questions/16229297/why-is-the-c-standard-library-not-working)), the simplest solution is

```bash
MACOSX_DEPLOYMENT_TARGET=10.9 pip install graphkernels
```

Finally, install `dragonfly`:

```bash
pip install dragonfly-opt -v
```

### Environment

Set PYTHONPATH for imports:

```bash
source setup.sh 
```

### Getting data

ChEMBL data as txt can be found [in kevinid's repo](https://github.com/kevinid/molecule_generator/releases/), [official downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads). ZINC database can be downloaded from [the official site](http://zinc.docking.org/browse/subsets/). Run the following to automatically download the datasets and put them into the right directory:

```bash
bash download_data.sh
```

