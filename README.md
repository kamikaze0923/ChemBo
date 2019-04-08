# Dragonfly Chemist
## DOE framework for joint molecular optimization and synthesis

*Authors: Ksenia Korovina (kkorovin@cs.cmu.edu), Celsius Xu*


## Structure of the repo

* `run_chemist.py` script illustrates usage of the classes.
* `chemist_opt` directory isolates the Chemist class which performs joint optimization and synthesis. Contains harnesses for calling molecular functions (`MolFunctionCaller`) and handling optimization over molecular domains (`MolDomain`). Calls for `mols` and `explore`.
* `explore` implements the exploration of molecular domain. Calls for `synth`.
* `mols` contains the `Molecule` class and a few example of objective function definitions, as well as implementations of molecular versions of all components needed for BO to work: `MolCPGP` and `MolCPGPFitter` class and molecular kernels.
* `synth` is responsible for performing forward synthesis (using a third-party repo).
* `rdkit_contrib` is an extension to rdkit that provides computation of a few molecular scores.


## Getting started

**Python packages.** 

First, set up environment for RDKit and Dragonfly:

```bash
conda create -c rdkit -n my-rdkit-env rdkit
conda activate my-rdkit-env
```

Install basic requirements with conda:

```bash
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

or pip:

```bash
pip install -r requirements.txt
```

In addition to these requirements, a `graphkernels` package should be installed. It automatically installs `igraph` and other dependencies. However, it does not install `eigen3`, `pkg-config`, therefore those are included into requirements. Install `graphkernels` via pip (on Mac):

```bash
pip install graphkernels
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
