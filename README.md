# Dragonfly Chemist
DOE framework for joint molecular optimization and synthesis
Author: Ksenia Korovina (kkorovin@cs.cmu.edu)

## Structure of the repo

* `run_chemist.py` script illustrates usage of the classes
* `chemist_opt` directory isolates the Chemist class which performs joint optimization and synthesis. Contains harnesses for calling molecular functions (`MolFunctionCaller`) and handling optimization over molecular domains (`MolDomain`). Calls for `mols` and `explore`.
* `explore` implements the exploration of molecular domain. Calls for `synth`.
* `mols` contains the `Molecule` class and a few example of objective function definitions, as well as implementations of molecular versions of all components needed for BO to work: MolGP class and molecular kernels.
* `synth` is responsible for performing forward synthesis (using a third-party repo).
* `rdkit_contrib` is an extension to rdkit that provides computation of a few molecular scores.

## Getting started

Install `dragonfly`:
```bash
pip install dragonfly-opt -v
```
