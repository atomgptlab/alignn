# ALIGNN & ALIGNN-FF

[![PyPI](https://badge.fury.io/py/alignn.svg)](https://badge.fury.io/py/alignn)
[![Build](https://github.com/atomgptlab/alignn/actions/workflows/main.yml/badge.svg)](https://github.com/atomgptlab/alignn/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/atomgptlab/alignn/branch/main/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/atomgptlab/alignn)
[![Downloads](https://pepy.tech/badge/alignn)](https://pepy.tech/project/alignn)

The **Atomistic Line Graph Neural Network (ALIGNN)** introduces a graph convolution layer
that explicitly models both two- and three-body interactions in atomistic systems. It
composes two edge-gated graph convolution layers: one applied to the atomistic line graph
*L(g)* (triplet interactions) and one to the atomistic bond graph *g* (pair interactions).

**ALIGNN-FF** is a universal force-field built on ALIGNN. It was trained on the JARVIS-DFT
dataset (~75,000 materials and 4M+ energy/force entries) and supports any combination of
89 elements. Pretrained models can be fine-tuned or trained from scratch on new data.

![ALIGNN layer schematic](https://github.com/atomgptlab/alignn/blob/develop/alignn/tex/schematic_lg.jpg?raw=true)

## Highlights

- Property prediction (regression & binary classification)
- Multi-output regression (e.g. energy + bandgap + DOS)
- Universal force-field (ALIGNN-FF) with ASE calculator
- Pretrained models on JARVIS, Materials Project, QM9, MOF datasets
- Multi-GPU training via `torchrun` (`DistributedDataParallel`)
- CLI entry points: `train_alignn.py`, `pretrained.py`, `run_alignn_ff.py`

## Quick links

- [Install ALIGNN](installation.md)
- [Colab notebooks](examples-colab.md)
- [Train your first model](training/single-output-regression.md)
- [Use a pretrained model](pretrained/index.md)
- [Run the ALIGNN-FF ASE calculator](usage/ase-calculator.md)
- [Performance benchmarks](performance.md)

## Citing

If you use ALIGNN, please cite the relevant papers listed on the
[References](references.md) page. The primary references are:

1. Choudhary, K., DeCost, B. *Atomistic Line Graph Neural Network for improved
   materials property predictions.* **npj Comput Mater** 7, 185 (2021).
2. Choudhary, K., et al. *Unified graph neural network force-field for the periodic
   table.* **Digital Discovery** (2023).

## Correspondence

Please open issues at <https://github.com/atomgptlab/alignn/issues> or email
`drkamal@jhu.edu`.
