![alt text](https://github.com/atomgptlab/alignn/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/atomgptlab/alignn/branch/main/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/atomgptlab/alignn)
[![PyPI version](https://badge.fury.io/py/alignn.svg)](https://badge.fury.io/py/alignn)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/atomgptlab/alignn)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/atomgptlab/alignn)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/atomgptlab/alignn)
[![Downloads](https://pepy.tech/badge/alignn)](https://pepy.tech/project/alignn)

> 📖 **Full documentation:** https://atomgptlab.github.io/alignn/

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Colab notebooks](#colab)
* [Pre-trained models](#pretrained)
* [JARVIS-ALIGNN webapp](#webapp)
* [ALIGNN-FF & ASE Calculator](#alignnff)
* [Peformances on a few datasets](#performances)
* [Useful notes](#notes)
* [References](#refs)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)

<a name="intro"></a>
# ALIGNN & ALIGNN-FF (Introduction)

The Atomistic Line Graph Neural Network ([paper](https://www.nature.com/articles/s41524-021-00650-1)) introduces a graph convolution layer that explicitly models both two- and three-body interactions in atomistic systems. The ALIGNN-FF variant ([paper](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b)) extends this to a force-field for structurally and chemically diverse systems across 89 elements.

See [docs/index.md](docs/index.md) for the full introduction.

![ALIGNN layer schematic](https://github.com/atomgptlab/alignn/blob/develop/alignn/tex/schematic_lg.jpg)

> ⚡ **Pure PyTorch — DGL is no longer required.** ALIGNN now runs fully in
> native PyTorch. Neighbor lists, line graphs, and batched readout are all
> built with plain torch tensor/scatter ops via
> [`alignn/torch_graph_builder.py`](alignn/torch_graph_builder.py), so you can
> train and run inference without installing DGL. To use the pure path, set the
> model name to the `*_pure` variant (e.g. `alignn_atomwise_pure`) and
> `neighbor_strategy` to `"pure_torch"` in your config. The example configs and
> tests in this repository already default to this pure-PyTorch path.

<a name="install"></a>
## Installation

See [docs/installation.md](docs/installation.md) for conda, GitHub, and pip installation methods.

<a name="example"></a>
## Examples

See [docs/training/](docs/training/) for dataset format and training examples:

- [Dataset format](docs/training/dataset-format.md)
- [Single-output regression](docs/training/single-output-regression.md)
- [Classification](docs/training/classification.md)
- [Multi-output regression](docs/training/multi-output-regression.md)
- [Force-field training](docs/training/force-field.md)
- [Multi-GPU training](docs/training/multi-gpu.md)

<a name="colab"></a>
## Colab notebooks

Ready-to-run notebooks covering property prediction, force-field
training, and pretrained-model usage. Click a badge to open in Colab.

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg

| Notebook | Open in Colab | Description |
| --- | --- | --- |
| Regression task (graph-wise prediction) | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/alignn_jarvis_leaderboard.ipynb) | Single-output regression for 2D-material exfoliation energies. |
| ML force-field training from scratch | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Train_ALIGNNFF_Mlearn.ipynb) | Train an ALIGNN-FF force field for Silicon. |
| ALIGNN-FF: relaxation, EV curve, phonons, interfaces | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb) | Pretrained ALIGNN-FF for relaxation, EV curves, phonons, and interfaces. |
| Scaling / timing comparison | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Timing_uMLFF.ipynb) | Scaling/timing analysis of universal MLFFs. |
| Melt-Quench MD | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Fast_Melt_Quench.ipynb) | Generate amorphous structures via molecular dynamics. |
| Miscellaneous training tasks | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb) | Single-output, multi-output (phonon/electron DOS), classification, and pretrained usage. |
| Superconductor Tc | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Sc.ipynb) | Train a model for superconductor transition temperature. |
| Build `id_prop.json` from VASP runs | [![Open In Colab][colab-badge]](https://colab.research.google.com/gist/knc6/5513b21f5fd83a7943509ffdf5c3608b/make_id_prop.ipynb) | Compile `vasprun.xml` files into `id_prop.json` for ALIGNN-FF training. |
| LAMMPS MD with ALIGNN-FF (`pair_alignn`) | [![Open In Colab][colab-badge]](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_FF_LAMMPS_Colab.ipynb) | Build LAMMPS with the native `pair_alignn` style and run NVE / melt-quench MD with the default ALIGNN-FF `mps` force field. |

<a name="pretrained"></a>
## Using pre-trained models

See [docs/pretrained/](docs/pretrained/):

- [Property predictor](docs/pretrained/property-predictor.md)
- [ALIGNN-FF](docs/pretrained/alignn-ff.md)

<a name="webapp"></a>
## Web-apps

See [docs/usage/webapps.md](docs/usage/webapps.md). Direct links: [AtomGPT ALIGNN app](https://atomgpt.org/alignn), [ALIGNN-FF app](https://atomgpt.org/alignn_ff_dynamics).

<a name="alignnff"></a>
## ALIGNN-FF ASE Calculator

```python
from ase.build import bulk
from alignn.ff.unified_calculator import (
    AlignnUnifiedCalculator, AlignnUnifiedConfig)

cfg = AlignnUnifiedConfig(
    energy=True, forces=True, stress=True,
    properties=["formation_energy_peratom", "optb88vdw_bandgap"],
)
calc = AlignnUnifiedCalculator(cfg)          # models loaded once, reused

si = bulk("Si", "diamond", a=5.43); si.calc = calc
si.get_potential_energy(); si.get_forces(); si.get_stress()
print(calc.predictions())                    # extra property predictors
```

A single pydantic config selects the outputs (force-field
energy/forces/stress plus any pretrained scalar property predictors).
See [docs/usage/ase-calculator.md](docs/usage/ase-calculator.md) for more,
and the ASE docs page *Calculators → ALIGNN*.

<a name="performances"></a>
## Performances

See [docs/performance.md](docs/performance.md) for benchmark tables on JARVIS-DFT, Materials Project, QM9, hMOF, qMOF, OpenCatalyst, and other datasets. Also see [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard/).

<a name="notes"></a>
## Useful notes

See [docs/notes.md](docs/notes.md) for common pitfalls and FAQs.

<a name="refs"></a>
## References

See [docs/references.md](docs/references.md) for the publication list.

<a name="contrib"></a>
## How to contribute

See [Contribution instructions](https://github.com/atomgptlab/jarvis/blob/master/Contribution.rst) and [docs/contributing.md](docs/contributing.md).

<a name="corres"></a>
## Correspondence

Please report bugs as [GitHub issues](https://github.com/atomgptlab/alignn/issues) or email drkamal@jhu.edu.

<a name="fund"></a>
## Funding support

- [NIST-MGI](https://www.nist.gov/mgi)
- [NIST-CHIPS](https://www.nist.gov/chips)

## Code of conduct

Please see [Code of conduct](https://github.com/atomgptlab/jarvis/blob/master/CODE_OF_CONDUCT.md).
