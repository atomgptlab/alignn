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
* [Examples — train every model type](#example)
* [Reproducing a JARVIS-Leaderboard contribution](#reproduce)
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
## Examples — train every model type

All training recipes live on this page. Each one ships a **self-contained,
runnable example** under [`alignn/examples/recipes/`](alignn/examples/recipes/)
with a `make_toy_dataset.py` (generates a tiny synthetic `id_prop.json`), a
`config_example.json`, and its own detailed `README.md`. Every recipe below runs
in ~1–2 minutes on CPU.

> ⚠️ **The toy datasets are smoke tests, not real models.** They are 40 rattled
> Si cells with synthetic labels, meant only to prove the pipeline runs. For a
> usable model, replace the structures/labels with **real DFT data**
> (thousands → millions of entries), raise `epochs` to **100–300** and
> `batch_size` to **32–64**, and expect to use a GPU. See each recipe's README.

| Recipe | Task | Graph | Example dir |
| --- | --- | --- | --- |
| kNN | scalar property | kNN (cutoff 8) | [`recipes/knn`](alignn/examples/recipes/knn) |
| Radius | scalar property (MD-compatible) | radius (cutoff 5) | [`recipes/radius`](alignn/examples/recipes/radius) |
| Tensor | D-dim response tensor | kNN | [`recipes/tensor`](alignn/examples/recipes/tensor) |
| Spectra | DOS / Raman curve | kNN | [`recipes/spectra`](alignn/examples/recipes/spectra) |
| Force field | energy + forces + stress | radius | [`recipes/forcefield`](alignn/examples/recipes/forcefield) |
| Atomwise | per-atom charge / moment | kNN | [`recipes/atomwise`](alignn/examples/recipes/atomwise) |

Every recipe reads an `id_prop.json`: a JSON list where each entry has a `jid`,
an inline jarvis `Atoms` dict, and the target(s). See
[Dataset format](docs/training/dataset-format.md) for the full spec.

<details>
<summary><b>1. kNN graph — scalar property</b> (formation energy, band gap, Tc, …)</summary>

Wider k-nearest-neighbour graph (`cutoff: 8.0`, `max_neighbors: 12`) — the more
accurate choice for property prediction.

```bash
cd alignn/examples/recipes/knn
python make_toy_dataset.py                    # -> id_prop.json (40 toy entries)
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key target --id_key jid
```
Key knobs: `cutoff: 8.0`, `model.output_features: 1`, `graphwise_weight: 1.0`,
`calculate_gradient: false`. More: [recipes/knn/README.md](alignn/examples/recipes/knn/README.md).
</details>

<details>
<summary><b>2. Radius graph — scalar property</b> (MD-compatible neighbour list)</summary>

Same scalar task, but the fixed-radius graph (`cutoff: 5.0`) that is continuous
under displacement — use it when you need MD-consistency.

```bash
cd alignn/examples/recipes/radius
python make_toy_dataset.py
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key target --id_key jid
```
Key knobs: `cutoff: 5.0` (vs 8.0 for kNN). More: [recipes/radius/README.md](alignn/examples/recipes/radius/README.md).
</details>

<details>
<summary><b>3. Tensor property</b> (dielectric D=9, piezo D=18, elastic D=36)</summary>

Predict a fixed-length response tensor per structure. Target is a length-`D` list.

```bash
cd alignn/examples/recipes/tensor
python make_toy_dataset.py
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key target --id_key jid
```
Key knobs: set `model.output_features` to your tensor dimension (9/18/36) and
match `D` in `make_toy_dataset.py`. More: [recipes/tensor/README.md](alignn/examples/recipes/tensor/README.md).
</details>

<details>
<summary><b>4. Spectra / multi-output curve</b> (eDOS 300, pDOS 200, Raman 200)</summary>

Predict a full curve on a fixed grid. Target is a length-`D` list (one per bin).

```bash
cd alignn/examples/recipes/spectra
python make_toy_dataset.py
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key target --id_key jid
```
Key knobs: `model.output_features` = number of bins (200/300); match `D` in the
toy script. More: [recipes/spectra/README.md](alignn/examples/recipes/spectra/README.md).
</details>

<details>
<summary><b>5. Force field</b> (energy + forces + stress, ALIGNN-FF)</summary>

Train an interatomic potential with energy-conserving (gradient) forces and
stress — usable for relaxation, MD, and LAMMPS (`pair_alignn`).

```bash
cd alignn/examples/recipes/forcefield
python make_toy_dataset.py
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key energy_per_atom --force_key forces --id_key jid
```
Key knobs: `model.calculate_gradient: true`, and the loss mixture
`graphwise_weight` (energy) / `gradwise_weight` (forces) / `stresswise_weight`
(stress). **Energy must be per atom.** More:
[recipes/forcefield/README.md](alignn/examples/recipes/forcefield/README.md).
</details>

<details>
<summary><b>6. Atomwise property</b> (per-atom charges, magnetic moments)</summary>

Predict one value per atom. Target is a length-`Natoms` list under a per-atom key.

```bash
cd alignn/examples/recipes/atomwise
python make_toy_dataset.py
train_alignn.py --root_dir . --config_name config_example.json \
    --output_dir toy_out --target_key target --id_key jid --atomwise_key charges
```
Key knobs: `model.atomwise_output_features: 1`, `atomwise_weight: 1.0`,
`graphwise_weight: 0.0`; pass `--atomwise_key charges`. More:
[recipes/atomwise/README.md](alignn/examples/recipes/atomwise/README.md).
</details>

For the historical per-topic docs see also [docs/training/](docs/training/)
(dataset format, classification, multi-GPU).

<a name="reproduce"></a>
## Reproducing a JARVIS-Leaderboard contribution

Every ALIGNN entry on the [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard/)
ships the exact config, split, and `run.sh` used to produce it, so any result can
be reproduced end to end:

```bash
# 1) install ALIGNN (pure-PyTorch, no DGL needed)
pip install alignn
#    or from source:
git clone https://github.com/atomgptlab/alignn.git
cd alignn && pip install -e . && cd ..

# 2) get the leaderboard (holds every contribution's config + data split + run.sh)
git clone https://github.com/atomgptlab/jarvis_leaderboard.git
cd jarvis_leaderboard
pip install -e .

# 3) pick a contribution and re-run it
#    contributions live under jarvis_leaderboard/contributions/<name>/
ls jarvis_leaderboard/contributions/alignn_model/
#    each folder has: the benchmark CSV, metadata.json, and run.sh
cat jarvis_leaderboard/contributions/alignn_model/run.sh
bash jarvis_leaderboard/contributions/alignn_model/run.sh
```

`run.sh` downloads the benchmark's train/val/test split (from the matching
`jarvis_leaderboard/benchmarks/.../*.json.zip`), writes the `id_prop`/config, and
calls `train_alignn.py` with the same settings that produced the leaderboard
number — so you reproduce the published MAE exactly. To submit a new ALIGNN
result, copy an existing contribution folder, drop in your predictions CSV +
`metadata.json`, and open a PR (see the leaderboard's `CONTRIBUTING`).


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
