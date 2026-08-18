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

Every ALIGNN entry on the [JARVIS-Leaderboard](https://atomgptlab.github.io/jarvis_leaderboard/)
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
energy/forces/stress plus any pretrained ALIGNN 2.0 property predictors —
scalar, spectra, or D-dim tensor; radius or kNN graph).
See [docs/usage/ase-calculator.md](docs/usage/ase-calculator.md) for more,
and the ASE docs page *Calculators → ALIGNN*.

<a name="performances"></a>
## Performances

ALIGNN 2.0 benchmarked across single-property, multi-property (spectra / per-atom /
tensor), and interatomic-force-field tasks. Columns compare ALIGNN 2.0 on the radius
and 8 Å kNN graphs against the original ALIGNN and CGCNN; **bold** marks the row best.
Skill is `100 · (1 − MAE / MAD)` vs the mean-absolute-deviation baseline. For the live,
continually-updated numbers see the
[JARVIS-Leaderboard](https://atomgptlab.github.io/jarvis_leaderboard/).

<details>
<summary><b>Full benchmark table (54 tasks)</b></summary>

**(a) Single-property prediction — test MAE**

| # | Task (unit) | N tr/val/te | ALIGNN 2.0 (radius) | ALIGNN 2.0 (kNN) | orig. ALIGNN | CGCNN | Baseline (MAD) | Skill % |
|---|---|---|---|---|---|---|---|---|
| 1 | formation_energy (eV/atom) | 44569/5572/5572 | 0.0316 | **0.0307** | 0.0331 | 0.0551 | 0.876 | 96.5 |
| 2 | optb88vdw_total_energy (eV/atom) | 44569/5572/5572 | 0.0321 | **0.0314** | 0.0367 | 0.0584 | 1.786 | 98.2 |
| 3 | optb88vdw_bandgap (eV) | 44569/5572/5572 | 0.1314 | **0.1306** | 0.1423 | 0.1857 | 0.999 | 86.9 |
| 4 | mbj_bandgap (eV) | 14535/1817/1815 | **0.2721** | 0.2730 | 0.3104 | 0.3261 | 1.765 | 84.6 |
| 5 | QM9 HOMO–LUMO gap (eV) | 110,000/10,000/10,829 | **0.031** |   | 0.0345 |   | 0.834 | 96.3 |
| 6 | QMOF bandgap (eV) | 16,340/2042/2042 | 0.208 |   | **0.202** |   | 0.946 | 78.7 |
| 7 | ehull (eV/atom) | 44290/5537/5537 | **0.0576** | 0.0590 | 0.0763 | 0.0590 | 1.148 | 95.0 |
| 8 | bulk_modulus_kv (GPa) | 15744/1968/1968 | 9.885 | **9.302** | 10.399 | 11.015 | 53.76 | 82.7 |
| 9 | shear_modulus_gv (GPa) | 15744/1968/1968 | 9.063 | **8.825** | 9.476 | 10.079 | 27.06 | 67.4 |
| 10 | magmom_oszicar (μ_B) | 41766/5222/5222 | 0.2608 | **0.2567** | 0.2574 | 0.3065 | 1.254 | 79.5 |
| 11 | slme (%) | 7250/906/906 | 4.493 | **4.447** | 4.521 | 5.014 | 11.21 | 60.3 |
| 12 | spillage | 9101/1137/1137 | 0.3527 | **0.3456** | 0.3510 | 0.3844 | 0.518 | 33.3 |
| 13 | kpoint_length_unit (Å) | 44313/5540/5539 | 9.699 | **9.342** | 9.515 | 9.875 | 17.94 | 47.9 |
| 14 | encut (eV) | 44308/5539/5539 | 131.81 | **128.08** | 133.80 | 134.83 | 262.6 | 51.2 |
| 15 | epsx | 35592/4449/4449 | 20.705 | **20.139** | 20.394 | 22.199 | 57.45 | 64.9 |
| 16 | epsy | 35592/4449/4449 | 20.088 | **19.829** | 19.999 | 21.787 | 57.32 | 65.4 |
| 17 | epsz | 35592/4449/4449 | 19.633 | **19.453** | 19.568 | 21.121 | 55.79 | 65.1 |
| 18 | mepsx | 13447/1681/1681 | 24.646 | **23.847** | 24.046 | 26.929 | 63.39 | 62.4 |
| 19 | mepsy | 13447/1681/1681 | 23.823 | 24.044 | **23.648** | 26.556 | 63.68 | 62.6 |
| 20 | mepsz | 13447/1681/1681 | **23.247** | 23.531 | 23.731 | 26.629 | 60.71 | 61.7 |
| 21 | dfpt_piezo_max_dij (pC/N) | 2677/334/334 | 12.603 | **12.498** | 20.570 | 18.392 | 22.69 | 44.9 |
| 22 | dfpt_piezo_max_dielectric | 3764/470/470 | 26.823 | **24.305** | 28.151 | 30.961 | 43.91 | 44.7 |
| 23 | exfoliation_energy (meV/atom) | 650/81/81 | 40.272 | **37.628** | 52.703 | 45.762 | 61.03 | 38.3 |
| 24 | max_efg (10²¹V/m^2) | 9493/1186/1186 | 19.802 | 19.248 | **19.121** | 22.957 | 44.46 | 56.7 |
| 25 | avg_elec_mass (m_e) | 14114/1764/1764 | 0.0837 | **0.0810** | 0.0853 | 0.0921 | 0.225 | 64.1 |
| 26 | avg_hole_mass (m_e) | 14114/1764/1764 | 0.1299 | 0.1240 | **0.1239** | 0.1406 | 0.399 | 68.9 |
| 27 | n_Seebeck (\muV/K) | 18568/2321/2321 | 41.524 | **40.346** | 40.921 | 45.660 | 111.5 | 63.8 |
| 28 | n_powerfact (\muW/mK^2) | 18568/2321/2321 | 469.07 | 451.90 | **442.30** | 485.59 | 709.2 | 36.3 |
| 29 | ph_heat_capacity (J/mol/K) | 9644/1205/1205 | **9.577** | – | 9.606 | 12.936 | 40.16 | 76.2 |
| 30 | Thermal Cond. (log₁₀κ_L) | 3227/–/404 | 0.376 | **0.362** | – | – | 0.597 | 39.4 |
| 31 | Tc_supercon (K) | 556/30/30 | 1.637 | **1.490** | 2.032 | – | 2.723 | 45.3 |
| 32 | Tc_supercon_hydride (K) | 763/95/95 | 9.937 | **9.425** | – | – | 33.56 | 71.9 |
| 33 | Tc_supercon_ hydride_plus_bulk (K) | 1595/199/199 | 8.670 | **8.407** | – | – | 22.33 | 62.3 |
| 34 | alex_supercon Tc (K) | 6592/824/825 | **0.883** |   |   |   | 2.818 | 68.7 |
| 35 | alex_supercon N(E_F) (states/eV) | 6592/824/825 | **0.821** |   |   |   | 1.559 | 47.3 |
| 36 | alex_supercon θ_D (K) | 6592/824/825 | **11.33** |   |   |   | 80.30 | 85.9 |
| 37 | alex_supercon λ | 6592/824/825 | **0.0707** |   |   |   | 0.194 | 63.6 |
| 38 | alex_supercon ω_log (K) | 6592/824/825 | **20.31** |   |   |   | 55.37 | 63.3 |

**(b) Multi-property — spectra / per-atom / tensor; held-out MAE (col. "radius")**

| # | Task (unit) | N tr/val/te | ALIGNN 2.0 (radius) | ALIGNN 2.0 (kNN) | orig. ALIGNN | CGCNN | Baseline (MAD) | Skill % |
|---|---|---|---|---|---|---|---|---|
| 39 | eDOS, electronic DOS (D=300) | 4103/227/229 | **0.0138** |   |   |   | 0.0213 | 35.2 |
| 40 | pDOS, phonon DOS (D=200) | 4103/227/229 | 0.0819 |   |   |   | 0.117 | 29.8 |
| 41 | Raman spectrum (D=200) | 4059/507/508 | 0.0378 | **0.0326** |   |   | 0.0497 | 34.4 |
| 42 | Bader charge, per atom (e) | 75,028/3000/3000 | 0.0192 |   |   |   | 2.124 | 99.1 |
| 43 | Net charge, per atom (e) | 75,033/3000/3000 | 0.0167 |   |   |   | – | – |
| 44 | Magnetic moment, per atom (μ_B) | 89,231/3000/3000 | 0.0256 |   |   |   | 2.063 | 98.8 |
| 45 | Dielectric tensor (D=9) | 4103/227/229 | 1.690 |   |   |   | 3.401 | 50.3 |
| 46 | Born effective charge (e) | 4472/248/249 | 0.234 |   |   |   | – | – |
| 47 | Piezoelectric tensor, C/m^2 (D=18) | 4513/250/252 | 0.077 |   |   |   | 0.089 | 13.9 |
| 48 | Elastic C_{ij} tensor, GPa (D=36) | 15,936/885/886 | 5.593 |   |   |   | 18.73 | 70.1 |

**(c) Interatomic force fields — `mlearn` per-element energy/force; large sets energy / force**

| # | Task (unit) | N tr/val/te | ALIGNN 2.0 (radius) | ALIGNN 2.0 (kNN) | orig. ALIGNN | CGCNN | Baseline (MAD) | Skill % |
|---|---|---|---|---|---|---|---|---|
| 49 | `mlearn`-Si, energy (meV/atom) | 214/–/25 | 13.88‡ |   |   |   | – |   |
| 50 | `mlearn`-Si, force (eV/Å) | 214/–/25 | 0.0872‡ |   |   |   | – |   |
| 51 | ALIGNN-FF-DB (E/F) | 276,401/–/15,355 | 32.4† / 0.0564† |   |   |   | – |   |
| 52 | MATPES-PBE (E/F) | 391,241/21,736/– | 40.4 / 0.1475 |   |   |   | – |   |
| 53 | FD-FF, 1.1 M (E/F) | 1,097,227/60,957/60,958 | 28.9† / 0.0445† |   |   |   | – |   |
| 54 | MPtrj (E/F) | ~1.5 M | 56.7† / 0.0707† |   |   |   | – |   |
*Blank cells: not run for that graph/model. `–`: baseline unavailable or ill-defined.
† still training. ‡ `mlearn` MAE pending re-verification against a consistent per-atom
energy convention.*
</details>

<a name="notes"></a>
## Useful notes

<details>
<summary><b>Tips & FAQ</b></summary>

**Pure-PyTorch path (no DGL)**

- ALIGNN 2.0 runs fully in native PyTorch — set the model name to a `*_pure` variant
  (e.g. `alignn_atomwise_pure`) and `neighbor_strategy: "pure_torch"`. DGL is optional.
- If you *do* use the legacy DGL path, install a DGL build matching your CUDA runtime;
  mismatched builds are the most common install failure.

**Structure file parsing**

- Simple `.cif`/`.pdb` are handled by `jarvis-tools` directly.
- For complex CIFs: `pip install cif2cell==2.0.0a3`. For complex PDBs:
  `conda install -c ambermd pytraj`.

**Training hyperparameters**

- Example configs ship with a small `batch_size`/`epochs` so tests run fast. **Use
  `batch_size: 32`–`64` and `epochs: 100`–`300` for real trainings** — otherwise
  training is slow and under-performing.
- `pandas >= 1.2.3` required. Since March 2024, `pytorch-ignite` is no longer a dependency.

**CLIs are importable scripts**

- `train_alignn.py`, `pretrained.py`, and `run_alignn_ff.py` install as executables in
  your environment's `bin/` — just run them by name, no absolute path needed.

**Known dataset issues**

- **QM9**: see [issue #54](https://github.com/atomgptlab/alignn/issues/54) for a
  data-split discrepancy affecting reproducibility.

**Getting help**

- GitHub issues: <https://github.com/atomgptlab/alignn/issues> · Email: `drkamal@jhu.edu`
</details>

<a name="refs"></a>
## References

If ALIGNN or ALIGNN-FF contributed to your work, please cite the relevant papers.

<details>
<summary><b>Publication list</b></summary>

**Core**

1. Choudhary, K. & DeCost, B. **Atomistic Line Graph Neural Network for improved materials property predictions.** *npj Computational Materials* 7, 185 (2021). [Link](https://www.nature.com/articles/s41524-021-00650-1)
2. Choudhary, K., DeCost, B., Major, L., Butler, K., Thiyagalingam, J., Tavazza, F. **Unified graph neural network force-field for the periodic table.** *Digital Discovery* (2023). [Link](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b)

**Applications**

3. **Prediction of the Electron Density of States for Crystalline Compounds with ALIGNN.** [Link](https://link.springer.com/article/10.1007/s11837-022-05199-y)
4. **Recent advances and applications of deep learning methods in materials science.** [Link](https://www.nature.com/articles/s41524-022-00734-6)
5. **Designing High-Tc Superconductors with BCS-inspired Screening, DFT, and Deep-learning.** [Link](https://arxiv.org/abs/2205.00060)
6. **A Deep-learning Model for Fast Prediction of Vacancy Formation in Diverse Materials.** [Link](https://arxiv.org/abs/2205.08366)
7. **Graph neural network predictions of MOF CO₂ adsorption properties.** [Link](https://www.sciencedirect.com/science/article/pii/S092702562200163X)
8. **Rapid Prediction of Phonon Structure and Properties using ALIGNN.** [Link](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.023803)
9. **Large Scale Benchmark of Materials Design Methods.** [Link](https://www.nature.com/articles/s41524-024-01259-w)
10. **Prediction of Magnetic Properties in van der Waals Magnets using GNNs.** [Link](https://doi.org/10.1103/PhysRevMaterials.8.114002)
11. **CHIPS-FF: Benchmarking universal force-fields.** [Link](https://github.com/atomgptlab/chipsff)

A complete list is maintained at [jarvis-tools publications](https://jarvis-tools.readthedocs.io/en/master/publications.html).
</details>

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
