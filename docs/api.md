# Package Overview

High-level tour of the `alignn` package. For authoritative details, read the source —
this page is a map, not an API reference.

## Top-level modules

| Module | Role |
|---|---|
| `alignn.train_alignn` | Training CLI entry point |
| `alignn.train` | Training loop (called by the CLI) |
| `alignn.config` | Pydantic config schema for training / models |
| `alignn.data` | Dataset loaders — CSV/JSON index → torch datasets |
| `alignn.dataset` | Low-level dataset classes |
| `alignn.graphs` | Crystal graph + line graph construction |
| `alignn.lmdb_dataset` | LMDB-backed dataset for large-scale training |
| `alignn.pretrained` | Load and apply pretrained property predictors |
| `alignn.run_alignn_ff` | ALIGNN-FF CLI entry point |
| `alignn.cli` | Shared CLI argument parsing |
| `alignn.utils` | Misc helpers (logging, config loading, …) |
| `alignn.profiler` | Optional training profiler |

## Models (`alignn.models`)

| Module | Model |
|---|---|
| `alignn.models.alignn` | Original ALIGNN property predictor |
| `alignn.models.alignn_atomwise` | ALIGNN with atomwise outputs (forces/charges/mag) |
| `alignn.models.ealignn_atomwise` | Equivariant atomwise variant |
| `alignn.models.utils` | Shared layers and helpers |

## Force-field (`alignn.ff`)

| Module | Role |
|---|---|
| `alignn.ff.ff` | `AlignnAtomwiseCalculator`, `default_path`, training utils |
| `alignn.ff.calculators` | ASE calculator implementations |
| `alignn/registry/alignn_property.yml` | Registry of property-predictor checkpoints |
| `alignn/registry/alignn_ff.yml` | Registry of ALIGNN-FF checkpoints |
| `alignn/registry/alignn_inverse.yml` | Registry of generative inverse-design checkpoints |

Pretrained checkpoints are not bundled: they are downloaded by name on first
use and cached in `~/.cache/atomgptlab/` (or `~/.alignn2_models/` for
`alignn.pretrained`).

## Examples & scripts

- [`alignn/examples/`](https://github.com/atomgptlab/alignn/tree/main/alignn/examples) —
  runnable sample datasets and configs (`sample_data`, `sample_data_ff`,
  `sample_data_multi_prop`, …).
- [`alignn/scripts/`](https://github.com/atomgptlab/alignn/tree/main/alignn/scripts) —
  high-throughput training scripts that download public datasets and train one model
  per target.

## CLIs at a glance

```bash
train_alignn.py   -h   # training (regression, classification, atomwise, multi-output)
pretrained.py     -h   # apply a pretrained property predictor
run_alignn_ff.py  -h   # apply a pretrained ALIGNN-FF (optimize, EV curve, phonons, …)
```

All three are installed to your environment's `bin/` directory.
