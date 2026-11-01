# Pre-trained Models

ALIGNN ships a large catalog of pre-trained checkpoints for direct use.

- Models are hosted on [Figshare](https://figshare.com/projects/ALIGNN_models/126478).
- Two CLIs load them: `pretrained.py` (single-property predictors) and
  `run_alignn_ff.py` (universal force-field).
- Model names and sources are listed in
  [`alignn/registry/`](https://github.com/atomgptlab/alignn/blob/main/alignn/registry):
  `alignn_ff.yml` (force fields), `alignn_property.yml` (property predictors)
  and `alignn_inverse.yml` (generative inverse design).

| Page | What it covers |
|---|---|
| [Property predictors](property-predictor.md) | Per-property checkpoints (formation energy, bandgap, …) |
| [ALIGNN-FF](alignn-ff.md) | Universal force-field: EV curve, optimization, phonons, MD |
| [ASE calculator](../usage/ase-calculator.md) | Drive ALIGNN-FF from Python via ASE |

See [Performance](../performance.md) for per-dataset accuracy numbers.
