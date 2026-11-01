# Pre-trained ALIGNN-FF

ALIGNN-FF is a universal graph neural network force-field. Pretrained versions
are hosted on
[Figshare](https://figshare.com/projects/ALIGNN_models/126478) and downloaded
by name on first use.

## CLI

```bash
run_alignn_ff.py -h
```

Common tasks:

```bash
# Single-point energy on the unrelaxed structure
run_alignn_ff.py --file_path POSCAR --task="unrelaxed_energy"

# Structure optimization
run_alignn_ff.py --file_path POSCAR --task="optimize"

# Energy vs. volume curve
run_alignn_ff.py --file_path POSCAR --task="ev_curve"
```

Additional tasks supported by the CLI include phonon calculations, interface
gamma-surface scans, and simple MD — run `-h` to see the full set.

## Available checkpoints

Per-element-embedding models (figshare
[10.6084/m9.figshare.34846446](https://doi.org/10.6084/m9.figshare.34846446)),
test-set errors:

| Name | Training data | Energy (meV/atom) | Force (meV/Å) | Stress (GPa) |
|---|---|---|---|---|
| `alignn_ff_mptrj` (default) | MPtrj | 17.9 | 58.6 | 0.288 |
| `alignn_ff_matpes_pbe` | MatPES-PBE | 26.3 | 101.2 | 0.528 |
| `alignn_ff_matpes_r2scan` | MatPES-r2SCAN | 25.5 | 128.1 | 0.593 |
| `alignn_ff_omat24_1m` | OMat24 (1M subset) | 28.1 | 106.0 | 0.737 |

The r2SCAN model predicts r2SCAN total energies, which are not on the same
scale as the PBE models. `matpes_pbe` and `matpes_r2scan` load the same models;
the previous 2/2/128 versions are `matpes_pbe_old` and `matpes_r2scan_old`.

Earlier releases remain available by name, e.g. `v12.2.2024_dft_3d_307k`,
`v10.30.2024_dft_3d_307k`, `v5.27.2024`, `mps` and `alignnff_wt01`.

The full list is in
[`alignn_ff.yml`](https://github.com/atomgptlab/alignn/blob/main/alignn/registry/alignn_ff.yml).

```python
from alignn.ff.ff import get_figshare_model_ff
from alignn.ff.calculators import AlignnAtomwiseCalculator

calc = AlignnAtomwiseCalculator(
    path=get_figshare_model_ff(model_name="alignn_ff_matpes_pbe")
)
```

Models are cached in `~/.cache/atomgptlab/alignn_ff` (set `ATOMGPTLAB_CACHE` to
move it). From 2026.11.1, caches written by older versions are removed
automatically, with a warning, on the first model download.

## Default path helper

```python
from alignn.ff.ff import default_path
print(default_path())  # directory of the default alignn_ff_mptrj checkpoint
```

## Using it programmatically

See the [ASE calculator](../usage/ase-calculator.md) page for a full worked example
covering relaxation and an EV curve.

## See also

- [Train / fine-tune a force-field](../training/force-field.md)
- [Property predictors](property-predictor.md)
- [Colab: ALIGNN-FF relaxation, EV curve, phonons, interfaces](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ALIGNN_Structure_Relaxation_Phonons_Interface.ipynb)
- [Colab: Melt-quench MD](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Fast_Melt_Quench.ipynb)
