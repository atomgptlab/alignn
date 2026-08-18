# Property Predictors

Single-property ALIGNN checkpoints are invoked through
[`pretrained.py`](https://github.com/atomgptlab/alignn/blob/develop/alignn/pretrained.py).

## CLI help

```bash
pretrained.py -h
```

## Example — formation energy (JARVIS-DFT)

```bash
pretrained.py \
  --model_name jv_formation_energy_peratom_alignn \
  --file_format poscar \
  --file_path alignn/examples/sample_data/POSCAR-JVASP-10.vasp
```

## Available models

A non-exhaustive list — names match the `--model_name` flag:

| Target | Model name |
|---|---|
| Formation energy per atom (JARVIS-DFT) | `jv_formation_energy_peratom_alignn` |
| Bandgap — OPT88vdW | `jv_optb88vdw_bandgap_alignn` |
| Bandgap — MBJ | `jv_mbj_bandgap_alignn` |
| Total energy per atom (JARVIS-DFT) | `jv_optb88vdw_total_energy_alignn` |
| Ehull | `jv_ehull_alignn` |
| Bulk modulus (K_v) | `jv_bulk_modulus_kv_alignn` |
| Shear modulus (G_v) | `jv_shear_modulus_gv_alignn` |
| Dielectric (εx, OPT) | `jv_epsx_alignn` |
| Max. piezo dielectric (DFPT) | `jv_dfpt_piezo_max_dielectric_alignn` |
| Spillage | `jv_spillage_alignn` |
| SLME | `jv_slme_alignn` |
| Magnetic moment | `jv_magmom_oszicar_alignn` |
| Raman | `jv_raman_alignn` |
| Superconductor T_c | `jv_supercon_tc_alignn` |
| Interface CBM / VBM | `intermat_cbm`, `intermat_vbm` |
| hMOF CO₂ adsorption | `hmof_co2_absp_alignn` |

See
[`all_models_alignn.json`](https://github.com/atomgptlab/alignn/blob/main/alignn/ff/all_models_alignn.json)
for the full machine-readable list.

## File formats

Pass `--file_format` matching your structure file:

- `poscar` — VASP POSCAR
- `cif`
- `xyz`
- `pdb`

## Using from Python

```python
from ase.build import bulk
from alignn.ff.unified_calculator import (
    AlignnUnifiedCalculator, AlignnUnifiedConfig)

# Any pretrained ALIGNN 2.0 predictor(s): scalar (formation_energy_peratom,
# optb88vdw_bandgap, ...), spectra (edos, pdos, ir, raman) or tensor
# (dielectric_tensor, elastic_tensor, piezo_tensor). prop_graph is "radius"
# (default, force-field-compatible) or "knn".
cfg = AlignnUnifiedConfig(
    energy=False, forces=False, stress=False,   # property-only (no force field)
    properties=["formation_energy_peratom", "optb88vdw_bandgap"],
)
calc = AlignnUnifiedCalculator(cfg)   # models downloaded+cached, loaded once

atoms = bulk("Si", "diamond", a=5.43)
calc.calculate(atoms)                 # single forward pass
print(calc.predictions())
# {'formation_energy_peratom': 0.0026, 'optb88vdw_bandgap': 0.7725}
```

The pure-PyTorch ALIGNN 2.0 models are pulled from
[`pretrained2.py`](https://github.com/atomgptlab/alignn/blob/develop/alignn/pretrained2.py)
(`list_alignn2_models()` shows the full registry). Set `energy=True` to also get
force-field energy/forces/stress from the same calculator.

## See also

- [ALIGNN-FF pretrained models](alignn-ff.md)
- [Performance](../performance.md)
