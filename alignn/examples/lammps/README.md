# LAMMPS + ALIGNN-FF (pure-torch) examples

These examples drive LAMMPS with the **native `pair_alignn`** style backed
by the pure-PyTorch ALIGNN-FF model (TorchScripted to `alignn_ff.pt`).
No Python callback is involved — forces/stresses come directly from the
C++ pair style calling libtorch.

## Prerequisites

1. Build LAMMPS with `pair_alignn` linked against libtorch:
   ```bash
   bash ../../scripts/torch/build_lammps_alignn.sh
   ```
2. Download the default ALIGNN-FF `mps` model from figshare and export
   it to TorchScript (`alignn_ff.pt`) into this directory:
   ```bash
   python get_model.py
   ```
   To use a different pretrained model, edit `default_path()` in
   `get_model.py` (e.g. `model_name="v12.2.2024_dft_3d_307k"`).

## Examples

| File | Purpose |
| --- | --- |
| `nve_si_stability.in` + `build_si.py` | NVE stability test on Si (216 atoms). Run this first to confirm the model conserves energy. |
| `melt_quench_si.in` + `build_si.py` | Melt-quench Si: 300 K → 3500 K → quench → 300 K. Produces amorphous Si. |
| `melt_quench_sio2.in` + `build_sio2.py` | Melt-quench α-quartz SiO₂: 300 K → 4000 K → quench → 300 K. Produces amorphous silica. |

## How to run

```bash
# 1. Build the initial structure
python build_si.py            # writes si.data

# 2. NVE stability check (look for flat etotal in the log)
lmp -in nve_si_stability.in

# 3. Melt-quench
lmp -in melt_quench_si.in
```

For SiO₂:
```bash
python build_sio2.py          # writes sio2.data
lmp -in melt_quench_sio2.in
```

## NVE stability — what to look for

Open `log.lammps` and check `etotal`. Over a 5 ps NVE run the total
energy should drift by less than a few meV/atom. Large drift (>10 meV/atom)
indicates an undertrained model, too-aggressive timestep, or a cutoff
mismatch between the model and the LAMMPS neighbor list.

The script also dumps `nve_si.lammpstrj`; visualize with OVITO.
