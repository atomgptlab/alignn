# ALIGNN example: Force field (energy + forces + stress)

Train an **ALIGNN-FF** interatomic potential that outputs energy, analytic forces (gradients of the energy), and stress — usable for relaxation and molecular dynamics (including LAMMPS via `pair_alignn`). Uses the radius graph (`cutoff: 5.0`).

## Run it (CPU, ~1-2 min)

```bash
# 1) generate a tiny synthetic dataset -> id_prop.json
python make_toy_dataset.py

# 2) train (10 epochs on the toy data)
train_alignn.py --root_dir . --config_name config_example.json --output_dir toy_out --target_key energy_per_atom --force_key forces --id_key jid
```

You should see per-epoch `Train Loss` / `Val Loss` lines and a final `Test MAE`,
with results written to `toy_out/` (`history_val.json`, predictions, checkpoint).

## Key config knobs (`config_example.json`)

- `model.calculate_gradient: true` — forces are the true gradient of the energy (energy-conserving).
- `graphwise_weight` (energy), `gradwise_weight` (forces), `stresswise_weight` (stress) — the loss mixture.
- Pass `--force_key forces`; stresses are read automatically if present.

> **Energy must be per atom** (`energy_per_atom`) in `id_prop.json`, not per structure. Forces are `Natoms x 3`, stresses Voigt-6.

## ⚠️ This is a smoke test, not a real model

The synthetic dataset (40 rattled Si cells with fake targets) exists only to prove
the pipeline runs. For a usable model:

- Replace the toy structures/labels in `make_toy_dataset.py` with **real DFT data**
  (thousands to millions of entries).
- Raise `epochs` to **100-300** and `batch_size` to **32-64** in `config_example.json`.
- Expect training to take much longer and to need a GPU for large datasets.

## Dataset format (`id_prop.json`)

A JSON list; each entry has a `jid`, an inline jarvis `Atoms` dict, and the target(s):

```json
[{"jid": "toy-0", "atoms": {...}, "energy_per_atom": ...}]
```
