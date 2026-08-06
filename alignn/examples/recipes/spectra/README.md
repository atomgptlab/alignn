# ALIGNN example: Spectra / multi-output curve

Predict a full **spectral curve** on a fixed grid — electronic DOS (300 bins), phonon DOS (200 bins), or Raman spectrum (200 bins). The target is a length-`D` list (one value per bin).

## Run it (CPU, ~1-2 min)

```bash
# 1) generate a tiny synthetic dataset -> id_prop.json
python make_toy_dataset.py

# 2) train (10 epochs on the toy data)
train_alignn.py --root_dir . --config_name config_example.json --output_dir toy_out --target_key target --id_key jid
```

You should see per-epoch `Train Loss` / `Val Loss` lines and a final `Test MAE`,
with results written to `toy_out/` (`history_val.json`, predictions, checkpoint).

## Key config knobs (`config_example.json`)

- `model.output_features: 200` — **number of bins** (200 Raman/pDOS, 300 eDOS); match `D` in `make_toy_dataset.py`.
- `criterion: l1` — averaged over bins.

> Real DOS/Raman curves are smooth; the toy script uses a Gaussian bump as a stand-in.

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
[{"jid": "toy-0", "atoms": {...}, "target": ...}]
```
