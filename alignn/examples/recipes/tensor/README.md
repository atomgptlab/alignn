# ALIGNN example: Tensor property (D-dimensional)

Predict a fixed-length response **tensor** per structure — e.g. the 9-component dielectric tensor, 18-component piezoelectric tensor, or 36-component elastic $C_{ij}$. The target in `id_prop.json` is a length-`D` list.

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

- `model.output_features: 9` — **set this to your tensor dimension** (9 dielectric, 18 piezo, 36 elastic) and match `D` in `make_toy_dataset.py`.
- `criterion: l1` — mean-absolute-error over components.

> The toy script emits a length-9 vector; change `D` in both `make_toy_dataset.py` and `output_features` together.

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
