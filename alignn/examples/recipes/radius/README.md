# ALIGNN example: Radius graph — scalar property

Same scalar-property task as the kNN recipe, but with the **radius graph** (`cutoff: 5.0`). The radius neighbour list is continuous under atomic displacement, so it is the construction used for molecular dynamics / force fields. For static property prediction kNN is usually a little more accurate; use radius when you need MD-consistency.

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

- `cutoff: 5.0` — fixed radius cutoff (vs 8.0 for kNN).
- `model.output_features: 1`, `graphwise_weight: 1.0`.



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
