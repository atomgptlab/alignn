# ALIGNN example: Atomwise (per-atom) property

Predict a **per-atom** scalar — atomic charges (Bader), site magnetic moments, etc. Each atom gets its own prediction, so the target in `id_prop.json` is a list of length `Natoms` under the `charges` key.

## Run it (CPU, ~1-2 min)

```bash
# 1) generate a tiny synthetic dataset -> id_prop.json
python make_toy_dataset.py

# 2) train (10 epochs on the toy data)
train_alignn.py --root_dir . --config_name config_example.json --output_dir toy_out --target_key target --id_key jid --atomwise_key charges
```

You should see per-epoch `Train Loss` / `Val Loss` lines and a final `Test MAE`,
with results written to `toy_out/` (`history_val.json`, predictions, checkpoint).

## Key config knobs (`config_example.json`)

- `model.atomwise_output_features: 1`, `atomwise_weight: 1.0` — turn on the per-atom head.
- `model.graphwise_weight: 0.0` — the graph-level target is unused.
- Pass `--atomwise_key charges` (rename to your per-atom key).

> `target` (graph-level) is kept as a dummy 0.0 because `graphwise_weight` is 0.

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
