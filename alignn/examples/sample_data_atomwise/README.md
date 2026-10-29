# Per-atom (atomwise) training example

Trains the per-atom head (`atomwise_output_features: 1`,
`atomwise_weight: 1.0`, graph-level loss disabled) on per-atom labels —
e.g. Bader charges or site-projected magnetic moments.

**The labels in `id_prop.json` here are synthetic** (format illustration and
smoke tests only). Build a real dataset with
`alignn/scripts/make_atomwise_dataset.py`.

Schema per entry:

```json
{"jid": "...", "atoms": {...}, "target": <cell-level scalar, unused>,
 "charges": [q1, ..., qN], "magmoms": [m1, ..., mN]}
```

Train (the two configs are identical; the per-atom target is selected by
`--atomwise_key`):

```bash
python -m alignn.train_alignn --root_dir . --config_name config_magmom.json \
    --target_key target --atomwise_key magmoms --output_dir out_magmom
python -m alignn.train_alignn --root_dir . --config_name config_charge.json \
    --target_key target --atomwise_key charges --output_dir out_charge
```

Evaluate per-atom MAE from the run's `Test_results.json`:

```bash
python alignn/scripts/eval_atomwise.py --run_dir out_magmom
```
