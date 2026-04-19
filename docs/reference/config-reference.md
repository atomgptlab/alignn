# Config Reference

Every training run in ALIGNN is controlled by a single JSON file. The schema is
enforced by two pydantic classes: `TrainingConfig` (top-level) in
[`alignn/config.py`](https://github.com/atomgptlab/alignn/blob/main/alignn/config.py)
and a per-model config (e.g. `ALIGNNAtomWiseConfig`) in
[`alignn/models/alignn_atomwise.py`](https://github.com/atomgptlab/alignn/blob/main/alignn/models/alignn_atomwise.py).

At runtime the config is validated and any unknown field is rejected, so this
file is the authoritative list of what's available.

---

## Top-level (`TrainingConfig`)

### Dataset & target

| Field | Type / default | Meaning |
|---|---|---|
| `dataset` | one of `dft_3d`, `megnet`, `qm9`, `user_data`, … (see enum) / `"dft_3d"` | Which dataset. Use `"user_data"` for your own `POSCAR`s. |
| `target` | string / `"exfoliation_energy"` | Column in `id_prop.csv` to predict. |
| `id_tag` | `"jid"` \| `"id"` \| `"_oqmd_entry_id"` / `"jid"` | Which column holds the ID. |
| `classification_threshold` | `float?` / `None` | If set, regression targets above it become class 1; enables classification mode. |
| `target_multiplication_factor` | `float?` / `None` | Scale all targets by this constant (useful when units don't match loss scale). |

### Atoms → graph

| Field | Default | Meaning |
|---|---|---|
| `atom_features` | `"cgcnn"` | Node feature scheme. `cgcnn` = 92-d one-hot-ish table, `atomic_number` = scalar Z, `basic` = 11 hand-picked features, `cfid` = 438 chemical descriptors. Must match `model.atom_input_features`. |
| `neighbor_strategy` | `"k-nearest"` | How to build the neighbor graph. See [Neighbor strategies](#neighbor-strategies). |
| `cutoff` | `8.0` | Radius cutoff (Å) for pair edges. |
| `cutoff_extra` | `3.0` | Fallback slack for `radius_graph` if too few neighbors are found. |
| `three_body_cutoff` | `None` | Separate cutoff (Å) for line-graph / angle triplets. Only honored by `neighbor_strategy="pure_torch"`. `None` → use `cutoff`. Must be ≤ `cutoff`. |
| `max_neighbors` | `12` | Per-source cap. |
| `use_canonize` | `True` | Canonicalize k-nearest edge order (reproducibility). |
| `compute_line_graph` | `True` | Build the bond-angle line graph. |

#### Neighbor strategies

| Value | Backend | Autograd through `r`? | Notes |
|---|---|---|---|
| `k-nearest` (default) | Jarvis | no | Fastest for small property-prediction cells. |
| `radius_graph` | pure torch | no | Honest cutoff; slower. |
| `radius_graph_jarvis` | Jarvis | no | Uses Jarvis's `NeighborsAnalysis`. |
| `fast_graph` | matscipy (C) | no | Same topology as `radius_graph` but ~200× faster on large cells. Requires `pip install matscipy`. |
| `torch_graph` | matscipy + torch | **yes** | Recomputes `r = positions[dst] − positions[src] + shift @ lattice` in torch → gradients flow to positions and lattice. |
| `pure_torch` | matscipy + torch | **yes** | Like `torch_graph`, plus respects `three_body_cutoff` for the line graph. |

### Splitting

| Field | Default | Meaning |
|---|---|---|
| `train_ratio` / `val_ratio` / `test_ratio` | `0.8 / 0.1 / 0.1` | Fractions (used when counts are not set). |
| `n_train` / `n_val` / `n_test` | `None` | Absolute counts; override the ratios when set. |
| `keep_data_order` | `True` | If `True`, no shuffling before the split — makes comparisons across runs reproducible. |
| `random_seed` | `123` | Seed for Python / numpy / torch / CUDA RNGs. |

### Training loop

| Field | Default | Meaning |
|---|---|---|
| `epochs` | `300` | Number of training passes. |
| `batch_size` | `64` | Per-step batch size. |
| `learning_rate` | `1e-2` | Base LR. |
| `weight_decay` | `0` | AdamW / SGD weight decay. |
| `warmup_steps` | `2000` | OneCycle warmup steps (ignored if `scheduler="none"`). |
| `criterion` | `"mse"` | Regression loss. `"l1"` is what the force-field examples use. `"zig"` / `"poisson"` are specialized. |
| `optimizer` | `"adamw"` | `"adamw"` or `"sgd"`. |
| `scheduler` | `"onecycle"` | `"onecycle"` or `"none"`. |
| `n_early_stopping` | `None` | Stop if val loss hasn't improved in this many epochs (~50 is common). |

### Graph caching

| Field | Default | Meaning |
|---|---|---|
| `use_lmdb` | `True` | Cache built graphs in LMDB (fast reload, memory-bounded). |
| `read_existing` | `False` | If `True`, reuse an existing LMDB cache on disk. **Must match the current model backend** (DGL vs pure-torch). Default `False` rebuilds from scratch every run to avoid footguns. |

### Runtime & distributed

| Field | Default | Meaning |
|---|---|---|
| `dtype` | `"float32"` | `"float32"` or `"float64"`. |
| `num_workers` | `4` | `DataLoader` workers. `0` for debugging. |
| `pin_memory` | `False` | Pinned host memory for faster H2D copies on GPU. |
| `distributed` | `False` | Enable DDP multi-process training (paired with `torchrun`). |
| `data_parallel` | `False` | Legacy single-process multi-GPU. Prefer `distributed`. |

### Logging & outputs

| Field | Default | Meaning |
|---|---|---|
| `output_dir` | `.` | Where checkpoints / logs / predictions are written. |
| `filename` | `"sample"` | Prefix for cache dirs (`<prefix>train_data`, etc.). |
| `write_checkpoint` | `True` | Save `best_model.pt` / `current_model.pt`. |
| `write_predictions` | `True` | Dump per-sample predictions at the end. |
| `store_outputs` | `True` | Keep intermediate outputs in the history JSONs. |
| `progress` | `True` | `tqdm` progress bars. |
| `log_tensorboard` | `False` | Also write a TensorBoard run. |
| `save_dataloader` | `False` | Pickle the built DataLoader objects (large; rarely useful). |
| `standard_scalar_and_pca` | `False` | Z-score + PCA the targets (legacy). |
| `normalize_graph_level_loss` | `False` | Divide the graph-level loss by number of atoms (enable for extensive quantities mixed with intensive ones). |

### `model`

A nested object whose schema is chosen by `model.name`. Currently available:

| `name` | Class | Purpose |
|---|---|---|
| `alignn` | `ALIGNNConfig` | Original ALIGNN (scalar regression only). |
| `alignn_atomwise` | `ALIGNNAtomWiseConfig` | Adds atomwise / gradwise / stresswise outputs — this is what `force-field.md` uses. |
| `alignn_atomwise_pure` | `ALIGNNAtomWisePureConfig` | **DGL-free**, scatter-based reimplementation of `alignn_atomwise`. Accepts the same config fields; TorchScript-exportable for LAMMPS. See [Pure-torch notes](#pure-torch-notes). |
| `ealignn_atomwise` | `eALIGNNAtomWiseConfig` | Equivariant variant. |

See [Model config (`alignn_atomwise`)](#model-config-alignn_atomwise) below.

---

## Model config (`alignn_atomwise`)

All fields below apply to both `alignn_atomwise` and `alignn_atomwise_pure`
unless noted.

### Architecture

| Field | Default | Meaning |
|---|---|---|
| `name` | required | `"alignn_atomwise"` or `"alignn_atomwise_pure"`. |
| `alignn_layers` | `2` | Number of ALIGNN (node + edge + triplet) conv blocks. |
| `gcn_layers` | `2` | Number of subsequent node + edge conv blocks (no triplets). |
| `atom_input_features` | `1` | Width of incoming atom features. **Must match** `atom_features` (92 for `cgcnn`, 1 for `atomic_number`, 11 for `basic`, 438 for `cfid`). |
| `edge_input_features` | `80` | Number of RBF bins for bond lengths. |
| `triplet_input_features` | `40` | Number of RBF bins for bond-angle cosines. |
| `embedding_features` | `64` | Width of the two-layer MLP inside the edge/angle embeddings. |
| `hidden_features` | `64` | Core channel width running through all conv blocks. |
| `output_features` | `1` | Graph-level output dimension (>1 for multi-output regression). |

### Output heads and loss weights

| Field | Default | Meaning |
|---|---|---|
| `calculate_gradient` | `True` | Compute per-atom forces via `∂E/∂r`. |
| `atomwise_output_features` | `0` | Per-atom prediction dim (e.g. charges, magmoms). `0` disables. |
| `additional_output_features` | `0` | Extra per-graph head dim. `0` disables. |
| `graphwise_weight` | `1.0` | Loss weight on the graph-level target. |
| `gradwise_weight` | `1.0` | Loss weight on forces. Setting to `0` automatically disables `calculate_gradient`. |
| `stresswise_weight` | `0.0` | Loss weight on the 3×3 stress tensor. |
| `atomwise_weight` | `0.0` | Loss weight on per-atom outputs (requires `atomwise_output_features > 0`). |
| `additional_output_weight` | `0.0` | Loss weight on the additional head. |

### Force-field specifics

| Field | Default | Meaning |
|---|---|---|
| `grad_multiplier` | `-1` | Multiplier applied to `∂E/∂r` when converting to forces. Physics convention `F = −∂E/∂x` ⇒ `-1`. |
| `add_reverse_forces` | `True` | Sum `∂E/∂r` over both `(i→j)` **and** `(j→i)` edges (proper Newton's third law on directed graphs). |
| `lg_on_fly` | `True` | Recompute bond-angle cosines inside the autograd graph every forward (needed for exact force/stress derivatives). Leave `True` unless profiling. |
| `batch_stress` | `True` | Compute stress per graph in the batch (vs. one stress per whole batch). |
| `force_mult_natoms` | `False` | Scale forces by number of atoms (rare; use for unit conventions). |
| `energy_mult_natoms` | `True` | Treat the graph-head output as energy *per atom* and multiply by `natoms` before taking the gradient. Standard for ALIGNN-FF. |
| `stress_multiplier` | `1.0` | Final scalar on the stress output (for unit fixes). |
| `include_pos_deriv` | `False` | Alternative force path: differentiate through Cartesian positions directly (rather than through edge displacements). Currently experimental; the `_pure` variant accepts the field but ignores it. |

### Cutoff shaping

| Field | Default | Meaning |
|---|---|---|
| `use_cutoff_function` | `False` | Apply a smooth polynomial envelope to distances before RBF. Useful when `cutoff` is tight. |
| `multiply_cutoff` | `False` | If `use_cutoff_function=True`, multiply the edge embedding by the envelope (instead of the bondlength). |
| `inner_cutoff` | `3.0` | Envelope radius (Å); envelope = 1 below this, smoothly → 0 up to `cutoff`. |
| `exponent` | `5` | Exponent of the smooth-cutoff polynomial. |

### Short-distance penalty

| Field | Default | Meaning |
|---|---|---|
| `use_penalty` | `True` | Add a repulsive penalty when any bondlength falls below `penalty_threshold`. Prevents the model from hallucinating attractive wells at unphysical distances — important for MD stability. |
| `penalty_factor` | `0.1` | Strength. |
| `penalty_threshold` | `1.0` | Distance (Å) below which the penalty kicks in. |

### Miscellaneous

| Field | Default | Meaning |
|---|---|---|
| `link` | `"identity"` | Output activation. `"log"` applies `exp()` (positive outputs), `"logit"` applies `sigmoid()`. |
| `classification` | `False` | Two-class NLL head (set automatically when `classification_threshold` is non-null at the training level). |
| `zero_inflated` | `False` | Zero-inflated Gaussian output (specialized). |
| `extra_features` | `0` | If > 0, concatenate this many pre-computed fingerprint features per atom (e.g. Gong et al.). |

---

## Pure-torch notes

`model.name = "alignn_atomwise_pure"` swaps the DGL message-passing core for
`scatter_add` / `scatter_mean` primitives. The **forward outputs are numerically
equivalent** to `alignn_atomwise` — parity sweep across all doc examples shows
`|Δ| ≤ 0.04` on all losses after 3 epochs (force-field scenario is bit-identical,
see `alignn/scripts/parity_dgl_vs_pure.py`). Residual drift is `scatter_add`
nondeterminism on CUDA, not a modelling difference.

Not carried over from `alignn_atomwise` (config fields accepted but ignored):

- `include_pos_deriv`
- Some `extra_features` code paths — experimental in the DGL model too.

When the pure model is selected, the training pipeline automatically:

- Uses `alignn/pure_lmdb_dataset.py` (pickles `TorchGraph` objects, no DGL).
- Uses `PureTorchLMDBDataset.collate_line_graph` for batching.
- Leaves dataloader / loss / metrics / checkpoints untouched.

For LAMMPS integration, call
[`alignn/scripts/export_torchscript.py`](https://github.com/atomgptlab/alignn/blob/main/alignn/scripts/export_torchscript.py) — it scripts the
`forward_tensors_z(positions, lattice, atomic_numbers, src, dst, shift,
compute_stress)` entry point and bakes the atomic-number → feature lookup into
the `.pt` so the C++ host only needs atomic numbers.

---

## Minimal examples

### Single-output regression

```json
{
  "dataset": "user_data",
  "target": "formation_energy_peratom",
  "atom_features": "cgcnn",
  "neighbor_strategy": "k-nearest",
  "cutoff": 8.0,
  "max_neighbors": 12,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 1e-3,
  "criterion": "mse",
  "model": {
    "name": "alignn_atomwise",
    "atom_input_features": 92,
    "alignn_layers": 4,
    "gcn_layers": 4,
    "hidden_features": 256,
    "output_features": 1,
    "calculate_gradient": false,
    "graphwise_weight": 1.0,
    "gradwise_weight": 0.0,
    "stresswise_weight": 0.0,
    "use_penalty": false
  }
}
```

### ALIGNN-FF (energy + forces + stress)

```json
{
  "dataset": "user_data",
  "target": "target",
  "atom_features": "cgcnn",
  "neighbor_strategy": "fast_graph",
  "cutoff": 6.0,
  "max_neighbors": 12,
  "epochs": 200,
  "batch_size": 2,
  "learning_rate": 1e-3,
  "criterion": "l1",
  "model": {
    "name": "alignn_atomwise",
    "atom_input_features": 92,
    "alignn_layers": 3,
    "gcn_layers": 3,
    "hidden_features": 128,
    "output_features": 1,
    "calculate_gradient": true,
    "graphwise_weight": 0.85,
    "gradwise_weight": 0.05,
    "stresswise_weight": 0.05,
    "add_reverse_forces": true,
    "lg_on_fly": true,
    "use_penalty": true
  }
}
```

### Pure-torch FF with separate 3-body cutoff

```json
{
  "neighbor_strategy": "pure_torch",
  "cutoff": 6.0,
  "three_body_cutoff": 4.0,
  "max_neighbors": 12,
  "model": { "name": "alignn_atomwise_pure", "…": "…" }
}
```

### Classification

```json
{
  "classification_threshold": 0.01,
  "target": "is_metal",
  "criterion": "mse",
  "model": {
    "name": "alignn_atomwise",
    "classification": true,
    "graphwise_weight": 1.0,
    "gradwise_weight": 0.0,
    "calculate_gradient": false
  }
}
```

---

## Common gotchas

- **`atom_input_features` vs `atom_features`.** These must match (cgcnn → 92,
  atomic_number → 1, basic → 11, cfid → 438). A mismatch raises a shape error
  at the first forward pass, not at config parse.
- **`read_existing`.** Keep `false` unless you're sure the cache matches the
  current model backend + graph settings. Backend-mismatch (DGL vs pure) is
  caught and raises a clear error; parameter drift (different cutoff etc.) is
  not caught.
- **`gradwise_weight=0`** silently disables `calculate_gradient` in the model
  code. If you think forces should be training but see `Grad=0.0000`, check
  both flags.
- **`use_penalty=true` with a loose cutoff** can inflate training loss when
  your data legitimately has short bonds (e.g. H-containing systems). Lower
  `penalty_threshold` or set `use_penalty=false` in that case.
- **`batch_stress=false` + large batch** gives one stress for the whole
  batch, not per-graph — almost never what you want.
- **`lg_on_fly=false`** caches triplet angle cosines; saves time but forces
  computed by autograd will be missing the angle-contribution unless you
  recompute them. Leave `true` for force-field training.
