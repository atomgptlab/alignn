# LAMMPS native pair style (`pair_alignn`)

`pair_alignn` is a C++ LAMMPS pair style that runs a TorchScript-exported
ALIGNN-FF model as the force engine. It replaces the Python+ASE bridge
(`alignn.ff.lammps_bridge`) for production runs:

| Path | Overhead per step | When to use |
|---|---|---|
| ASE calculator (Python) | high (~40 ms for 200 atoms) | one-off scripts, debugging |
| `lammps_bridge_torch.py` | medium (~15 ms) | quick Python-hosted MD |
| **`pair_alignn` (this page)** | **low (~3–10 ms)** | production MD, large systems |

Everything stays on the GPU; Python is never invoked inside the MD step
loop.

!!! note "Status"
    MVP single-rank implementation. Matches the TorchScript Pure model
    forward to ~10⁻⁶ eV on Si. Multi-rank / domain decomposition is not
    yet implemented — cross-rank ghost contributions are skipped.

## What you need

1. A trained ALIGNN-FF model directory containing `best_model.pt` + `config.json`
2. LAMMPS built against **libtorch with the same CUDA/ABI flags** as your
   Python PyTorch install
3. `pair_alignn.cpp` / `pair_alignn.h` compiled into that LAMMPS binary
4. A TorchScript `.pt` of your model exported via `export_torchscript.py`

## Building LAMMPS with `pair_alignn`

The repo ships a self-contained build script
(`alignn/scripts/torch/build_lammps_alignn.sh`) that:

1. Clones LAMMPS `stable_29Aug2024_update1`
2. Drops `pair_alignn.cpp/.h` into `src/`
3. Appends a libtorch `find_package` to `cmake/CMakeLists.txt`
4. Configures + builds + installs the Python module

Prerequisites in your conda env (one-time):

```bash
# Matching CUDA toolkit for libtorch (check your PyTorch first)
python -c "import torch; print(torch.version.cuda)"
# If 12.8, install:
mamba install -y -c nvidia -c conda-forge cuda-toolkit=12.8 mkl-devel mkl-include
```

Then run the build script:

```bash
bash alignn/scripts/torch/build_lammps_alignn.sh
```

On success, you'll see:
```
pair_alignn available: True
```

The built `lmp` executable lands at `~/lammps-alignn/build/lmp`, and the
conda `lammps` Python module is overwritten with the libtorch-linked
version.

### Common build issues

| Symptom | Cause | Fix |
|---|---|---|
| `PyTorch: CUDA cannot be found` | CUDA toolkit version mismatches libtorch's build version | install matching `cuda-toolkit=X.Y` via conda |
| `Imported target "torch" includes non-existent path ".../MKL_INCLUDE_DIR-NOTFOUND"` | MKL not installed | `mamba install mkl-devel mkl-include` |
| `fatal error: cuda.h: No such file or directory` | CUDA headers not on include path | cmake flag: `-DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include"` |
| `Enabled packages: <None>` | cmake didn't wire pair_alignn in | verify `append find_package(Torch)` block lives at the end of `cmake/CMakeLists.txt` |
| Builds but `pair_alignn` not listed | `pair_alignn.cpp` not in `src/` | `cp pair_alignn.{cpp,h} ~/lammps-alignn/src/` |

## Exporting the TorchScript model

```bash
python alignn/scripts/torch/export_torchscript.py \
    --model-dir MELT/OutputDir \
    --out alignn_ff.pt \
    --atom-features atomic_number \
    --dtype float32
```

Output:
```
smoke test OK: {'energy': torch.Size([]), 'forces': torch.Size([2, 3]), 'stress': torch.Size([3, 3])}
wrote alignn_ff.pt  (143,297 params)
```

**`--atom-features` must match what was used at training time** — if the
model was trained with CGCNN features (92-dim), pass `--atom-features
cgcnn`. The bundled species table is baked into the `.pt` so the C++ side
never needs `jarvis`.

## Input script syntax

```
pair_style  alignn <cutoff_Å> [max_neighbors]
pair_coeff  * *  <model.pt>  <Sym_for_type_1>  [<Sym_for_type_2> ...]
```

| Argument | Meaning | Example |
|---|---|---|
| `<cutoff_Å>` | Neighbor cutoff in Å. **Must match training** | `5.0` |
| `[max_neighbors]` | k-nearest cap per atom. Default 12 | `12` |
| `<model.pt>` | Path to the TorchScript file | `alignn_ff.pt` |
| `<Sym_for_type_N>` | Element symbol for LAMMPS atom type N | `Si` |

Example for Si trained with cutoff 5.0 Å, 12 neighbors:
```
pair_style  alignn 5.0 12
pair_coeff  * *   alignn_ff.pt  Si
```

Look up the correct values from your trained model:
```bash
python -c "
import json
c = json.load(open('OutputDir/config.json'))
print('cutoff:', c['cutoff'], 'Å')
print('max_neighbors:', c['max_neighbors'])
print('atom_features:', c.get('atom_features'))
"
```

## A complete melt-quench input

```
units           metal                  # eV, Å, ps, bar, K
atom_style      atomic
boundary        p p p
read_data       si.data
mass            1 28.0855

pair_style      alignn 5.0 12
pair_coeff      * * alignn_ff.pt Si

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        0.001                 # 1 fs
thermo          50
thermo_style    custom step temp pe ke etotal press vol
dump            traj all custom 100 si.lammpstrj id type x y z

# ── Stage 1: equilibrate at 300 K ──
velocity        all create 300.0 12345 mom yes rot yes
fix             nvt1 all nvt temp 300.0 300.0 0.1
run             200
unfix           nvt1

# ── Stage 2: heat to 3500 K over 1 ps ──
fix             nvt2 all nvt temp 300.0 3500.0 0.1
run             1000
unfix           nvt2

# ── Stage 3: hold molten ──
fix             nvt3 all nvt temp 3500.0 3500.0 0.1
run             2000
unfix           nvt3

# ── Stage 4: quench to 300 K over 4 ps ──
fix             nvt4 all nvt temp 3500.0 300.0 0.1
run             4000

write_data      si_quenched.data
```

Making the Si data file:
```python
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from ase.io import write
a = Atoms.from_dict(get_jid_data(jid='JVASP-1002', dataset='dft_3d')['atoms'])
ase = a.make_supercell([3, 3, 3]).ase_converter()    # 216 atoms
write("si.data", ase, format="lammps-data", specorder=["Si"])
```

## Running

Three equivalent ways:

=== "Custom LAMMPS binary"

    ```bash
    ~/lammps-alignn/build/lmp -in si_native.in
    ```

=== "Python-driven"

    ```python
    from lammps import lammps
    l = lammps()
    l.file("si_native.in")
    ```

=== "In a Jupyter cell"

    ```python
    !~/lammps-alignn/build/lmp -in si_native.in
    ```

## Validating against the Python calculator

Before trusting a production run, compare LAMMPS forces against the
Python reference:

```bash
python alignn/scripts/torch/validate_pair_alignn.py \
    --model-dir MELT/OutputDir \
    --ts-model alignn_ff.pt \
    --jid JVASP-1002 \
    --supercell 2 2 2
```

Expected output shape:
```
E (Python):     -346.003876 eV
E (LAMMPS):     -345.xxxxxx eV
|ΔE|:            < 1e-4 eV       [PASS]
max|ΔF|:         < 1e-3 eV/Å     [PASS]
```

If energies agree to ~1e-6 but forces are off, you have a gradient /
autograd issue. If everything is off by a consistent factor, it's a unit
or normalization mismatch. If the C++ energy is off by an O(1) factor vs
a single-precision residual, check cutoff and atom_features.

## Required LAMMPS settings

- `newton_pair on` — LAMMPS default; the pair style requires it for force
  reduction on ghost atoms
- atom IDs must be enabled (default)
- `atom->map()` must be built — pair_alignn auto-initializes it

## What pair_alignn does per step

1. Copy local atom positions + types to a GPU tensor (`(N_local, 3)`)
2. Walk LAMMPS's full+ghost neighbor list. For each local `i`, collect all
   neighbors within cutoff; for ghosts, look up the owner via
   `atom->map(tag)` and compute the PBC shift vector
3. Keep only the `max_neighbors` closest per atom (matches jarvis's
   `neighbor_strategy="k-nearest"` / `"pure_torch"` behavior)
4. Pack `(src, dst, shift)` tensors and call the TorchScript model's
   `forward_tensors_z(positions, lattice, atomic_numbers, src, dst, shift,
   compute_stress)` entry point
5. Read back `energy` (scalar), `forces` (N×3), and `stress` (3×3)
6. Scatter forces to `f[i]` for local atoms; accumulate energy and virial
7. LAMMPS's reverse comm handles any ghost-force reductions (no-op on a
   single rank)

## Model-side requirements

Your trained model must:

1. Use `name: "alignn_atomwise_pure"` in its config (so
   `forward_tensors_z` exists). If you trained with `alignn_atomwise`
   (DGL), retrain or convert weights — the two variants are currently
   NOT numerically equivalent on this trained checkpoint (known issue).
2. Be TorchScript-scriptable — the pure variant is designed for this
   (`@torch.jit.ignore` on DGL-specific methods, `@torch.jit.export` on
   tensor-only forwards)
3. Have `calculate_gradient: True` — forces come from autograd

## Known limitations

- **Single MPI rank only.** Cross-rank ghost atoms get skipped silently.
  For multi-rank, the compute() needs a proper ghost-force reverse_comm
  handshake that mirrors pair_allegro's pattern.
- **Orthorhombic box assumption for PBC shifts.** Triclinic boxes work
  approximately but may produce wrong shift vectors near cell corners.
  TODO: use the full 3×3 inverse lattice for shift reconstruction.
- **Pure model currently diverges from DGL model** on the same trained
  weights by ~10% energy. Root cause: `forward_tensors_z`'s inline line
  graph construction enumerates ~9× more bond-angle triplets than DGL's
  `g.line_graph()`. To match DGL exactly, either retrain with the pure
  model directly or patch the line-graph filter.
- **fp32 only** in the C++ path. Changing `dtype_` in `pair_alignn.h` to
  `torch::kFloat64` also requires a matching fp64 TorchScript export and
  changed `accessor<double,2>` types.
- **No `eflag_atom` / `vflag_atom`** (per-atom energy/stress) — only
  global quantities.

## Speed reference

On a single NVIDIA RTX-class GPU, 216-atom Si at cutoff 5.0 Å:

| Path | ms / step |
|---|---|
| AlignnAtomwiseCalculator (Python via ASE) | ~40 |
| lammps_bridge_torch (Python callback) | ~15 |
| **pair_alignn (native C++)** | **~3–5** |

Bottleneck in the C++ path is the forward pass itself, not the neighbor
list building. Memory stays resident on the GPU between steps.

## Files in the repo

```
alignn/scripts/torch/
├── export_torchscript.py        # Export .pt TorchScript model
├── build_lammps_alignn.sh       # Full LAMMPS+libtorch build
├── validate_pair_alignn.py      # Python↔LAMMPS force/energy check
├── lammps_bridge_torch.py       # Alternative Python bridge (no C++)
└── pair_alignn/
    ├── pair_alignn.cpp          # Pair style source
    ├── pair_alignn.h            # Pair style header
    ├── si_native.in             # Example LAMMPS melt-quench
    └── CMakeLists-snippet.txt   # Standalone build fragment
```

## Related

- [MD & Relaxation (pure PyTorch)](md-integrators.md) — on-device
  integrators that skip LAMMPS entirely
- [ASE Calculator](ase-calculator.md) — the slower but more flexible
  Python path
