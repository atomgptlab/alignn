# LAMMPS + ALIGNN-FF (pure-torch) — handoff notes

## What's here

| File | Purpose |
| --- | --- |
| `get_model.py` | Downloads the figshare `mps` ALIGNN-FF checkpoint and exports it to TorchScript (`alignn_ff.pt`) using `alignn.scripts.torch.export_torchscript`. |
| `export_smooth.py` | Exports a **smooth-variant** checkpoint (`alignn_atomwise_pure_smooth`) to TorchScript. The stock exporter hardcodes the non-smooth class. |
| `build_si.py` | Builds a 54-atom Si supercell from JARVIS JVASP-1002, relaxes it with `AlignnAtomwiseCalculator + ExpCellFilter + FIRE`, writes `si.data`. |
| `build_sio2.py` | Same for α-quartz JVASP-41 (162 atoms), writes `sio2.data`. |
| `nve_si_stability.in` | NVE energy-conservation test on Si. |
| `melt_quench_si.in` | 9 ps Si melt-quench (300 K → 3500 K → 300 K). |
| `melt_quench_sio2.in` | 13.5 ps SiO₂ melt-quench (300 K → 4000 K → 300 K). |
| `alignn_ff.pt` | Currently the **smooth** TorchScript model (143k params). |
| `alignn_ff_mps.pt` | Backup of the non-smooth `mps` export, kept for comparison. |

## How it runs

```bash
# 1. Build LAMMPS once with pair_alignn (see ../../scripts/torch/build_lammps_alignn.sh)

# 2. Get a model:
#    a) non-smooth mps checkpoint from figshare
python get_model.py
#    b) OR the smooth checkpoint (currently the active alignn_ff.pt)
python export_smooth.py \
    --model-dir ~/Software/ollama311/alignn/MELT_smooth_long/OutputDir \
    --out alignn_ff.pt

# 3. Build relaxed structures, then run
python build_si.py            && lmp -in nve_si_stability.in
python build_si.py            && lmp -in melt_quench_si.in
python build_sio2.py          && lmp -in melt_quench_sio2.in
```

## Gotchas discovered while bringing this up

1. **`pair_style alignn` needs explicit cutoff and max_neighbors** —
   syntax is `pair_style alignn <cutoff> <max_neighbors>`. These **must
   match the model's training config**: look at `cutoff` and
   `max_neighbors` at the top level of `config.json` (not under
   `model:`). For both checkpoints used here that's `5.0 12`. Passing
   `8.0 25` puts the model far out of distribution — saw 200 neighbors/atom
   and immediate blow-up.

2. **Relax with the model before LAMMPS.** The unrelaxed JARVIS
   geometries (especially JVASP-1002 Si) give ~MBar residual stress
   under both checkpoints — LAMMPS `fix box/relax` can't escape the
   minimum, so MD blows up within a few hundred steps. `build_si.py` /
   `build_sio2.py` now run `ExpCellFilter + FIRE(fmax=0.05)` first.

3. **`reset_timestep` must come before any `dump`** (or you `undump`
   first). Initial NVE file had the dump active across the reset.

4. **`pair_alignn.cpp settings(...)` reads the args** — confirmed it
   keeps the `max_neighbors_` closest neighbors within `cutoff_`.
   So the model receives exactly the graph it was trained on; the
   instability problems were the geometry and the model itself,
   not the bridge.

## Stability comparison: non-smooth vs smooth

5 ps NVE on relaxed 54-atom Si, 1 fs timestep, both models with
identical 143k params, both trained with `atom_input_features=1`,
`cutoff=5.0`, `max_neighbors=12`:

| Metric | non-smooth (`mps`) | smooth (`MELT_smooth_long`) |
| --- | --- | --- |
| Total energy drift rate | +12.3 meV/atom/ps | **+0.47 meV/atom/ps** |
| Total energy range over 5 ps | 4424 meV | **112 meV** |
| Temperature trend | 113 K → 279 K (heating) | **89 K → 94 K** (flat) |
| Temperature range | [89, 345] K | **[76, 105] K** |

The smooth variant's polynomial cutoff envelope (`CutoffPolynomial`,
coeff=5.0) eliminates force discontinuities at the cutoff. Net effect:
forces are essentially conservative, so NVE behaves like NVE instead
of slowly heating up.

The non-smooth `mps` checkpoint is bounded (atoms don't fly off) but
**not energy-conserving** to MLIP standards. It's clearly a smoke-test
checkpoint — `stresswise_weight=0` in its training config, so stress
was never fit.

## Patches to the model source (needed to TorchScript-export the smooth variant)

`alignn/models/alignn_atomwise_pure_smooth.py`:

- `np.sqrt(2.0)` and `np.sqrt(np.pi)` inside `Fourier.forward()` →
  replaced with the float constants. TorchScript can't take weak
  references to numpy ufuncs.
- `self.radial_envelope` was only assigned in the gaussian branch.
  Added an explicit `Optional[CutoffPolynomial]` class-body
  annotation (and default `= None` before the basis-kind switch) so
  TorchScript can compile `_embed_radial` regardless of which branch
  is taken at construction.
- Imported `Optional` from `typing`.

## Open / next

- The MD runs use CPU torch (≈30 steps/s for 54 atoms, ≈12 steps/s
  for 162 atoms). For longer runs export with CUDA — `pair_alignn.cpp`
  already picks CUDA automatically if `torch::cuda::is_available()`.
- The SiO₂ melt-quench is short (13.5 ps); for production-quality
  amorphous silica it should be ≥50 ps total with a slower (~0.2 K/fs)
  quench.
- The non-smooth `mps` checkpoint is not worth rerunning for science;
  keep it only as a TorchScript-export smoke test.
