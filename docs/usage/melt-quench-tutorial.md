# Tutorial — Train → Export → LAMMPS Melt-Quench → RDF

End-to-end walkthrough: train an ALIGNN-FF model on a Si dataset, export
it to TorchScript, run a melt-quench simulation in LAMMPS with the native
`pair_alignn` style, and plot the radial distribution function (RDF) of
the quenched structure.

Prereqs:

- `alignn` installed (`pip install -e .` from a clone)
- `lammps` Python package compiled with `pair_alignn` (see
  [LAMMPS Pair Style](lammps-pair-style.md) for the build)
- Plotly (`pip install plotly`)

Working directory for this tutorial: `MELT/` (create anywhere).

---

## 1. Train the model

Standard training command — nothing new here:

```bash
train_alignn.py \
    --root_dir DataDir/ \
    --config config.json \
    --output_dir OutputDir
```

When training finishes, `OutputDir/` contains:

```
OutputDir/
├── best_model.pt        # weights
├── config.json          # hyperparameters + cutoff + neighbor_strategy
├── history_*.json
└── ...
```

**Important — requirements for the LAMMPS path:**

1. In `config.json`, set `model.name: "alignn_atomwise_pure"` so
   `forward_tensors_z` (the TorchScript entry point) is available. If
   you trained with `alignn_atomwise` (DGL-backed), see the note at the
   end of this page.
2. Set `calculate_gradient: true` for forces to be computed.
3. Note `cutoff` and `max_neighbors` from `config.json` — you'll pass
   them to `pair_style alignn`.

Quick look-up:
```bash
python -c "
import json
c = json.load(open('OutputDir/config.json'))
print('cutoff        :', c['cutoff'], 'Å')
print('max_neighbors :', c['max_neighbors'])
print('atom_features :', c.get('atom_features'))
print('model.name    :', c['model']['name'])
"
```

---

## 2. Export to TorchScript

After `pip install -e .`, the export script is on your PATH:

```bash
export_torchscript.py \
    --model-dir OutputDir \
    --out alignn_ff.pt \
    --atom-features atomic_number \
    --dtype float32
```

Expected output:
```
smoke test OK: {'energy': torch.Size([]), 'forces': torch.Size([2, 3]), 'stress': torch.Size([3, 3])}
wrote alignn_ff.pt  (143,297 params)
```

**`--atom-features` must match training.** For most ALIGNN-FF trainings
this is `atomic_number` (single scalar per atom) or `cgcnn` (92-dim).
Check your config's top-level `atom_features` field.

---

## 3. Build a Si structure

LAMMPS needs a data file. Build a 216-atom cubic Si supercell:

```python
# make_si_data.py
from ase.build import bulk
from ase.io import write

# Cubic (orthorhombic) conventional cell, repeated 3×3×3 → 216 atoms.
# Avoid triclinic primitive cells when possible — pair_alignn's shift
# computation is most accurate for orthorhombic boxes.
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
write("si.data", atoms, format="lammps-data", specorder=["Si"])
print(f"wrote si.data  N={len(atoms)}  cell={atoms.cell.lengths()}")
```

```bash
python make_si_data.py
# wrote si.data  N=216  cell=[16.29 16.29 16.29]
```

---

## 4. Write the LAMMPS input script

Five-stage melt-quench: equilibrate → heat → hold molten → quench →
anneal. Save as `si_melt_quench.in`:

```
# Si melt-quench via pair_alignn.
units           metal                   # eV, Å, ps, bar, K
atom_style      atomic
boundary        p p p
read_data       si.data
mass            1 28.0855

# ── ALIGNN-FF native pair style ──
pair_style      alignn 5.0 12           # cutoff (Å), max_neighbors
pair_coeff      * * alignn_ff.pt Si

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        0.001                   # 1 fs
thermo          100
thermo_style    custom step temp pe ke etotal press vol
dump            traj all custom 200 si.lammpstrj id type x y z

# ── Stage 1 — equilibrate 300 K (0.2 ps) ──
velocity        all create 300.0 12345 mom yes rot yes
fix             nvt1 all nvt temp 300.0 300.0 0.1
run             200
unfix           nvt1

# ── Stage 2 — heat to 3500 K (1 ps) ──
fix             nvt2 all nvt temp 300.0 3500.0 0.1
run             1000
unfix           nvt2

# ── Stage 3 — hold molten (2 ps) ──
fix             nvt3 all nvt temp 3500.0 3500.0 0.1
run             2000
unfix           nvt3

# ── Stage 4 — quench to 300 K (4 ps) ──
fix             nvt4 all nvt temp 3500.0 300.0 0.1
run             4000
unfix           nvt4

# ── Stage 5 — anneal 300 K (2 ps) — RDF measured here ──
compute         rdf all rdf 150 1 1 cutoff 6.0
fix             rdf_out all ave/time 100 10 1000 c_rdf[*] file si_rdf.dat mode vector
fix             nvt5 all nvt temp 300.0 300.0 0.1
run             2000
unfix           nvt5
unfix           rdf_out

write_data      si_quenched.data
```

The `compute rdf` + `fix ave/time` block samples the radial distribution
function every 100 steps and writes the averaged result to `si_rdf.dat`.

---

## 5. Run it

```bash
~/lammps-alignn/build/lmp -in si_melt_quench.in
```

Or from Python:
```python
from lammps import lammps
lmp = lammps()
lmp.file("si_melt_quench.in")
```

Runtime on a single modest GPU: ~10–30 min for the full 9.2 ps. Tail of
the thermo output will show temperature ramping up and coming back down:

```
Step  Temp  PotEng   KinEng   TotEng    Press        Volume
0     300   -987.23  5.58     -981.65   -5.2e4       4318.9
200   842   -984.11  15.67    -968.44   -2.3e6       4318.9
...
3200  3501  -961.22  65.12    -896.10   +3.1e5       4318.9
...
9200  301   -985.40  5.60     -979.80   -1.4e4       4318.9
```

**Known quirk**: your pressure numbers will look extreme if your model
was trained with `stresswise_weight: 0.0` — stresses aren't learned,
only energies and forces. That's fine for NVT; don't use NPT with such a
model.

---

## 6. Plot the RDF

The `si_rdf.dat` file LAMMPS wrote has a specific multi-block format.
This script parses it and plots with Plotly:

```python
# plot_rdf.py
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def parse_lammps_rdf(path: str | Path):
    """Parse LAMMPS `fix ave/time ... mode vector` RDF output.

    Each averaged block begins with a header line `<timestep> <nbins>`
    followed by `nbins` lines of `index r g(r) coord(r)`.
    Returns the LAST block (steady-state RDF).
    """
    with open(path) as f:
        lines = [L.rstrip() for L in f if L.strip() and not L.startswith("#")]
    # Find block starts (lines with exactly 2 tokens = timestep, nbins)
    starts = [i for i, L in enumerate(lines)
              if len(L.split()) == 2 and not L.startswith(" ")]
    last = starts[-1]
    nbins = int(lines[last].split()[1])
    data = np.array([list(map(float, L.split())) for L in lines[last+1:last+1+nbins]])
    r    = data[:, 1]
    g_r  = data[:, 2]
    coord = data[:, 3]
    return r, g_r, coord

def plot_rdf(r, g_r, coord, title="Si RDF after melt-quench"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=g_r, name="g(r)",
                             line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=r, y=coord, name="coordination",
                             line=dict(color="firebrick", width=1.5, dash="dash"),
                             yaxis="y2"))
    fig.update_layout(
        title=title, template="plotly_white",
        width=800, height=500,
        xaxis=dict(title="r (Å)"),
        yaxis =dict(title="g(r)",         titlefont=dict(color="steelblue")),
        yaxis2=dict(title="coordination", overlaying="y", side="right",
                   titlefont=dict(color="firebrick"), showgrid=False),
    )
    return fig

if __name__ == "__main__":
    r, g_r, coord = parse_lammps_rdf("si_rdf.dat")
    # Report key features
    i_peak = np.argmax(g_r)
    print(f"first-neighbor peak: r = {r[i_peak]:.3f} Å,  g(r) = {g_r[i_peak]:.2f}")
    print(f"coordination at 3.0 Å: {np.interp(3.0, r, coord):.2f} "
          f"(expected ≈ 4 for crystalline Si, lower for amorphous)")
    fig = plot_rdf(r, g_r, coord)
    fig.write_html("si_rdf.html")
    print("wrote si_rdf.html")
```

Run it:
```bash
python plot_rdf.py
```

**Expected Si RDF:**

| Feature | Crystalline Si | Amorphous Si (well-relaxed) | Amorphous Si (poorly quenched) |
|---|---|---|---|
| First peak position | 2.35 Å | 2.35–2.40 Å | similar |
| First peak height | sharp, g>10 | broad, g≈3–5 | low, g≈2 |
| 2nd-neighbor splitting at ~3.8 Å | distinct peak | shoulder or gone | absent |
| Coordination at 3.0 Å | 4.0 (exact) | ~3.8–4.2 | variable |

A **4 ps quench from 3500 K is very fast** — don't expect a perfectly
amorphous or perfectly crystalline result. For publication-quality
glasses, quench over 100+ ps.

---

## 7. Verifying the run was physical

Sanity checks before trusting the results:

1. **Energy conservation during NVE windows.** Add a short NVE block and
   watch `etotal` — drift should be < 1e-4 eV/atom/ps.
2. **No model blow-ups.** If temperature suddenly spikes or PE diverges,
   the force call probably hit an unphysical configuration outside the
   model's training domain. Use a shorter timestep (0.0005 ps) or a
   softer heating ramp.
3. **Compare to the Python calculator.** Run one single-point with
   `AlignnAtomwiseCalculator` on the starting structure; LAMMPS should
   match to ~1e-4 eV for energy. If it doesn't, see [Model-side
   caveats](#model-side-caveats) below.
4. **Check the trajectory.** Open `si.lammpstrj` in OVITO; atoms
   shouldn't fly out of the box.

---

## Model-side caveats

!!! warning "`alignn_atomwise` vs `alignn_atomwise_pure`"

    The TorchScript entry point `forward_tensors_z` only exists on
    `ALIGNNAtomWisePure`. If your training config used `name:
    alignn_atomwise` (the DGL-backed variant), `export_torchscript.py`
    will still run by loading the DGL weights into Pure — but the two
    models are not numerically identical on the same weights (currently
    a ~10% energy difference due to line-graph construction
    differences).

    **Clean fix**: retrain with `name: alignn_atomwise_pure` in your
    training config. All other hyperparameters can stay the same.

!!! warning "Stress not learned"

    If your training data didn't include stresses (or
    `stresswise_weight: 0.0`), `pair_alignn` still returns virial
    numbers, but they're noise. Don't use NPT / `fix npt`.

!!! warning "Single MPI rank"

    Current `pair_alignn` silently drops cross-rank ghost contributions.
    Run with `mpirun -n 1` only. For larger systems, increase
    parallelism via threads (`OMP_NUM_THREADS`) until multi-rank ghost
    handling lands.

---

## What to try next

- **Transport**: replace NVT with NVE in stage 5, compute diffusion
  coefficient from the mean-squared-displacement (LAMMPS
  `compute msd`).
- **Cool more slowly**: extend stage 4 to 40 ps and compare the RDF —
  you should see sharper peaks and a clearer second-neighbor split.
- **Longer correlation**: rerun with `mode vector 10` (average over 10
  snapshots per write) for smoother RDFs.
- **Different material**: retrain on another element / alloy, re-export,
  swap `Si` in `pair_coeff` for the target species.

---

## Files produced by this tutorial

```
MELT/
├── alignn_ff.pt           # exported TorchScript model
├── si.data                # LAMMPS input structure
├── si_melt_quench.in      # LAMMPS input script
├── si.lammpstrj           # trajectory (custom dump)
├── si_rdf.dat             # RDF samples (fix ave/time)
├── si_quenched.data       # final structure
├── plot_rdf.py            # RDF plotter
└── si_rdf.html            # interactive RDF plot
```

---

## See also

- [LAMMPS Pair Style (native C++)](lammps-pair-style.md) — full
  reference for `pair_alignn`
- [MD & Relaxation (pure PyTorch)](md-integrators.md) — on-device
  integrators as an alternative to LAMMPS
- [ASE Calculator](ase-calculator.md) — the Python-side calculator
  used for the reference comparison in step 7
