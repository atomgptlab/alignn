# Pure-PyTorch MD & Relaxation

`alignn.md` provides on-device molecular dynamics integrators and a FIRE
relaxer that pair with the pure-PyTorch ALIGNN-FF model
(`ALIGNNAtomWisePure`). Everything runs on GPU without bouncing through
ASE/NumPy each step, which is the prerequisite for scaling to large
systems and (eventually) multi-GPU.

## When to use this vs. the ASE calculator

| Use | Pick |
|---|---|
| Small systems, quick scripts, trajectory files, ensembles you trust (NPT, surfaces, constraints, free-energy tools) | ASE calculator (`AlignnAtomwiseCalculator`) |
| Large systems where per-step CPU→GPU transfer dominates | `alignn.md` integrators |
| Low-precision (bf16/fp16) or multi-GPU MD | `alignn.md` integrators |
| Differentiable MD / gradients through trajectories | `alignn.md` integrators |

The two paths share the same model code; you can mix them (e.g.,
equilibrate with ASE, production with `alignn.md`).

## Components

```python
from alignn.md import (
    AlignnForces,               # wraps a pure-PyTorch model as forces_fn
    VelocityVerlet,             # NVE
    Langevin,                   # NVT — stochastic, BAOAB splitting
    NVTBerendsen,               # weak-coupling (equilibration only)
    NVTBussi,                   # canonical NVT — stochastic rescale
    NVTNoseHooverChain,         # canonical NVT — deterministic chain
    FIRE,                       # structural relaxation
    run, maxwell_boltzmann,
)
```

All integrators share a `step(positions, velocities) -> (positions, velocities)`
API. Positions are `(N, 3)` Å, velocities `(N, 3)` Å/fs, masses `(N,)` amu,
time `fs`, energy `eV`, temperature `K` — ASE's unit convention.

## Forces function

Every integrator takes a `forces_fn(positions) -> (energy, forces)`.
`AlignnForces` wraps the pure-PyTorch model:

```python
import numpy as np, torch
from ase.build import bulk
from ase.data import atomic_masses
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)
from alignn.md import AlignnForces

a = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
Z = a.get_atomic_numbers()
cell = np.array(a.cell)

model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(
    name="alignn_atomwise_pure",
    calculate_gradient=True,
    atomwise_output_features=0,
    atom_input_features=92,
))
# load trained weights if you have them:
# model.load_state_dict(torch.load("OutputDir/best_model.pt"), strict=False)

forces_fn = AlignnForces(model, Z, cell, device="cuda", dtype=torch.float32)
```

!!! warning "Neighbor-list bottleneck"
    The current `AlignnForces` rebuilds the neighbor/line graph on CPU
    every step. For large systems this dominates wall time. An on-device
    neighbor list is planned.

## NVE — `VelocityVerlet`

Symplectic, time-reversible, second-order. Energy conservation is the
standard sanity check.

```python
from alignn.md import VelocityVerlet, maxwell_boltzmann, run

masses = torch.tensor([atomic_masses[z] for z in Z],
                      dtype=torch.float32, device="cuda")
pos = torch.tensor(a.get_positions(), dtype=torch.float32, device="cuda")
vel = maxwell_boltzmann(masses, T=300.0)

integ = VelocityVerlet(forces_fn=forces_fn, masses=masses, dt=1.0)
pos, vel, hist = run(integ, pos, vel, nsteps=1000, log_every=100)
```

## NVT — four options

| Thermostat | Ensemble | Dynamics preserved? | Best for |
|---|---|---|---|
| `NVTBerendsen` | **not canonical** (crushes KE fluctuations) | Yes (deterministic) | Equilibration, melt-quench |
| `Langevin` | Canonical | No (adds friction + noise) | General NVT |
| `NVTBussi` | Canonical | Nearly — velocity-direction preserved | General NVT; fast relaxation |
| `NVTNoseHooverChain` | Canonical | Yes (time-reversible) | Transport coefficients, spectra |

### Berendsen

Fast, stable, but **samples a distribution with wrong variance** — safe
only for equilibration. Exposes `.set_temperature(T)` for ramping.

```python
from alignn.md import NVTBerendsen
integ = NVTBerendsen(forces_fn=forces_fn, masses=masses,
                    dt=1.0, T=300.0, taut=20.0)
```

### Langevin (BAOAB splitting)

Solves $\ddot{x} = F/m - \gamma \dot{x} + \sqrt{2\gamma k_B T / m}\,\eta$
with the Leimkuhler-Matthews BAOAB scheme.

```python
from alignn.md import Langevin
integ = Langevin(forces_fn=forces_fn, masses=masses,
                 dt=1.0, T=300.0, friction=0.01)   # 1/fs
```

Friction 0.01 fs⁻¹ = 100 fs damping time. Use larger γ for faster
equilibration, smaller γ to perturb dynamics less.

### Bussi-Donadio-Parrinello

Stochastic rescale (Bussi et al., JCP 126, 014101, 2007). Same `taut`
interface as Berendsen but samples the true canonical distribution. **Use
this as your default NVT** unless you specifically need deterministic
dynamics.

```python
from alignn.md import NVTBussi
integ = NVTBussi(forces_fn=forces_fn, masses=masses,
                 dt=1.0, T=300.0, taut=20.0)
```

### Nosé-Hoover chain

Martyna-Tobias-Klein integrator. Deterministic and time-reversible —
preferred when measuring transport coefficients (diffusion, viscosity,
VACF) where stochastic thermostats perturb the relevant dynamics.
Default chain length 3 (never use `chain_length=1` — it's
non-ergodic).

```python
from alignn.md import NVTNoseHooverChain
integ = NVTNoseHooverChain(forces_fn=forces_fn, masses=masses,
                           dt=1.0, T=300.0, taut=20.0,
                           chain_length=3)
```

### Verifying "true NVT"

Canonical KE should fluctuate with
$\sigma(KE)/\langle KE \rangle = \sqrt{2/(3N - 3)}$. Measured on a 108-atom
Cu system at 300 K (random-init model, isolates thermostat behavior):

| Thermostat | ⟨KE⟩ eV | σ(KE) / σ_canonical |
|---|---|---|
| Berendsen | 4.143 ✓ | 0.04 (crushed) |
| Bussi | 4.173 ✓ | 1.13 ✓ |
| NH-chain | 4.171 ✓ | 0.52 (correlated, converges with longer runs) |

See `md_test_bussi.py` in the repo root.

## FIRE relaxation

Fast Inertial Relaxation Engine (Bitzek et al., PRL 97, 170201, 2006).
ASE-compatible defaults.

```python
from alignn.md import FIRE

relax = FIRE(forces_fn=forces_fn, masses=masses,
             dt=0.1, dt_max=1.0, max_step=0.2)
relaxed_pos, history = relax.run(
    positions, fmax=0.05, max_steps=500, log_every=10,
)
```

**Defaults match ASE:** `dt=0.1`, `dt_max=1.0`, `max_step=0.2`,
`N_min=5`, `f_inc=1.1`, `f_dec=0.5`, `α_start=0.1`, `f_α=0.99`.
Convergence: `max(|F_i|) < fmax` in eV/Å.

**Not yet supported:** variable-cell relaxation (no `ExpCellFilter`
equivalent), fixed-atom constraints. Use the ASE calculator path for
those.

## Velocity initialization

```python
from alignn.md import maxwell_boltzmann

vel = maxwell_boltzmann(masses, T=300.0, generator=torch.Generator(device="cuda").manual_seed(0))
# COM drift removed and velocities rescaled to hit T exactly.
```

## The `run` driver

```python
def cb(step, row):
    print(f"t={row['time_fs']:.1f} fs  T={row['T_K']:.1f} K")

pos, vel, history = run(integ, pos, vel,
                        nsteps=10000, log_every=100, callback=cb)
```

Each log row contains `step`, `time_fs`, `T_K`, `KE_eV`, `wall_s`.

## Melt-quench workflow (full example)

Drop-in replacement for the ASE `NVTBerendsen` melt-quench pattern:

```python
from alignn.md import AlignnForces, NVTBerendsen, run, maxwell_boltzmann
from alignn.md.integrators import wrap_pbc

forces_fn = AlignnForces(model, Z, cell_np, device="cuda")
vel = maxwell_boltzmann(masses, T=3500.0)

integ = NVTBerendsen(forces_fn=forces_fn, masses=masses,
                    dt=1.0, T=3500.0, taut=20.0)

# Stage 1 — melt
pos, vel, _ = run(integ, pos, vel, nsteps=1000, log_every=20)

# Stage 2 — quench
integ.set_temperature(300.0)
pos, vel, _ = run(integ, pos, vel, nsteps=2000, log_every=20)

pos = wrap_pbc(pos, cell)    # tidy coordinates for POSCAR output
```

See `md_melt_quench.py` in the repo root for a complete script with
JARVIS loading and POSCAR output.

## Choosing a thermostat — quick guide

- **Equilibrating a new structure:** Berendsen or Bussi with `taut = 20-100 fs`
- **Production sampling at fixed T:** Bussi (fast, simple) or NHC (if
  you need dynamics)
- **Transport coefficients (D, η, κ):** NHC (`chain_length=3-5`) with
  weak coupling (`taut ≈ 100-500 fs`)
- **Melt-quench:** Berendsen for the ramp, Bussi for any equilibration at
  the target T
- **Free energies, heat capacity, structural fluctuations:** Bussi or NHC
  — **never** Berendsen

## Known limitations

1. **CPU neighbor list** rebuilt every step (see `AlignnForces._build_graph`).
   Unblocks once replaced with an on-device version.
2. **No trajectory file output** — plug an `ase.io.Trajectory` writer
   into the `callback` parameter of `run()` if needed.
3. **No NPT** — planned (Parrinello-Rahman / MTK barostat).
4. **No constraints** (fixed atoms, SHAKE/RATTLE, symmetry).
5. **Single GPU** — multi-GPU domain decomposition is a separate roadmap
   item.

## Related

- [ASE Calculator](ase-calculator.md) — the canonical, full-featured
  path via ASE.
- `ALIGNNAtomWisePure` in `alignn.models.alignn_atomwise_pure` — the
  DGL-free model powering these integrators.
