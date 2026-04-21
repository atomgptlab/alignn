"""Demo: NVE (Velocity-Verlet) and NVT (Langevin) MD on a Cu supercell
using pure-PyTorch ALIGNN-FF + pure-PyTorch integrators.

This uses a random-init model for the integrator correctness check
(energy conservation under NVE, temperature stabilization under NVT).
Swap in trained weights for production.
"""
import numpy as np, torch
from ase.build import bulk
from ase.data import atomic_masses

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)
from alignn.md import AlignnForces, VelocityVerlet, Langevin, run
from alignn.md.integrators import maxwell_boltzmann


def make_model():
    cfg = ALIGNNAtomWisePureConfig(
        name="alignn_atomwise_pure",
        calculate_gradient=True,
        atomwise_output_features=0,
        atom_input_features=92,
    )
    return ALIGNNAtomWisePure(cfg)


def setup(n=4, T0=300.0, device="cuda", dtype=torch.float32):
    ase_atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((n, n, n))
    N = len(ase_atoms)
    Z = ase_atoms.get_atomic_numbers()
    cell = np.array(ase_atoms.cell)
    pos = torch.tensor(ase_atoms.get_positions(), dtype=dtype, device=device)
    masses = torch.tensor(
        [atomic_masses[z] for z in Z], dtype=dtype, device=device
    )
    model = make_model()
    forces_fn = AlignnForces(model, Z, cell, device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(0)
    vel = maxwell_boltzmann(masses, T=T0, generator=g)
    print(f"N atoms = {N}")
    return pos, vel, masses, forces_fn


def demo_nve(nsteps=200, n=3):
    pos, vel, masses, fn = setup(n=n, T0=300.0)
    integ = VelocityVerlet(forces_fn=fn, masses=masses, dt=1.0)
    print("\n--- NVE (Velocity-Verlet) ---")
    _, _, hist = run(integ, pos, vel, nsteps=nsteps, log_every=20)
    # energy conservation metric
    T = [r["T_K"] for r in hist]
    print(f"\nT drift: min={min(T):.1f}  max={max(T):.1f}  range={max(T)-min(T):.1f} K")


def demo_nvt(nsteps=400, n=3, T=300.0):
    pos, vel, masses, fn = setup(n=n, T0=T)
    integ = Langevin(forces_fn=fn, masses=masses, dt=1.0, T=T, friction=0.01)
    print(f"\n--- NVT Langevin (target {T} K) ---")
    _, _, hist = run(integ, pos, vel, nsteps=nsteps, log_every=40)
    T_tail = np.mean([r["T_K"] for r in hist[-5:]])
    print(f"\nmean T over last 5 log points: {T_tail:.1f} K  (target {T} K)")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_nve(nsteps=100, n=3)
    demo_nvt(nsteps=200, n=3, T=300.0)
