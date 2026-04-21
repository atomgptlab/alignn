"""Perturb a Cu FCC cell and relax with FIRE."""
import numpy as np, torch
from ase.build import bulk
from ase.data import atomic_masses

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)
from alignn.md import AlignnForces, FIRE


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    a = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))  # 32 atoms
    Z = a.get_atomic_numbers()
    cell = np.array(a.cell)
    pos = torch.tensor(a.get_positions(), dtype=dtype, device=device)
    # random perturbation ±0.15 Å
    pos = pos + 0.15 * torch.randn_like(pos)
    masses = torch.tensor([atomic_masses[z] for z in Z], dtype=dtype, device=device)

    # Toy harmonic forces_fn: V = 0.5 k (x - x0)², F = -k(x - x0).
    # This validates FIRE independently of the (untrained) ALIGNN-FF model.
    x0 = pos.clone().detach()
    k = 2.0  # eV/Å²
    def fn(positions):
        dx = positions - x0
        E = 0.5 * k * (dx * dx).sum()
        F = -k * dx
        return E, F
    # perturb AFTER defining x0 so forces are nonzero
    pos = pos + 0.30 * torch.randn_like(pos)

    relax = FIRE(forces_fn=fn, masses=masses, dt=0.1, dt_max=1.0, max_step=0.2)
    _, hist = relax.run(pos, fmax=0.05, max_steps=500, log_every=20)

    print(f"\ntotal logged points: {len(hist)}")
    print(f"initial |F|max = {hist[0]['fmax_eV_A']:.4f} eV/Å")
    print(f"final   |F|max = {hist[-1]['fmax_eV_A']:.4f} eV/Å")


if __name__ == "__main__":
    main()
