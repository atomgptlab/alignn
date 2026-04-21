"""Compare NVTBerendsen vs NVTBussi: relaxation + KE fluctuations.

Canonical KE variance:  <ΔKE²> = (3N/2) (kT)²  =>  σ(KE)/⟨KE⟩ = sqrt(2/3N).
Berendsen should under-fluctuate; Bussi should match.
"""
import numpy as np, torch
from ase.build import bulk
from ase.data import atomic_masses

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)
from alignn.md import (
    AlignnForces, NVTBerendsen, NVTBussi, NVTNoseHooverChain,
    run, maxwell_boltzmann,
)
from alignn.md.integrators import kinetic_energy


def setup(n=3, T=300.0, device="cuda", dtype=torch.float32):
    a = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((n, n, n))
    Z = a.get_atomic_numbers()
    cell = np.array(a.cell)
    pos = torch.tensor(a.get_positions(), dtype=dtype, device=device)
    masses = torch.tensor([atomic_masses[z] for z in Z], dtype=dtype, device=device)
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(
        name="alignn_atomwise_pure", calculate_gradient=True,
        atomwise_output_features=0, atom_input_features=92))
    fn = AlignnForces(model, Z, cell, device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(0)
    vel = maxwell_boltzmann(masses, T=T, generator=g)
    return pos, vel, masses, fn


def sample_KE(integ, pos, vel, nsteps=400, skip=100):
    """Run, collect KE samples after equilibration."""
    Ks = []
    for i in range(nsteps):
        pos, vel = integ.step(pos, vel)
        if i >= skip:
            Ks.append(kinetic_energy(integ.masses, vel).item())
    return np.array(Ks)


def main():
    torch.manual_seed(1)
    T = 300.0
    pos, vel, masses, fn = setup(n=3, T=T)  # 108 atoms
    N = masses.numel()
    KE_mean_expected = 0.5 * (3*N - 3) * 8.617e-5 * T
    sigma_expected = KE_mean_expected * np.sqrt(2.0 / (3*N - 3))
    print(f"N={N}   expected ⟨KE⟩ = {KE_mean_expected:.3f} eV   "
          f"expected σ(KE) = {sigma_expected:.4f} eV")

    for name, integ in [
        ("Berendsen", NVTBerendsen(forces_fn=fn, masses=masses, dt=1.0, T=T, taut=20.0)),
        ("Bussi",     NVTBussi   (forces_fn=fn, masses=masses, dt=1.0, T=T, taut=20.0)),
        ("NH-chain3", NVTNoseHooverChain(forces_fn=fn, masses=masses, dt=1.0, T=T, taut=20.0, chain_length=3)),
    ]:
        Ks = sample_KE(integ, pos.clone(), vel.clone(), nsteps=400, skip=100)
        print(f"{name:10s}  ⟨KE⟩={Ks.mean():.3f} eV   σ={Ks.std():.4f} eV   "
              f"σ/σ_expected={Ks.std()/sigma_expected:.2f}")


if __name__ == "__main__":
    main()
