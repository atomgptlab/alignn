"""Pure-PyTorch MD integrators: Velocity-Verlet (NVE) and Langevin (NVT, BAOAB).

State kept entirely on-device. Unit system follows ASE:
    energy eV, length Å, mass amu, time fs, temperature K.
With these units, the acceleration factor `F/m` needs the conversion
    1 eV/(Å·amu) = 9.6485e-3 Å/fs²   (i.e., ASE's `ase.units.fs` scaling)
which we bake in as `EV_AMU_A_PER_FS2`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import math, time
import torch

KB_EV = 8.617333262e-5                # Boltzmann, eV/K
EV_AMU_A_PER_FS2 = 9.6485332e-3        # eV/(Å·amu) -> Å/fs²

ForcesFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def wrap_pbc(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap positions into the primary unit cell (3x3 cell, row vectors)."""
    inv = torch.linalg.inv(cell)
    frac = positions @ inv
    frac = frac - torch.floor(frac)
    return frac @ cell


def kinetic_energy(masses: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    # KE in eV: 0.5 * m * v²  with m in amu, v in Å/fs -> multiply by 1/EV_AMU_A_PER_FS2
    return 0.5 * (masses * (velocities ** 2).sum(dim=-1)).sum() / EV_AMU_A_PER_FS2


def instantaneous_temperature(masses, velocities):
    ke = kinetic_energy(masses, velocities)
    n_dof = 3 * masses.numel() - 3  # remove COM
    return 2.0 * ke / (n_dof * KB_EV)


def maxwell_boltzmann(masses: torch.Tensor, T: float, generator=None) -> torch.Tensor:
    # sigma_v = sqrt(kT/m), in Å/fs after unit conversion
    sigma = torch.sqrt(KB_EV * T * EV_AMU_A_PER_FS2 / masses).unsqueeze(-1)
    v = torch.randn((masses.numel(), 3), device=masses.device,
                    dtype=masses.dtype, generator=generator) * sigma
    # remove COM drift
    v -= (masses.unsqueeze(-1) * v).sum(dim=0, keepdim=True) / masses.sum()
    # Rescale to exact target T (removes finite-sample noise)
    T_now = instantaneous_temperature(masses, v)
    v = v * torch.sqrt(torch.tensor(T, device=v.device, dtype=v.dtype) / T_now)
    return v


# ---------------------------------------------------------------------------
# NVE: Velocity-Verlet
# ---------------------------------------------------------------------------
@dataclass
class VelocityVerlet:
    forces_fn: ForcesFn
    masses: torch.Tensor          # (N,)  amu
    dt: float = 1.0               # fs

    def __post_init__(self):
        self._forces = None

    def step(self, positions, velocities):
        if self._forces is None:
            _, self._forces = self.forces_fn(positions)
        m = self.masses.unsqueeze(-1)
        a = EV_AMU_A_PER_FS2 * self._forces / m
        # Half-kick
        velocities = velocities + 0.5 * self.dt * a
        # Drift
        positions = positions + self.dt * velocities
        # New forces
        _, new_forces = self.forces_fn(positions)
        a_new = EV_AMU_A_PER_FS2 * new_forces / m
        # Half-kick
        velocities = velocities + 0.5 * self.dt * a_new
        self._forces = new_forces
        return positions, velocities


# ---------------------------------------------------------------------------
# NVT: Langevin via BAOAB splitting (Leimkuhler & Matthews 2013)
# ---------------------------------------------------------------------------
@dataclass
class Langevin:
    forces_fn: ForcesFn
    masses: torch.Tensor
    dt: float = 1.0               # fs
    T: float = 300.0              # K
    friction: float = 0.01        # 1/fs  (typical: 0.01 = 100 fs damping)
    generator: Optional[torch.Generator] = None

    def __post_init__(self):
        self._forces = None
        self.c1 = math.exp(-self.friction * self.dt)
        self.c3 = math.sqrt(1.0 - self.c1 ** 2)

    def step(self, positions, velocities):
        if self._forces is None:
            _, self._forces = self.forces_fn(positions)
        m = self.masses.unsqueeze(-1)
        # B: half-kick
        a = EV_AMU_A_PER_FS2 * self._forces / m
        velocities = velocities + 0.5 * self.dt * a
        # A: half-drift
        positions = positions + 0.5 * self.dt * velocities
        # O: Ornstein-Uhlenbeck in velocity
        sigma = torch.sqrt(KB_EV * self.T * EV_AMU_A_PER_FS2 / self.masses).unsqueeze(-1)
        noise = torch.randn_like(velocities) if self.generator is None \
                else torch.randn(velocities.shape, device=velocities.device,
                                 dtype=velocities.dtype, generator=self.generator)
        velocities = self.c1 * velocities + self.c3 * sigma * noise
        # A: half-drift
        positions = positions + 0.5 * self.dt * velocities
        # B: half-kick with new forces
        _, new_forces = self.forces_fn(positions)
        a_new = EV_AMU_A_PER_FS2 * new_forces / m
        velocities = velocities + 0.5 * self.dt * a_new
        self._forces = new_forces
        return positions, velocities


# ---------------------------------------------------------------------------
# NVT: Berendsen weak-coupling thermostat
# (Not a true ensemble — fine for equilibration / melt-quench, not for
# equilibrium sampling or fluctuation-based observables.)
# ---------------------------------------------------------------------------
@dataclass
class NVTBerendsen:
    forces_fn: ForcesFn
    masses: torch.Tensor
    dt: float = 1.0               # fs
    T: float = 300.0              # K (target; mutable via set_temperature)
    taut: float = 20.0            # fs  (relaxation time)

    def __post_init__(self):
        self._forces = None

    def set_temperature(self, T: float):
        self.T = float(T)

    def step(self, positions, velocities):
        if self._forces is None:
            _, self._forces = self.forces_fn(positions)
        m = self.masses.unsqueeze(-1)
        # VV half-kick
        a = EV_AMU_A_PER_FS2 * self._forces / m
        velocities = velocities + 0.5 * self.dt * a
        # drift
        positions = positions + self.dt * velocities
        # new forces
        _, new_forces = self.forces_fn(positions)
        a_new = EV_AMU_A_PER_FS2 * new_forces / m
        velocities = velocities + 0.5 * self.dt * a_new
        # Berendsen velocity rescale
        T_now = instantaneous_temperature(self.masses, velocities)
        # Guard against T_now == 0 early on
        scale = torch.sqrt(
            1.0 + (self.dt / self.taut) * (self.T / T_now.clamp_min(1e-6) - 1.0)
        )
        velocities = velocities * scale
        self._forces = new_forces
        return positions, velocities


# ---------------------------------------------------------------------------
# NVT: Bussi-Donadio-Parrinello stochastic velocity rescaling (CSVR)
# Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
# Samples the true canonical distribution with Berendsen-like relaxation.
# ---------------------------------------------------------------------------
@dataclass
class NVTBussi:
    forces_fn: ForcesFn
    masses: torch.Tensor
    dt: float = 1.0               # fs
    T: float = 300.0              # K
    taut: float = 20.0            # fs  (same meaning as Berendsen τ_T)
    remove_com: bool = True       # subtract 3 DOF if true

    def __post_init__(self):
        self._forces = None

    def set_temperature(self, T: float):
        self.T = float(T)

    def _rescale_factor(self, K_now: torch.Tensor) -> torch.Tensor:
        """Bussi 2007 Eq. A7 — returns α s.t. v ← α v samples canonical KE."""
        Nf = 3 * self.masses.numel() - (3 if self.remove_com else 0)
        K_bar = 0.5 * Nf * KB_EV * self.T                      # target KE (eV)
        # convert K_now from "integrator-internal" kinetic units:
        # K_now already in eV since our kinetic_energy() returns eV.
        c = math.exp(-self.dt / self.taut)
        device, dtype = K_now.device, K_now.dtype
        # one N(0,1) and one chi²(Nf-1) draw
        R1 = torch.randn((), device=device, dtype=dtype)
        # chi²(k) sampled as sum of k squared normals — cheap for small/med N;
        # for huge N switch to torch.distributions.Gamma.
        k = max(Nf - 1, 1)
        S = (torch.randn((k,), device=device, dtype=dtype) ** 2).sum()
        factor = K_bar / (Nf * K_now.clamp_min(1e-12))
        alpha_sq = (
            c
            + (1.0 - c) * (S + R1 * R1) * factor
            + 2.0 * R1 * torch.sqrt(c * (1.0 - c) * factor)
        )
        # sign: α has the sign of (R1 + sqrt(c*Nf*K/(factor*(1-c)))) per BDP;
        # in practice α² > 0 and we take positive root.
        return torch.sqrt(alpha_sq.clamp_min(0.0))

    def step(self, positions, velocities):
        if self._forces is None:
            _, self._forces = self.forces_fn(positions)
        m = self.masses.unsqueeze(-1)
        a = EV_AMU_A_PER_FS2 * self._forces / m
        velocities = velocities + 0.5 * self.dt * a
        positions = positions + self.dt * velocities
        _, new_forces = self.forces_fn(positions)
        a_new = EV_AMU_A_PER_FS2 * new_forces / m
        velocities = velocities + 0.5 * self.dt * a_new
        # Stochastic canonical rescale
        K_now = kinetic_energy(self.masses, velocities)
        alpha = self._rescale_factor(K_now)
        velocities = velocities * alpha
        self._forces = new_forces
        return positions, velocities


# ---------------------------------------------------------------------------
# NVT: Nosé-Hoover chain (Martyna-Tobias-Klein integrator)
# Chains of length M ≥ 3 fix the non-ergodicity of a single Nosé-Hoover.
# Reference: Martyna, Tobias & Klein, J. Chem. Phys. 101, 4177 (1994);
#            Tuckerman, "Statistical Mechanics: Theory and Molecular
#            Simulation", Ch. 4.
# ---------------------------------------------------------------------------
@dataclass
class NVTNoseHooverChain:
    forces_fn: ForcesFn
    masses: torch.Tensor
    dt: float = 1.0               # fs
    T: float = 300.0              # K
    taut: float = 20.0            # fs — chain relaxation time; Q1 = Nf·kT·τ²
    chain_length: int = 3         # M; use ≥3 for ergodicity
    remove_com: bool = True

    def __post_init__(self):
        device, dtype = self.masses.device, self.masses.dtype
        M = self.chain_length
        self.v_xi = torch.zeros(M, device=device, dtype=dtype)
        self._forces = None
        self._set_masses()

    def _set_masses(self):
        device, dtype = self.masses.device, self.masses.dtype
        Nf = 3 * self.masses.numel() - (3 if self.remove_com else 0)
        kT = KB_EV * self.T
        self._Nf = Nf
        self._kT = kT
        tau2 = self.taut ** 2
        Q = torch.full((self.chain_length,), kT * tau2, device=device, dtype=dtype)
        Q[0] = Nf * kT * tau2
        self._Q = Q

    def set_temperature(self, T: float):
        self.T = float(T)
        self._set_masses()    # rescale Q so relaxation stays ~τ

    def _apply_chain_half(self, velocities):
        """Apply half a chain step (dt_half = dt/2 of MD timestep)."""
        dt_half = self.dt / 2
        dthalf = dt_half / 2
        dtqtr = dt_half / 4
        Q, v_xi = self._Q, self.v_xi
        M = self.chain_length
        K = kinetic_energy(self.masses, velocities)       # eV
        # --- reverse pass (top of chain down) ---
        G_M = (Q[M-2] * v_xi[M-2] ** 2 - self._kT) / Q[M-1] if M >= 2 else \
              (2 * K - self._Nf * self._kT) / Q[0]
        v_xi[M-1] = v_xi[M-1] + G_M * dthalf
        for i in range(M - 2, -1, -1):
            factor = torch.exp(-dtqtr * v_xi[i + 1])
            v_xi[i] = v_xi[i] * factor
            if i == 0:
                G_i = (2 * K - self._Nf * self._kT) / Q[0]
            else:
                G_i = (Q[i-1] * v_xi[i-1] ** 2 - self._kT) / Q[i]
            v_xi[i] = v_xi[i] + G_i * dthalf
            v_xi[i] = v_xi[i] * factor
        # --- rescale particle velocities ---
        scale = torch.exp(-dt_half * v_xi[0])
        velocities = velocities * scale
        K = K * scale * scale
        # --- forward pass (bottom up) ---
        for i in range(0, M - 1):
            factor = torch.exp(-dtqtr * v_xi[i + 1])
            v_xi[i] = v_xi[i] * factor
            if i == 0:
                G_i = (2 * K - self._Nf * self._kT) / Q[0]
            else:
                G_i = (Q[i-1] * v_xi[i-1] ** 2 - self._kT) / Q[i]
            v_xi[i] = v_xi[i] + G_i * dthalf
            v_xi[i] = v_xi[i] * factor
        if M >= 2:
            G_M = (Q[M-2] * v_xi[M-2] ** 2 - self._kT) / Q[M-1]
            v_xi[M-1] = v_xi[M-1] + G_M * dthalf
        return velocities

    def step(self, positions, velocities):
        if self._forces is None:
            _, self._forces = self.forces_fn(positions)
        m = self.masses.unsqueeze(-1)
        # Chain half-step
        velocities = self._apply_chain_half(velocities)
        # VV: kick-drift-kick
        a = EV_AMU_A_PER_FS2 * self._forces / m
        velocities = velocities + 0.5 * self.dt * a
        positions = positions + self.dt * velocities
        _, new_forces = self.forces_fn(positions)
        a_new = EV_AMU_A_PER_FS2 * new_forces / m
        velocities = velocities + 0.5 * self.dt * a_new
        # Chain half-step
        velocities = self._apply_chain_half(velocities)
        self._forces = new_forces
        return positions, velocities


# ---------------------------------------------------------------------------
# Driver loop
# ---------------------------------------------------------------------------
def run(
    integrator,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    nsteps: int,
    log_every: int = 100,
    callback: Optional[Callable[[int, dict], None]] = None,
):
    positions = positions.clone()
    velocities = velocities.clone()
    hist = []
    t0 = time.time()
    for i in range(nsteps):
        positions, velocities = integrator.step(positions, velocities)
        if (i % log_every) == 0 or i == nsteps - 1:
            ke = kinetic_energy(integrator.masses, velocities).item()
            T = instantaneous_temperature(integrator.masses, velocities).item()
            pe = integrator.forces_fn(positions)[0].item() \
                 if hasattr(integrator, "_forces") and integrator._forces is None \
                 else None
            row = {"step": i, "time_fs": i * integrator.dt, "T_K": T,
                   "KE_eV": ke, "wall_s": time.time() - t0}
            hist.append(row)
            if callback is not None:
                callback(i, row)
            else:
                print(f"step {i:6d}  t={row['time_fs']:8.1f} fs  "
                      f"T={T:7.1f} K  KE={ke:10.4f} eV  wall={row['wall_s']:6.1f}s")
    return positions, velocities, hist
