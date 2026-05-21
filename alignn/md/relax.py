"""FIRE structural relaxation (Bitzek et al., PRL 97, 170201, 2006).

ASE-compatible defaults. Drop-in for `ase.optimize.FIRE` when you want
to stay on-device and avoid the ASE roundtrip every step.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import time
import torch

from alignn.md.integrators import ForcesFn, EV_AMU_A_PER_FS2


@dataclass
class FIRE:
    forces_fn: ForcesFn
    masses: torch.Tensor              # (N,)  amu
    dt: float = 0.1                   # fs  (initial timestep)
    dt_max: float = 1.0               # fs
    max_step: float = 0.2             # Å  (per-atom cap on Δx)
    N_min: int = 5
    f_inc: float = 1.1
    f_dec: float = 0.5
    alpha_start: float = 0.1
    f_alpha: float = 0.99

    def run(
        self,
        positions: torch.Tensor,
        fmax: float = 0.05,           # eV/Å convergence criterion
        max_steps: int = 500,
        log_every: int = 10,
        callback: Optional[Callable[[int, dict], None]] = None,
    ):
        positions = positions.clone()
        velocities = torch.zeros_like(positions)
        m = self.masses.unsqueeze(-1)
        dt = self.dt
        alpha = self.alpha_start
        n_pos = 0
        history = []
        t0 = time.time()

        # initial force
        energy, forces = self.forces_fn(positions)
        for step in range(max_steps):
            fnorm_max = forces.norm(dim=-1).max().item()
            row = {
                "step": step, "E_eV": float(energy),
                "fmax_eV_A": fnorm_max, "dt_fs": dt, "alpha": alpha,
                "wall_s": time.time() - t0,
            }
            if step % log_every == 0 or fnorm_max < fmax:
                history.append(row)
                if callback is not None:
                    callback(step, row)
                else:
                    print(f"FIRE step={step:4d}  E={row['E_eV']:12.4f} eV  "
                          f"|F|max={fnorm_max:9.4f} eV/Å  dt={dt:5.3f} fs  "
                          f"α={alpha:5.3f}")
            if fnorm_max < fmax:
                print(
                    f"converged at step {step}: "
                    f"|F|max = {fnorm_max:.5f} < {fmax}"
                )
                break

            # FIRE velocity mix: v ← (1-α) v + α |v| F̂
            f_hat = forces / (forces.norm() + 1e-30)
            v_norm = velocities.norm()
            velocities = (1.0 - alpha) * velocities + alpha * v_norm * f_hat

            # Power P = F·v (sum over all atoms)
            P = (forces * velocities).sum()
            if P.item() > 0.0:
                n_pos += 1
                if n_pos > self.N_min:
                    dt = min(dt * self.f_inc, self.dt_max)
                    alpha = alpha * self.f_alpha
            else:
                n_pos = 0
                dt = dt * self.f_dec
                alpha = self.alpha_start
                velocities = torch.zeros_like(velocities)

            # Velocity Verlet step (with max_step clipping)
            a = EV_AMU_A_PER_FS2 * forces / m
            velocities = velocities + 0.5 * dt * a
            dx = dt * velocities
            # Per-atom displacement cap
            dx_norm = dx.norm(dim=-1, keepdim=True)
            scale = torch.where(
                dx_norm > self.max_step,
                self.max_step / dx_norm.clamp_min(1e-30),
                torch.ones_like(dx_norm),
            )
            dx = dx * scale
            positions = positions + dx
            energy, new_forces = self.forces_fn(positions)
            a_new = EV_AMU_A_PER_FS2 * new_forces / m
            velocities = velocities + 0.5 * dt * a_new
            forces = new_forces
        return positions, history
