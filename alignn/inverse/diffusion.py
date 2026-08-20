"""Noise processes for ALIGNN-CSP.

Two coupled processes run on the same timestep index ``t``:

**Lattice** — a lattice matrix ``L`` (rows are lattice vectors, so
``cart = frac @ L``) is only defined up to a global rotation: under
``x -> x R`` the matrix becomes ``L R``, leaving the metric tensor
``G = L L^T`` untouched.  We therefore diffuse the rotation-invariant
symmetric part ``S = G^{1/2}``, for which ``L = S`` is a valid canonical
representative of the same crystal.  ``S`` is symmetric positive-definite, so
we work with its matrix logarithm: that is an unconstrained point in R^6, and
``expm`` maps *any* R^6 vector back to a non-degenerate cell.  Noise can never
produce an invalid lattice.  A ``N^{-1/3}`` factor removes the trivial
dependence on cell size.  Standard DDPM (cosine schedule) runs on the
standardised 6-vector.

**Fractional coordinates** — these live on the torus [0,1)^3, so we use the
score-based wrapped-normal process of DiffCSP: ``f_t = w(f_0 + sigma_t * z)``
with a geometric sigma schedule.  The network predicts the *sigma-scaled*
score, which tends to ``-z`` as sigma -> 0 and to ``0`` as the distribution
approaches uniform, and so is well conditioned at both ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

# Index of the 6 independent entries of a symmetric 3x3 matrix.
_SYM_I = (0, 1, 2, 0, 0, 1)
_SYM_J = (0, 1, 2, 1, 2, 2)
_OFFDIAG_SCALE = math.sqrt(2.0)


def sym_to_vec6(m: torch.Tensor) -> torch.Tensor:
    """Flatten symmetric (..., 3, 3) to (..., 6), preserving Frobenius norm."""
    diag = torch.stack([m[..., i, i] for i in range(3)], dim=-1)
    off = (
        torch.stack([m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]], dim=-1)
        * _OFFDIAG_SCALE
    )
    return torch.cat([diag, off], dim=-1)


def vec6_to_sym(v: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`sym_to_vec6`."""
    out = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    for k in range(3):
        out[..., k, k] = v[..., k]
    off = v[..., 3:] / _OFFDIAG_SCALE
    out[..., 0, 1] = out[..., 1, 0] = off[..., 0]
    out[..., 0, 2] = out[..., 2, 0] = off[..., 1]
    out[..., 1, 2] = out[..., 2, 1] = off[..., 2]
    return out


def _sym_funm(m: torch.Tensor, fn) -> torch.Tensor:
    """Apply a scalar function to the eigenvalues of a symmetric matrix."""
    # Symmetrise first: guards against drift from upstream float error.
    m = 0.5 * (m + m.transpose(-1, -2))
    if not torch.isfinite(m).all():
        raise FloatingPointError(
            "non-finite symmetric matrix reached _sym_funm; the lattice "
            "state has diverged"
        )
    md = m.double()
    try:
        evals, evecs = torch.linalg.eigh(md)
    except torch._C._LinAlgError:
        # cuSOLVER's batched Jacobi solver occasionally fails to converge on
        # near-degenerate 3x3 blocks. These matrices are tiny, so falling back
        # to the CPU LAPACK path costs almost nothing and always succeeds.
        evals, evecs = torch.linalg.eigh(md.cpu())
        evals, evecs = evals.to(md.device), evecs.to(md.device)
    out = evecs @ torch.diag_embed(fn(evals)) @ evecs.transpose(-1, -2)
    return out.to(m.dtype)


def lattice_to_vec6(
    lattice: torch.Tensor, natoms: torch.Tensor
) -> torch.Tensor:
    """Map lattice matrices (B, 3, 3) to log-space 6-vectors (B, 6)."""
    scale = natoms.to(lattice.dtype).clamp_min(1.0) ** (1.0 / 3.0)
    lat = lattice / scale.view(-1, 1, 1)
    gram = lat @ lat.transpose(-1, -2)
    # S = G^{1/2}; log S = 0.5 * log G, done in one eigendecomposition.
    log_s = _sym_funm(gram, lambda e: 0.5 * torch.log(e.clamp_min(1e-8)))
    return sym_to_vec6(log_s)


def vec6_to_lattice(vec: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`lattice_to_vec6`; always yields a valid cell."""
    log_s = vec6_to_sym(vec)
    s = _sym_funm(log_s, torch.exp)
    scale = natoms.to(vec.dtype).clamp_min(1.0) ** (1.0 / 3.0)
    return s * scale.view(-1, 1, 1)


def wrap_frac(f: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coordinates into [0, 1)."""
    return f - torch.floor(f)


def wrap_diff(d: torch.Tensor) -> torch.Tensor:
    """Wrap a fractional difference into [-0.5, 0.5)."""
    return d - torch.round(d)


def wrapped_normal_score(
    delta: torch.Tensor, sigma: torch.Tensor, n_images: int = 5
) -> torch.Tensor:
    """Sigma-scaled score of the wrapped normal, ``sigma * d/d(delta) log p``.

    ``delta`` is the (already wrapped) displacement, ``sigma`` broadcasts
    against it.  Computed as a softmax-weighted mean over periodic images, so
    it stays stable for sigma spanning several orders of magnitude.
    """
    d = wrap_diff(delta)
    images = torch.arange(
        -n_images, n_images + 1, device=d.device, dtype=d.dtype
    )
    # (..., 2n+1) displacement to each periodic image
    u = d.unsqueeze(-1) + images
    logits = -0.5 * (u / sigma.unsqueeze(-1)) ** 2
    weights = torch.softmax(logits, dim=-1)
    # score = -E[u] / sigma^2, so sigma * score = -E[u] / sigma
    return -(weights * u).sum(dim=-1) / sigma


@dataclass
class DiffusionSchedule:
    """Coupled DDPM (lattice) and VE score (coordinates) schedules."""

    num_steps: int = 1000
    sigma_min: float = 0.005
    sigma_max: float = 0.5
    cosine_s: float = 0.008

    def __post_init__(self):
        t = torch.arange(self.num_steps + 1, dtype=torch.float64)
        # Cosine alpha-bar schedule (Nichol & Dhariwal).
        f = (
            torch.cos(
                ((t / self.num_steps) + self.cosine_s)
                / (1.0 + self.cosine_s)
                * math.pi
                * 0.5
            )
            ** 2
        )
        alpha_bar = (f / f[0]).clamp(1e-6, 1.0)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, 0.999)

        self.alpha_bar = alpha_bar.float()  # (T+1,), index 0 == t=0
        self.betas = betas.float()  # (T,), index i == step i+1
        self.alphas = 1.0 - self.betas

        # Geometric sigma ladder for the torus process; index 0 is unused.
        sigmas = torch.exp(
            torch.linspace(
                math.log(self.sigma_min),
                math.log(self.sigma_max),
                self.num_steps,
                dtype=torch.float64,
            )
        ).float()
        self.sigmas = torch.cat([torch.zeros(1), sigmas])  # (T+1,)

    def to(self, device):
        for name in ("alpha_bar", "betas", "alphas", "sigmas"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ── forward (noising) ────────────────────────────────────────────────
    def noise_lattice(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, eps)`` for the lattice 6-vector."""
        ab = self.alpha_bar[t].view(-1, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
        return x_t, eps

    def noise_frac(
        self, f0: torch.Tensor, t_node: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(f_t, target)`` where target is the sigma-scaled score."""
        sigma = self.sigmas[t_node].view(-1, 1)
        z = torch.randn_like(f0)
        f_t = wrap_frac(f0 + sigma * z)
        target = wrapped_normal_score(f_t - f0, sigma)
        return f_t, target
