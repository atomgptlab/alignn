"""Angular geometry and continuously-weighted topology for ALIGNN-CSP.

Two pieces of machinery live here, and they are deliberately independent so
that the ablations in :mod:`alignn.inverse.ablations` can switch one on
without the other.

**Explicit angular denoising.**  ALIGNN already carries bond angles on its
line graph, but only as an *input feature*.  Here they also become a
*denoising target*: the network predicts, per triplet, the angular
displacement that the forward process introduced.  The stochastic process and
the loss are ported from FoldingDiff (Wu et al., Nat. Commun. 15, 1059, 2024,
doi:10.1038/s41467-024-45051-2), which runs DDPM-style corruption and
denoising directly on protein bond and dihedral angles with wrapped angular
noise and a wrapped smooth-L1 objective.  Torsional Diffusion (Jing et al.,
NeurIPS 2022) is the general statement that a diffusion process can be defined
on an angular configuration space.

One difference from FoldingDiff has to be stated plainly, because it is the
main methodological caveat of this extension.  In a protein backbone the
internal-coordinate list is *fixed*: residue i always has the same three
angles, so a genuinely persistent state ``theta_t`` can be diffused
independently of anything else.  In a crystal being denoised from noise the
triplet set is not fixed — it is a function of the coordinates, and it changes
as they move.  There is therefore no persistent ``theta_t`` to diffuse.  What
is implemented instead is the closest well-defined thing: the angular
denoising *target* is computed on the triplet representation that exists at
the current step,

    delta_ijk = wrap(theta_ijk(f_t, L_t) - theta_ijk(f_0, L_0)),

with both angles evaluated on the *same* periodic-image identity
``(i, j, k, n_ji, n_jk)`` so that the difference measures the corruption of
one specific triplet rather than a change of neighbour.  Angles are still an
explicit denoising channel with their own head and their own loss; they are
not an independently-noised variable.  Section 3 of the design brief asks for
exactly this fallback, and asks that the distinction be documented rather than
papered over with an invented process.

**Continuously-weighted topology.**  During reverse diffusion a hard
neighbour-rank criterion for "does this triplet exist" is unjustified: at
large ``t`` the coordinates are close to uniform, so neighbour ranks swap
constantly and the line graph jumps discontinuously.  Instead every pair
carries a smooth relevance

    s_ij = u(r_ij ; r_c),

where ``u`` is the polynomial cutoff envelope introduced by DimeNet
(Gasteiger, Gross & Gunnemann, ICLR 2020, arXiv:2003.03123), whose value and
first two derivatives vanish at ``r_c``.  That envelope is already implemented
in this repository as
:class:`alignn.models.alignn_atomwise_pure_smooth.CutoffPolynomial`, so it is
reused rather than re-derived.  A triplet inherits the product of its two
constituent relevances,

    s_ijk = s_ji * s_jk,

which is the ReaxFF treatment of valence angles (van Duin et al.,
J. Phys. Chem. A 105, 9396, 2001, doi:10.1021/jp004368u): bond orders vary
continuously with distance and an angular term switches off smoothly as
either of its bonds dissociates.  Because ``s`` is exactly zero at and beyond
``r_c``, restricting the sparse line graph to pairs inside ``r_c`` removes
only terms that were already contributing nothing — a triplet can enter or
leave the computational graph without any finite jump in the messages.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from alignn.models.alignn_atomwise_pure_smooth import CutoffPolynomial
from alignn.torch_graph_builder import torch_bond_cosines

__all__ = [
    "CutoffPolynomial",
    "bond_angle",
    "wrap_angle",
    "pair_relevance",
    "triplet_relevance",
    "edge_vectors",
    "angular_denoising_target",
    "angle_denoising_loss",
    "TWO_PI",
]

TWO_PI = 2.0 * math.pi

# acos is not differentiable at +-1; the clamp keeps the gradient finite for
# collinear triplets, which are common in a crystal (i -> j -> i back-tracking
# triplets are cos = -1 exactly).
_COS_EPS = 1.0e-7


def bond_angle(r_ij: torch.Tensor, r_jk: torch.Tensor) -> torch.Tensor:
    """Bond angle at the shared atom ``j`` of a triplet, in radians.

    Uses ALIGNN's own cosine convention (:func:`torch_bond_cosines`), so the
    angle is the interior angle at ``j`` and lies in ``[0, pi]``.
    """
    cos = torch_bond_cosines(r_ij, r_jk)
    return torch.acos(cos.clamp(-1.0 + _COS_EPS, 1.0 - _COS_EPS))


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap an angular difference into ``[-pi, pi)``.

    FoldingDiff's forward process and loss are both defined modulo ``2 pi``.
    Bond angles themselves live in ``[0, pi]``, so a difference of two of them
    is already inside ``[-pi, pi]`` and this is a no-op up to the boundary
    case; it is applied anyway so that the objective is the wrapped one by
    construction rather than by an argument about ranges, and so that the same
    helper can serve a periodic angular variable if one is ever added.
    """
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def pair_relevance(
    dist: torch.Tensor, envelope: CutoffPolynomial
) -> torch.Tensor:
    """Smooth per-pair relevance ``s_ij = u(r_ij; r_c)`` in ``[0, 1]``.

    ``u`` is the DimeNet envelope: ``u(0) = 1`` and ``u`` together with its
    first two derivatives vanishes at ``r_c``.
    """
    return envelope(dist)


def triplet_relevance(
    s_edge: torch.Tensor, lg_src: torch.Tensor, lg_dst: torch.Tensor
) -> torch.Tensor:
    """ReaxFF-style product gate ``s_ijk = s_ji * s_jk`` for each triplet."""
    return s_edge[lg_src] * s_edge[lg_dst]


def edge_vectors(
    frac: torch.Tensor,
    lattice: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_graph_id: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Cartesian edge vectors for a *given* set of periodic images.

    ``image`` is the integer cell offset ``n`` that the minimum-image search
    settled on at the noised geometry, so that ``delta_f = f[dst] - f[src] +
    n``.  Re-using the same ``n`` on a different (clean) structure is what
    makes the angular target a corruption of one fixed triplet identity rather
    than a comparison between two different neighbours.
    """
    df = frac[dst] - frac[src] + image
    return torch.einsum("ei,eij->ej", df, lattice[edge_graph_id])


def angular_denoising_target(
    angle_t: torch.Tensor,
    frac0: torch.Tensor,
    lattice0: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_graph_id: torch.Tensor,
    image: torch.Tensor,
    lg_src: torch.Tensor,
    lg_dst: torch.Tensor,
) -> torch.Tensor:
    """Angular displacement the forward process applied to each triplet.

    Returns ``wrap(theta_t - theta_0)``, the angular analogue of the noise
    ``eps`` that FoldingDiff's network predicts.
    """
    r0 = edge_vectors(frac0, lattice0, src, dst, edge_graph_id, image)
    theta0 = bond_angle(r0[lg_src], r0[lg_dst])
    return wrap_angle(angle_t - theta0)


def angle_denoising_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    beta: float = 0.1 * math.pi,
) -> torch.Tensor:
    """Relevance-weighted wrapped smooth-L1 loss on the angular channel.

    The functional form — smooth L1 of the *wrapped* residual, with
    ``beta = 0.1 pi`` — is FoldingDiff's angular objective.  The weighting by
    ``s_ijk`` is what makes the loss continuous when a triplet enters or
    leaves the sparse line graph: a triplet at the cutoff has zero weight, so
    it contributes nothing on either side of the boundary.
    """
    if pred.numel() == 0:
        return pred.new_zeros(())
    d = wrap_angle(pred - target)
    per_triplet = torch.nn.functional.smooth_l1_loss(
        d, torch.zeros_like(d), beta=beta, reduction="none"
    )
    if weight is None:
        return per_triplet.mean()
    return (weight * per_triplet).sum() / weight.sum().clamp_min(1e-8)


def angle_histogram(
    angles_deg, bins: int = 180, lo: float = 0.0, hi: float = 180.0
) -> Tuple:
    """Normalised histogram of bond angles in degrees.

    Kept here so the evaluation code and the tests share one definition.
    """
    import numpy as np

    counts, edges = np.histogram(
        np.asarray(angles_deg, dtype=float), bins=bins, range=(lo, hi)
    )
    total = counts.sum()
    density = counts / total if total else counts.astype(float)
    return density, edges
