"""Mechanism-level evaluation for generated crystals.

The AtomBench pipeline already scores match rate, RMSD, ccRMSD, lattice MAE
and KLD, and none of that changes here.  What it cannot say is *why* one model
is better, and that is what an angular-diffusion experiment has to answer.
Two extra measurements are provided:

**Bond-angle distributions.**  Compare the angles a model actually generates
against the angles of held-out real structures.  This is FoldingDiff's own
diagnostic (Wu et al., Nat. Commun. 15, 1059, 2024): a generative model with
an explicit angular channel should reproduce the natural angular distribution,
and a model that merely places atoms plausibly on average need not.  Reported
as KL, Jensen-Shannon and 1-D Wasserstein distance between normalised
histograms, all on the same binning.

**Relaxation displacement.**  How far a generated structure has to move to
reach the nearest local minimum of the force field.  MatterGen (Zeni et al.,
Nature 639, 624, 2025) evaluates generated structures by how close they sit to
their relaxed counterparts; if explicit angular denoising produces locally
coherent geometry, its samples should need less repair.  Reported as the
translation-corrected Cartesian RMSD between the sample and its relaxed self,
plus the fractional volume change and the energy drop.

Nothing here is used to select a model — the suite is fixed before the
ablations are run, exactly so that a favourable metric cannot be chosen after
seeing the results.

Everything is plain PyTorch, on the same neighbour list the model itself uses,
so the angles being scored are the angles ALIGNN would see.  ``jarvis`` enters
only to parse POSCARs and the ASE force field only through the existing
:mod:`alignn.inverse.relax_rank`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

__all__ = [
    "bond_angles_deg",
    "collect_bond_angles",
    "compare_angle_distributions",
    "relaxation_displacement",
    "structures_from_benchmark_csv",
]

#: Neighbour cutoff and count used for every angle measurement.  These match
#: the pure-torch graph builder's own three-body defaults.
DEFAULT_ANGLE_CUTOFF = 3.5
DEFAULT_MAX_NEIGHBORS = 12
DEFAULT_BINS = 180

_EPS = 1e-12


def _atoms_to_tensors(atoms, dtype=torch.float64):
    """Cartesian positions and lattice of a jarvis ``Atoms`` as tensors."""
    return (
        torch.tensor(atoms.cart_coords, dtype=dtype),
        torch.tensor(atoms.lattice_mat, dtype=dtype),
    )


def _same_source_pairs(src: torch.Tensor, num_nodes: int):
    """Unordered pairs of edges sharing a source node.

    Same construction the line-graph builder uses: group the edges by their
    shared node, expand each group to all ordered pairs, then keep one of
    each two.  Returns indices into the edge list.
    """
    n_edges = int(src.shape[0])
    device = src.device
    if n_edges == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    order = torch.argsort(src, stable=True)
    src_sorted = src[order]
    counts = torch.bincount(src_sorted, minlength=num_nodes)
    starts = torch.cumsum(counts, 0) - counts
    # Every edge pairs with each edge in its own group.
    per_edge = counts[src_sorted]
    total = int(per_edge.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    positions = torch.arange(n_edges, device=device)
    left = torch.repeat_interleave(positions, per_edge)
    cum = torch.cumsum(per_edge, 0)
    row_start = cum - per_edge
    offsets = torch.arange(total, device=device) - torch.repeat_interleave(
        row_start, per_edge
    )
    right = torch.repeat_interleave(starts[src_sorted], per_edge) + offsets
    keep = left < right
    return order[left[keep]], order[right[keep]]


def bond_angles_deg(
    atoms,
    cutoff: float = DEFAULT_ANGLE_CUTOFF,
    max_neighbors: Optional[int] = DEFAULT_MAX_NEIGHBORS,
) -> torch.Tensor:
    """Every bond angle in one structure, in degrees.

    An angle is formed by each unordered pair of neighbours of a central atom,
    where "neighbour" is a periodic image within ``cutoff`` (truncated to the
    ``max_neighbors`` closest, as the graph builder does).
    """
    from alignn.torch_graph_builder import torch_neighbor_list

    positions, lattice = _atoms_to_tensors(atoms)
    src, _dst, _shift, r = torch_neighbor_list(
        positions,
        lattice,
        cutoff,
        max_neighbors=max_neighbors,
        use_matscipy_topology=False,
    )
    left, right = _same_source_pairs(src, int(positions.shape[0]))
    if left.numel() == 0:
        return torch.zeros(0, dtype=positions.dtype)
    # r points away from the shared atom, so the interior angle is the plain
    # angle between the two outgoing vectors.
    a = r[left]
    b = r[right]
    cos = (a * b).sum(-1) / (
        a.norm(dim=-1).clamp_min(_EPS) * b.norm(dim=-1).clamp_min(_EPS)
    )
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def collect_bond_angles(
    structures: Sequence,
    cutoff: float = DEFAULT_ANGLE_CUTOFF,
    max_neighbors: Optional[int] = DEFAULT_MAX_NEIGHBORS,
) -> torch.Tensor:
    """Bond angles pooled over a list of structures, in degrees."""
    parts = [
        bond_angles_deg(a, cutoff=cutoff, max_neighbors=max_neighbors)
        for a in structures
    ]
    parts = [p for p in parts if p.numel()]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _density(angles: torch.Tensor, bins: int) -> torch.Tensor:
    """Normalised histogram over [0, 180] degrees."""
    x = torch.as_tensor(angles, dtype=torch.float64).flatten()
    if x.numel() == 0:
        return torch.zeros(bins, dtype=torch.float64)
    # Bucket by index rather than torch.histc so that the closed right edge
    # (a perfectly straight 180-degree angle) lands in the last bin.
    idx = (x.clamp(0.0, 180.0) * (bins / 180.0)).long().clamp(0, bins - 1)
    counts = torch.bincount(idx, minlength=bins).to(torch.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def compare_angle_distributions(
    generated: torch.Tensor,
    reference: torch.Tensor,
    bins: int = DEFAULT_BINS,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Distances between two pooled bond-angle distributions.

    ``wasserstein_deg`` is exact for a 1-D histogram (the integral of the
    absolute CDF difference) and is in degrees, so it reads directly as "the
    generated angles are off by this much on average".
    """
    p = _density(generated, bins)
    q = _density(reference, bins)
    ps, qs = p + eps, q + eps
    ps, qs = ps / ps.sum(), qs / qs.sum()
    kl = float((ps * (ps / qs).log()).sum())
    m = 0.5 * (ps + qs)
    js = float(
        0.5 * (ps * (ps / m).log()).sum() + 0.5 * (qs * (qs / m).log()).sum()
    )
    width = 180.0 / bins
    emd = float((p.cumsum(0) - q.cumsum(0)).abs().sum() * width)
    return {
        "kl": kl,
        "js": js,
        "wasserstein_deg": emd,
        "n_generated": int(torch.as_tensor(generated).numel()),
        "n_reference": int(torch.as_tensor(reference).numel()),
        "bins": bins,
    }


def _min_image_displacement(before, after) -> torch.Tensor:
    """Cartesian displacement per atom, minimum-image and drift-corrected.

    A relaxation is free to translate the whole cell, and the benchmark's own
    metrics quotient that out, so the mean displacement is removed before the
    RMSD is taken.
    """
    f0 = torch.tensor(before.frac_coords, dtype=torch.float64)
    f1 = torch.tensor(after.frac_coords, dtype=torch.float64)
    df = f1 - f0
    df = df - df.round()
    df = df - df.mean(dim=0, keepdim=True)
    df = df - df.round()
    return df @ torch.tensor(after.lattice_mat, dtype=torch.float64)


def relaxation_displacement(
    before,
    after,
    energy_before: Optional[float] = None,
    energy_after: Optional[float] = None,
) -> Dict[str, float]:
    """How far one generated structure moved to reach its local minimum."""
    d = _min_image_displacement(before, after)
    norms = d.norm(dim=-1)
    v0 = float(
        torch.linalg.det(
            torch.tensor(before.lattice_mat, dtype=torch.float64)
        ).abs()
    )
    v1 = float(
        torch.linalg.det(
            torch.tensor(after.lattice_mat, dtype=torch.float64)
        ).abs()
    )
    out = {
        "rmsd_angstrom": float(norms.pow(2).mean().sqrt()),
        "max_displacement_angstrom": (
            float(norms.max()) if norms.numel() else 0.0
        ),
        "volume_change_frac": (v1 - v0) / v0 if v0 else float("nan"),
    }
    if energy_before is not None and energy_after is not None:
        drop = float(energy_before - energy_after)
        out["energy_drop_ev_per_atom"] = (
            drop if math.isfinite(drop) else float("nan")
        )
    return out


def structures_from_benchmark_csv(path, column: str = "prediction") -> List:
    """Read one POSCAR column of an AtomBench CSV into jarvis ``Atoms``.

    The CSV written by ``scripts/atombench/generate_benchmark.py`` holds both
    the generated structure (``prediction``) and the held-out reference
    (``target``), so one file supplies both sides of the angle comparison.
    """
    import csv

    from jarvis.core.atoms import Atoms

    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            text = row[column].replace("\\n", "\n")
            out.append(Atoms.from_poscar(text))
    return out
