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
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "bond_angles_deg",
    "collect_bond_angles",
    "compare_angle_distributions",
    "relaxation_displacement",
    "structures_from_benchmark_csv",
]

#: Neighbour cutoff and count used for every angle measurement.  These match
#: the pure-torch graph builder's own three-body defaults so that the angles
#: being scored are the ones ALIGNN would see.
DEFAULT_ANGLE_CUTOFF = 3.5
DEFAULT_MAX_NEIGHBORS = 12
DEFAULT_BINS = 180


def bond_angles_deg(
    atoms,
    cutoff: float = DEFAULT_ANGLE_CUTOFF,
    max_neighbors: Optional[int] = DEFAULT_MAX_NEIGHBORS,
) -> np.ndarray:
    """Every bond angle in one structure, in degrees.

    An angle is formed by each unordered pair of neighbours of a central atom,
    where "neighbour" is a periodic image within ``cutoff`` (truncated to the
    ``max_neighbors`` closest, as the graph builder does).
    """
    from alignn.torch_graph_builder import torch_neighbor_list

    positions = torch.tensor(
        np.asarray(atoms.cart_coords, dtype=float), dtype=torch.float64
    )
    lattice = torch.tensor(
        np.asarray(atoms.lattice_mat, dtype=float), dtype=torch.float64
    )
    src, _dst, _shift, r = torch_neighbor_list(
        positions,
        lattice,
        cutoff,
        max_neighbors=max_neighbors,
        use_matscipy_topology=False,
    )
    src_np = src.numpy()
    # r points from src outward, so the angle at the central atom is taken
    # between two of its outgoing vectors directly.
    vec = r.numpy()
    out: List[float] = []
    order = np.argsort(src_np, kind="stable")
    src_sorted, vec_sorted = src_np[order], vec[order]
    bounds = np.searchsorted(
        src_sorted, np.arange(int(positions.shape[0]) + 1)
    )
    for j in range(int(positions.shape[0])):
        v = vec_sorted[bounds[j] : bounds[j + 1]]
        if len(v) < 2:
            continue
        unit = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
        cos = unit @ unit.T
        iu = np.triu_indices(len(v), k=1)
        out.append(np.degrees(np.arccos(np.clip(cos[iu], -1.0, 1.0))))
    if not out:
        return np.zeros(0)
    return np.concatenate(out)


def collect_bond_angles(
    structures: Sequence,
    cutoff: float = DEFAULT_ANGLE_CUTOFF,
    max_neighbors: Optional[int] = DEFAULT_MAX_NEIGHBORS,
) -> np.ndarray:
    """Bond angles pooled over a list of structures, in degrees."""
    parts = [
        bond_angles_deg(a, cutoff=cutoff, max_neighbors=max_neighbors)
        for a in structures
    ]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0)


def _density(angles: np.ndarray, bins: int) -> np.ndarray:
    counts, _ = np.histogram(angles, bins=bins, range=(0.0, 180.0))
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def compare_angle_distributions(
    generated: np.ndarray,
    reference: np.ndarray,
    bins: int = DEFAULT_BINS,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Distances between two pooled bond-angle distributions.

    ``wasserstein`` is exact for a 1-D histogram (the integral of the absolute
    CDF difference) and is reported in degrees, so it reads directly as "the
    generated angles are off by this much on average".
    """
    p = _density(np.asarray(generated), bins)
    q = _density(np.asarray(reference), bins)
    ps, qs = p + eps, q + eps
    ps, qs = ps / ps.sum(), qs / qs.sum()
    kl = float((ps * np.log(ps / qs)).sum())
    m = 0.5 * (ps + qs)
    js = float(
        0.5 * (ps * np.log(ps / m)).sum() + 0.5 * (qs * np.log(qs / m)).sum()
    )
    width = 180.0 / bins
    emd = float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() * width)
    return {
        "kl": kl,
        "js": js,
        "wasserstein_deg": emd,
        "n_generated": int(np.asarray(generated).size),
        "n_reference": int(np.asarray(reference).size),
        "bins": bins,
    }


def _min_image_displacement(before, after) -> np.ndarray:
    """Cartesian displacement per atom, minimum-image and drift-corrected.

    A relaxation is free to translate the whole cell, and the benchmark's own
    metrics quotient that out, so the mean displacement is removed before the
    RMSD is taken.
    """
    f0 = np.asarray(before.frac_coords, dtype=float)
    f1 = np.asarray(after.frac_coords, dtype=float)
    df = f1 - f0
    df -= np.round(df)
    df -= df.mean(axis=0, keepdims=True)
    df -= np.round(df)
    return df @ np.asarray(after.lattice_mat, dtype=float)


def relaxation_displacement(
    before,
    after,
    energy_before: Optional[float] = None,
    energy_after: Optional[float] = None,
) -> Dict[str, float]:
    """How far one generated structure moved to reach its local minimum."""
    d = _min_image_displacement(before, after)
    norms = np.linalg.norm(d, axis=1)
    v0 = float(abs(np.linalg.det(np.asarray(before.lattice_mat, dtype=float))))
    v1 = float(abs(np.linalg.det(np.asarray(after.lattice_mat, dtype=float))))
    out = {
        "rmsd_angstrom": float(np.sqrt((norms**2).mean())),
        "max_displacement_angstrom": float(norms.max()) if norms.size else 0.0,
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
