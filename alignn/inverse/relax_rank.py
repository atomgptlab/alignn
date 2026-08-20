"""Relax and rank generated candidates with the pretrained ALIGNN force field.

This is the step that separates a generator built inside the ALIGNN ecosystem
from a standalone one.  The reference structures a reconstruction benchmark
scores against are DFT-relaxed, so they sit at local minima of the potential
energy surface.  A diffusion sample lands *near* such a minimum; pushing it
the rest of the way with a universal ML force field moves it onto exactly the
manifold the metric rewards — and the resulting energy gives a physically
meaningful way to pick the best of several candidates for the same
composition, which is the classic crystal-structure-prediction recipe.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class RelaxResult:
    """Outcome of relaxing one candidate."""

    atoms: object  # jarvis Atoms (relaxed, or the input)
    energy_per_atom: float
    converged: bool
    steps: int
    error: Optional[str] = None


class AlignnFFRelaxer:
    """Thin wrapper around the ALIGNN-FF ASE calculator.

    Parameters
    ----------
    model_path : str or None
        Force-field directory; defaults to ``alignn.ff.ff.default_path()``.
    relax_cell : bool
        Relax lattice vectors as well as positions.  For structure
        reconstruction this should stay on — the cell is half of what the
        benchmark measures.
    fmax, steps : float, int
        Standard ASE convergence controls.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        relax_cell: bool = True,
        fmax: float = 0.05,
        steps: int = 200,
        device: Optional[str] = None,
    ):
        from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

        self.relax_cell = relax_cell
        self.fmax = fmax
        self.steps = steps
        kw = {}
        if device is not None:
            kw["device"] = device
        self.calculator = AlignnAtomwiseCalculator(
            path=model_path or default_path(), **kw
        )

    def energy(self, jatoms) -> float:
        """Single-point energy per atom, in eV/atom."""
        ase_atoms = jatoms.ase_converter()
        ase_atoms.calc = self.calculator
        return float(ase_atoms.get_potential_energy()) / len(ase_atoms)

    def relax(self, jatoms) -> RelaxResult:
        """Relax one structure; never raises, reports failures instead."""
        from ase.optimize.fire import FIRE
        from jarvis.core.atoms import ase_to_atoms

        n = len(jatoms.elements)
        try:
            ase_atoms = jatoms.ase_converter()
            ase_atoms.calc = self.calculator
            target = ase_atoms
            if self.relax_cell:
                from ase.filters import FrechetCellFilter

                target = FrechetCellFilter(ase_atoms)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt = FIRE(target, logfile=None)
                opt.run(fmax=self.fmax, steps=self.steps)
                energy = float(ase_atoms.get_potential_energy()) / n
            return RelaxResult(
                atoms=ase_to_atoms(ase_atoms),
                energy_per_atom=energy,
                converged=bool(opt.converged()),
                steps=int(opt.get_number_of_steps()),
            )
        except Exception as exc:  # noqa: BLE001
            # A pathological sample (atoms on top of each other, collapsed
            # cell) can blow up the optimiser. Keep the unrelaxed candidate
            # and let energy ranking discard it.
            return RelaxResult(
                atoms=jatoms,
                energy_per_atom=float("inf"),
                converged=False,
                steps=0,
                error=f"{type(exc).__name__}: {exc}",
            )


def min_interatomic_distance(jatoms) -> float:
    """Shortest periodic interatomic distance, in Angstrom.

    Used as a cheap sanity filter: a sample with atoms 0.3 A apart is not a
    crystal, and is far cheaper to reject here than to relax.
    """
    coords = np.asarray(jatoms.frac_coords)
    lat = np.asarray(jatoms.lattice_mat)
    n = len(coords)
    offsets = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    best = np.inf
    for i in range(n):
        d = coords - coords[i]
        d = d - np.round(d)
        cand = (d[:, None, :] + offsets[None, :, :]) @ lat
        dist = np.linalg.norm(cand, axis=-1)
        # Mask out the atom paired with its own zero image.
        dist[i, 13] = np.inf
        best = min(best, float(dist.min()))
    return best


def rank_candidates(
    candidates: Sequence,
    relaxer: Optional[AlignnFFRelaxer],
    relax: bool = True,
    min_distance: float = 0.7,
) -> List[RelaxResult]:
    """Relax (optionally) and score every candidate for one composition.

    Returns results sorted by energy per atom, lowest first.  Candidates whose
    closest contact is below ``min_distance`` are pushed to the back rather
    than dropped, so a caller always gets a usable structure back even if
    every sample was poor.
    """
    results: List[RelaxResult] = []
    for cand in candidates:
        too_close = min_interatomic_distance(cand) < min_distance
        if relaxer is None:
            results.append(
                RelaxResult(
                    atoms=cand,
                    energy_per_atom=float("inf") if too_close else 0.0,
                    converged=False,
                    steps=0,
                )
            )
            continue
        if too_close:
            results.append(
                RelaxResult(
                    atoms=cand,
                    energy_per_atom=float("inf"),
                    converged=False,
                    steps=0,
                    error="overlapping atoms",
                )
            )
            continue
        res = (
            relaxer.relax(cand)
            if relax
            else RelaxResult(
                atoms=cand,
                energy_per_atom=_safe_energy(relaxer, cand),
                converged=False,
                steps=0,
            )
        )
        results.append(res)
    return sorted(results, key=lambda r: r.energy_per_atom)


def _safe_energy(relaxer: AlignnFFRelaxer, jatoms) -> float:
    try:
        return relaxer.energy(jatoms)
    except Exception:  # noqa: BLE001
        return float("inf")


# ── parallel relaxation ──────────────────────────────────────────────────
#
# Relaxing one small crystal is dominated by graph construction, not by the
# network: a 4-atom single point spends ~78% of its time building the
# neighbour and line graphs and only ~20 ms in ALIGNN itself. That makes the
# work CPU-bound and embarrassingly parallel, and it is why the workers below
# run on CPU — a contended GPU is actually several times *slower* per step
# here, and CPU workers scale across every core without fighting each other.

_WORKER: Optional[AlignnFFRelaxer] = None


def _worker_init(model_path, relax_cell, fmax, steps):
    global _WORKER
    import torch

    torch.set_num_threads(1)
    _WORKER = AlignnFFRelaxer(
        model_path=model_path,
        relax_cell=relax_cell,
        fmax=fmax,
        steps=steps,
        device="cpu",
    )


def _worker_task(payload):
    """Relax (or just score) one candidate given as a jarvis Atoms dict."""
    from jarvis.core.atoms import Atoms

    idx, atoms_dict, do_relax, min_distance = payload
    atoms = Atoms.from_dict(atoms_dict)
    if min_interatomic_distance(atoms) < min_distance:
        return idx, atoms_dict, float("inf"), False, 0, "overlapping atoms"
    if do_relax:
        res = _WORKER.relax(atoms)
        return (
            idx,
            res.atoms.to_dict(),
            res.energy_per_atom,
            res.converged,
            res.steps,
            res.error,
        )
    return idx, atoms_dict, _safe_energy(_WORKER, atoms), False, 0, None


def parallel_rank(
    candidate_groups: Sequence[Sequence],
    model_path: Optional[str] = None,
    relax: bool = True,
    relax_cell: bool = True,
    fmax: float = 0.05,
    steps: int = 150,
    min_distance: float = 0.7,
    n_workers: Optional[int] = None,
    progress_every: int = 50,
) -> List[List[RelaxResult]]:
    """Relax and rank many candidate groups across a process pool.

    ``candidate_groups[i]`` holds the candidates generated for target ``i``;
    the returned list has the same shape, each group sorted by energy per atom.
    """
    import multiprocessing as mp
    import os
    import sys
    import time

    from jarvis.core.atoms import Atoms

    n_workers = n_workers or max(1, min(24, (os.cpu_count() or 4) - 2))

    tasks = []
    for gi, group in enumerate(candidate_groups):
        for cand in group:
            tasks.append((gi, cand.to_dict(), relax, min_distance))
    if not tasks:
        return [[] for _ in candidate_groups]

    results: List[List[RelaxResult]] = [[] for _ in candidate_groups]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    done = 0
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(model_path, relax_cell, fmax, steps),
    ) as pool:
        for out in pool.imap_unordered(_worker_task, tasks, chunksize=1):
            gi, atoms_dict, energy, converged, nsteps, error = out
            results[gi].append(
                RelaxResult(
                    atoms=Atoms.from_dict(atoms_dict),
                    energy_per_atom=energy,
                    converged=converged,
                    steps=nsteps,
                    error=error,
                )
            )
            done += 1
            if progress_every and done % progress_every == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"    relaxed {done}/{len(tasks)} "
                    f"({rate:.1f}/s, eta {(len(tasks) - done) / rate:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    return [sorted(g, key=lambda r: r.energy_per_atom) for g in results]
