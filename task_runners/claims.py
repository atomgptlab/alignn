"""Every quantitative claim the manuscript makes about inverse design.

This is the coverage map. Each entry names a number printed in the paper, the
task that regenerates it, and where in that task's output it appears, so
``run_task.py verify`` can answer two different questions:

* **before running anything** -- is every claim reachable from an executable
  in this directory, and which tasks would I have to run?
* **after running** -- does what I measured agree with what was published?

A claim with no task is a claim this directory cannot reproduce, and there
should not be any; ``verify`` fails loudly if one appears.  Tolerances are
generous on purpose: with 103 test targets, a match rate differing by 0.03 is
three structures, and the manuscript itself reports a spread of nine across
seeds.  ``verify`` prints the measured spread next to the published value
rather than reducing agreement to a single pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Claim:
    """One published number, and where to find its measured counterpart."""

    source: str
    statement: str
    task: str
    metric: str
    published: float
    group: str = ""
    #: When set, ``published`` is metric[group] / metric[ref_group].
    ref_group: str = ""
    variant: str = "sym"
    #: Relative tolerance for calling the claim reproduced.
    tol: float = 0.10


A = "A: line graph"
B = "B: no line graph"
CSP = "ALIGNN-CSP"

CLAIMS: List[Claim] = [
    # -- splits -------------------------------------------------------------
    Claim(
        "text",
        "JARVIS Supercon-3D split is 847/105/103",
        "data-jarvis",
        "n_train",
        847,
        tol=0.0,
    ),
    Claim(
        "text",
        "JARVIS Supercon-3D split is 847/105/103",
        "data-jarvis",
        "n_val",
        105,
        tol=0.0,
    ),
    Claim(
        "text",
        "JARVIS Supercon-3D split is 847/105/103",
        "data-jarvis",
        "n_test",
        103,
        tol=0.0,
    ),
    Claim(
        "text",
        "Alexandria DS-A/B split is 6603/825/825",
        "data-alex",
        "n_train",
        6603,
        tol=0.0,
    ),
    Claim(
        "text",
        "Alexandria DS-A/B split is 6603/825/825",
        "data-alex",
        "n_val",
        825,
        tol=0.0,
    ),
    Claim(
        "text",
        "Alexandria DS-A/B split is 6603/825/825",
        "data-alex",
        "n_test",
        825,
        tol=0.0,
    ),
    # -- Table 3, tab:inverse_ablation --------------------------------------
    # Only the means are registered. The table's RMSD spreads (0.030+-0.001
    # against 0.048+-0.013) are known not to hold: the inverse-design README
    # records six models per arm giving 0.031+-0.012 against 0.044+-0.011 and
    # calls the original tightness a small-sample artifact. The RMSD
    # tolerances below are wide for that reason, and --aggregate prints the
    # measured spread so the point is visible rather than asserted.
    Claim(
        "Table 3",
        "denoising loss, line graph",
        "ablation-linegraph",
        "loss",
        1.997,
        group=A,
        tol=0.05,
    ),
    Claim(
        "Table 3",
        "denoising loss, no line graph",
        "ablation-linegraph",
        "loss",
        2.351,
        group=B,
        tol=0.05,
    ),
    Claim(
        "Table 3",
        "coordinate RMSD, line graph",
        "ablation-linegraph",
        "rmsd",
        0.030,
        group=A,
        tol=0.35,
    ),
    Claim(
        "Table 3",
        "coordinate RMSD, no line graph",
        "ablation-linegraph",
        "rmsd",
        0.048,
        group=B,
        tol=0.35,
    ),
    Claim(
        "Table 3",
        "ccRMSD, line graph",
        "ablation-linegraph",
        "ccrmsd",
        0.508,
        group=A,
    ),
    Claim(
        "Table 3",
        "ccRMSD, no line graph",
        "ablation-linegraph",
        "ccrmsd",
        0.521,
        group=B,
    ),
    Claim(
        "Table 3",
        "lattice MAE abc, line graph",
        "ablation-linegraph",
        "abc",
        0.535,
        group=A,
        tol=0.15,
    ),
    Claim(
        "Table 3",
        "lattice MAE abc, no line graph",
        "ablation-linegraph",
        "abc",
        0.542,
        group=B,
        tol=0.15,
    ),
    Claim(
        "Table 3",
        "lattice MAE angles, line graph",
        "ablation-linegraph",
        "ang",
        9.47,
        group=A,
        tol=0.15,
    ),
    Claim(
        "Table 3",
        "lattice MAE angles, no line graph",
        "ablation-linegraph",
        "ang",
        9.76,
        group=B,
        tol=0.15,
    ),
    Claim(
        "Table 3",
        "match rate, line graph",
        "ablation-linegraph",
        "match",
        0.4725,
        group=A,
        tol=0.10,
    ),
    Claim(
        "Table 3",
        "match rate, no line graph",
        "ablation-linegraph",
        "match",
        0.4725,
        group=B,
        tol=0.10,
    ),
    Claim(
        "Table 3 caption",
        "parameters matched within 1%: 3.79 M",
        "ablation-linegraph",
        "params",
        3.79e6,
        group=A,
        tol=0.01,
    ),
    Claim(
        "Table 3 caption",
        "parameters matched within 1%: 3.75 M",
        "ablation-linegraph",
        "params",
        3.75e6,
        group=B,
        tol=0.01,
    ),
    Claim(
        "text",
        "angles cost 2.4x in time per training step",
        "ablation-linegraph",
        "train_s",
        2.4,
        group=A,
        ref_group=B,
        tol=0.30,
    ),
    # -- Table 4, JARVIS Supercon-3D block ----------------------------------
    Claim(
        "Table 4",
        "JARVIS match rate, mean of three seeds",
        "bench-jarvis",
        "match",
        0.473,
        group=CSP,
        tol=0.10,
    ),
    Claim(
        "Table 4",
        "JARVIS coordinate RMSD",
        "bench-jarvis",
        "rmsd",
        0.030,
        group=CSP,
        tol=0.35,
    ),
    Claim(
        "Table 4", "JARVIS ccRMSD", "bench-jarvis", "ccrmsd", 0.508, group=CSP
    ),
    Claim(
        "Table 4",
        "JARVIS lattice MAE abc",
        "bench-jarvis",
        "abc",
        0.535,
        group=CSP,
        tol=0.15,
    ),
    Claim(
        "Table 4",
        "JARVIS lattice MAE angles",
        "bench-jarvis",
        "ang",
        9.47,
        group=CSP,
        tol=0.15,
    ),
    Claim(
        "Table 4",
        "JARVIS KLD",
        "bench-jarvis",
        "kld",
        0.023,
        group=CSP,
        tol=0.30,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, match",
        "bench-jarvis",
        "bestrun_match",
        0.524,
        group=CSP,
        tol=0.10,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, RMSD",
        "bench-jarvis",
        "bestrun_rmsd",
        0.023,
        group=CSP,
        tol=0.40,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, ccRMSD",
        "bench-jarvis",
        "bestrun_ccrmsd",
        0.470,
        group=CSP,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, MAE abc",
        "bench-jarvis",
        "bestrun_abc",
        0.433,
        group=CSP,
        tol=0.20,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, MAE angles",
        "bench-jarvis",
        "bestrun_ang",
        8.37,
        group=CSP,
        tol=0.20,
    ),
    Claim(
        "Table 4",
        "JARVIS best single run, KLD",
        "bench-jarvis",
        "bestrun_kld",
        0.018,
        group=CSP,
        tol=0.35,
    ),
    Claim(
        "text",
        "match rate across seeds spans 0.437 at the low end",
        "bench-jarvis",
        "match_min",
        0.437,
        group=CSP,
        tol=0.10,
    ),
    Claim(
        "text",
        "match rate across seeds spans 0.524 at the high end",
        "bench-jarvis",
        "match_max",
        0.524,
        group=CSP,
        tol=0.10,
    ),
    # -- Table 4, Alexandria DS-A/B block -----------------------------------
    Claim(
        "Table 4",
        "Alexandria match rate",
        "bench-alex",
        "match",
        0.485,
        group=CSP,
        tol=0.10,
    ),
    Claim(
        "Table 4",
        "Alexandria coordinate RMSD",
        "bench-alex",
        "rmsd",
        0.028,
        group=CSP,
        tol=0.35,
    ),
    Claim(
        "Table 4",
        "Alexandria ccRMSD",
        "bench-alex",
        "ccrmsd",
        0.343,
        group=CSP,
        tol=0.15,
    ),
    Claim(
        "Table 4",
        "Alexandria lattice MAE abc",
        "bench-alex",
        "abc",
        0.561,
        group=CSP,
        tol=0.15,
    ),
    Claim(
        "Table 4",
        "Alexandria lattice MAE angles",
        "bench-alex",
        "ang",
        10.09,
        group=CSP,
        tol=0.15,
    ),
    Claim(
        "Table 4",
        "Alexandria KLD",
        "bench-alex",
        "kld",
        0.023,
        group=CSP,
        tol=0.30,
    ),
    # -- "Closing the loop with the force field" ----------------------------
    # Quoted before symmetrisation: these are about what sampling and the
    # force field contribute, not about the lattice metrics.
    Claim(
        "text",
        "one sample, no selection or relaxation: match 0.22",
        "pipeline-ablation",
        "match",
        0.22,
        group="raw",
        variant="nosym",
        tol=0.15,
    ),
    Claim(
        "text",
        "one sample, no selection or relaxation: RMSD 0.29",
        "pipeline-ablation",
        "rmsd",
        0.29,
        group="raw",
        variant="nosym",
        tol=0.25,
    ),
    Claim(
        "text",
        "relaxation without selection contributes almost nothing",
        "pipeline-ablation",
        "match",
        0.24,
        group="relax",
        variant="nosym",
        tol=0.15,
    ),
    Claim(
        "text",
        "32 candidates ranked and relaxed: match 0.52",
        "pipeline-ablation",
        "match",
        0.52,
        group="full",
        variant="nosym",
        tol=0.10,
    ),
    Claim(
        "text",
        "32 candidates ranked and relaxed: RMSD 0.06",
        "pipeline-ablation",
        "rmsd",
        0.06,
        group="full",
        variant="nosym",
        tol=0.40,
    ),
    # -- the leakage caveat -------------------------------------------------
    Claim(
        "text",
        "18.4% of JARVIS test targets are reachable by recall",
        "leakage",
        "leak_fraction",
        0.184,
        group="jarvis:all",
        tol=0.05,
    ),
    Claim(
        "text",
        "15.4% of Alexandria test targets are reachable by recall",
        "leakage",
        "leak_fraction",
        0.154,
        group="alex:all",
        tol=0.05,
    ),
]


def tasks_needed() -> List[str]:
    """Distinct tasks that have to run before every claim can be checked."""
    seen = []
    for claim in CLAIMS:
        if claim.task not in seen:
            seen.append(claim.task)
    return seen
