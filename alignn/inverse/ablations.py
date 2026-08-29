"""Named configurations for the angular-diffusion ablation suite.

The point of this extension is not to obtain one better model, it is to
separate three claims that a single "it improved" number cannot:

1. does an **explicit three-body denoising objective** help, over and above
   angles being an ordinary ALIGNN input feature?
2. does **continuously varying interaction topology** help in the noisy
   regime, independently of any angular objective?
3. does it matter that the angular channel is **coupled back** into the
   coordinate/lattice pathway, rather than merely supervised alongside it?

Every entry below is a keyword dict for
:class:`alignn.inverse.denoiser.ALIGNNCSPDenoiser`.  They differ *only* in the
switches under test — hidden size, depth, schedule, optimiser, data splits and
seed policy come from the training script and must be held fixed across a
comparison for it to mean anything.

    A0  (A, F, L)                      current model, no angular objective
    A1  (A, F, L, Theta)               + explicit angular denoising
    A2  (A, F, L) + smooth topology    smooth graph, no angular objective
    A3  (A, F, L, Theta) + smooth      the proposed model
    A4  A3 with the coupling cut       control: auxiliary supervision only
    A6  A3 with a Fourier angle basis  angular-representation ablation

A5 in the design brief — hard kNN versus the smooth radius graph — is a
*comparison*, not a fifth configuration: it is A1 against A3 (and A0 against
A2), which is why there is no ``"A5"`` key.  :data:`COMPARISONS` spells out
which pair of runs answers which question.

Atom types are not diffused in this implementation.  The generator is
conditioned on composition and solves crystal structure prediction, so the
state is really ``(F, L)`` and, with these switches, ``(F, L, Theta)``; the
``A`` in the names is kept only to match the design brief's notation.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["ABLATIONS", "COMPARISONS", "ablation_config", "describe"]

_SMOOTH = {
    "topology": "radius",
    "gate_pair_messages": True,
}

ABLATIONS: Dict[str, Dict] = {
    "A0": {
        "angle_diffusion": False,
        "topology": "knn",
        "gate_pair_messages": False,
        "angle_feedback": True,
    },
    "A1": {
        "angle_diffusion": True,
        "topology": "knn",
        "gate_pair_messages": False,
        "angle_feedback": True,
    },
    "A2": {
        "angle_diffusion": False,
        "angle_feedback": True,
        **_SMOOTH,
    },
    "A3": {
        "angle_diffusion": True,
        "angle_feedback": True,
        **_SMOOTH,
    },
    "A4": {
        "angle_diffusion": True,
        "angle_feedback": False,
        **_SMOOTH,
    },
    "A6": {
        "angle_diffusion": True,
        "angle_feedback": True,
        "angle_basis": "fourier",
        **_SMOOTH,
    },
}

#: What each ablation is for, and which contrast it belongs to.
DESCRIPTIONS: Dict[str, str] = {
    "A0": "baseline: current ALIGNN 2.0 diffusion, angles as features only",
    "A1": "explicit angular denoising, baseline kNN line-graph topology",
    "A2": "smooth radius topology, no angular denoising objective",
    "A3": "proposed: explicit angular denoising + smooth topology",
    "A4": "control: angular objective with the angle->bond coupling removed",
    "A6": "A3 with the Fourier angular basis instead of ALIGNN's cosine RBF",
}

COMPARISONS = {
    "does explicit angular denoising help": ("A0", "A1"),
    "does smooth topology alone help": ("A0", "A2"),
    "do the two together help": ("A0", "A3"),
    "is the coupling doing the work (not just auxiliary loss)": ("A4", "A3"),
    "A5: hard kNN vs smooth radius, with angles on": ("A1", "A3"),
    "A5: hard kNN vs smooth radius, with angles off": ("A0", "A2"),
    "A6: does the angular basis matter": ("A3", "A6"),
}


def ablation_config(name: str) -> Dict:
    """Denoiser keyword arguments for one named ablation."""
    key = name.upper()
    if key not in ABLATIONS:
        raise KeyError(
            f"unknown ablation {name!r}; available: "
            f"{', '.join(sorted(ABLATIONS))} "
            "(A5 is the A1-vs-A3 comparison, not a configuration)"
        )
    return dict(ABLATIONS[key])


def describe(name: str) -> str:
    """One-line description of a named ablation."""
    return DESCRIPTIONS[name.upper()]
