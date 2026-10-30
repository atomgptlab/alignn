"""Inverse design (generative crystal structure prediction) for ALIGNN.

``alignn.inverse`` implements ALIGNN-CSP: a joint diffusion model over the
lattice and the fractional coordinates of a crystal, conditioned on the
composition and a scalar target property, with a pure-PyTorch ALIGNN as the
denoising network.

The pieces:

``diffusion``  noise schedules, the wrapped-normal (torus) process used for
               fractional coordinates and the DDPM process used for the
               symmetric lattice representation.
``denoiser``   the ALIGNN denoising network.
``data``       dataset / collation from the AtomBench split JSONs.
``sample``     ancestral + Langevin-corrector sampling with classifier-free
               guidance on the conditioning property.
``angles``     bond-angle denoising target and the smooth (DimeNet envelope /
               ReaxFF product gate) triplet topology.
``layers``     ALIGNN convolutions taking an optional per-edge or per-triplet
               weight, so a message can fade out instead of being deleted.
``ablations``  named configurations for the angular-diffusion ablation suite.
``evaluate``   bond-angle distribution and relaxation-displacement metrics.
"""

__all__ = [
    "ablations",
    "angles",
    "data",
    "denoiser",
    "diffusion",
    "evaluate",
    "layers",
    "sample",
]
