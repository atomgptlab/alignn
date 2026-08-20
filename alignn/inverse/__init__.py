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
"""

__all__ = ["data", "denoiser", "diffusion", "sample"]
