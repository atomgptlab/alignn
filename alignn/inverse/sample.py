"""Reverse-diffusion sampling for ALIGNN-CSP.

The lattice follows standard DDPM ancestral sampling in its normalised
log-symmetric 6-vector space.  The fractional coordinates follow the
predictor–corrector scheme for a variance-exploding process on the torus:
a reverse-SDE predictor step, then a few annealed Langevin corrector steps
that let the coordinates settle into the basin the score points at.

Classifier-free guidance runs the conditional and unconditional branches in a
single doubled batch.  Because the model was trained with each modality
dropped independently, ``active_modalities`` lets you guide on any subset —
composition only, property only, an XRD pattern only, or all of them.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from alignn.inverse.data import Normalizer
from alignn.inverse.denoiser import dense_pair_index
from alignn.inverse.diffusion import (
    DiffusionSchedule,
    vec6_to_lattice,
    wrap_frac,
)
from alignn.inverse.model import ALIGNNCSP, build_cond_values


def _double_pair_index(pair_index, n_nodes: int, n_graphs: int):
    """Replicate a pair index for the doubled (cond, uncond) batch."""
    src, dst, egid = pair_index
    return (
        torch.cat([src, src + n_nodes]),
        torch.cat([dst, dst + n_nodes]),
        torch.cat([egid, egid + n_graphs]),
    )


@torch.no_grad()
def sample(
    model: ALIGNNCSP,
    schedule: DiffusionSchedule,
    normalizer: Normalizer,
    batch: Dict,
    guidance: float = 2.0,
    active_modalities: Optional[Iterable[str]] = None,
    n_corrector: int = 1,
    step_lr: float = 1e-5,
    x0_clip: float = 4.0,
    device=None,
) -> Dict[str, torch.Tensor]:
    """Generate lattices and fractional coordinates for a conditioning batch.

    ``batch`` supplies the composition (``atomic_numbers``, ``natoms``,
    ``node_graph_id``) and whatever conditioning values the model was trained
    on; no geometry is needed.
    """
    device = device or batch["atomic_numbers"].device
    model.eval()
    schedule = schedule.to(device)
    normalizer = normalizer.to(device)

    natoms = batch["natoms"]
    node_graph_id = batch["node_graph_id"]
    z_atoms = batch["atomic_numbers"]
    n_graphs = int(natoms.shape[0])
    n_nodes = int(z_atoms.shape[0])

    # Composition is fixed throughout sampling, so the pair index is built
    # once instead of at every one of the T denoising steps.
    pair_index = dense_pair_index(natoms)

    cond_values = build_cond_values(batch, model.conditioner)
    on = model.conditioner.full_masks(n_graphs, device, active_modalities)
    off = model.conditioner.zero_masks(n_graphs, device)

    use_cfg = guidance != 1.0 and len(model.conditioner.names) > 0
    if use_cfg:
        pair_index_in = _double_pair_index(pair_index, n_nodes, n_graphs)
        z_in = torch.cat([z_atoms, z_atoms])
        gid_in = torch.cat([node_graph_id, node_graph_id + n_graphs])
        natoms_in = torch.cat([natoms, natoms])
        cond_in = {
            k: (None if v is None else torch.cat([v, v]))
            for k, v in cond_values.items()
        }
        masks_in = {
            k: torch.cat([on[k], off[k]]) for k in model.conditioner.names
        }
    else:
        pair_index_in = pair_index
        z_in, gid_in, natoms_in = z_atoms, node_graph_id, natoms
        cond_in, masks_in = cond_values, on

    # Priors: uniform on the torus, standard normal for the lattice.
    frac = torch.rand(n_nodes, 3, device=device)
    x_lat = torch.randn(n_graphs, 6, device=device)

    def _denoise(frac_t, x_lat_t, t_idx):
        lattice = vec6_to_lattice(normalizer.denorm_lattice(x_lat_t), natoms)
        if use_cfg:
            frac_in = torch.cat([frac_t, frac_t])
            lattice_in = torch.cat([lattice, lattice])
            xlat_in = torch.cat([x_lat_t, x_lat_t])
        else:
            frac_in, lattice_in, xlat_in = frac_t, lattice, x_lat_t
        t_vec = torch.full(
            (natoms_in.shape[0],), t_idx, device=device, dtype=torch.long
        )
        out = model(
            frac=frac_in,
            lattice=lattice_in,
            lattice_vec6=xlat_in,
            atomic_numbers=z_in,
            natoms=natoms_in,
            node_graph_id=gid_in,
            t=t_vec,
            cond_values=cond_in,
            cond_masks=masks_in,
            pair_index=pair_index_in,
        )
        if not use_cfg:
            return out["eps_frac"], out["eps_lattice"]
        ef_c, ef_u = out["eps_frac"][:n_nodes], out["eps_frac"][n_nodes:]
        el_c, el_u = (
            out["eps_lattice"][:n_graphs],
            out["eps_lattice"][n_graphs:],
        )
        return (
            ef_u + guidance * (ef_c - ef_u),
            el_u + guidance * (el_c - el_u),
        )

    sigmas = schedule.sigmas
    alphas = schedule.alphas
    alpha_bar = schedule.alpha_bar
    betas = schedule.betas

    for t in range(schedule.num_steps, 0, -1):
        sigma_t = sigmas[t]
        sigma_prev = sigmas[t - 1]

        # ── corrector: annealed Langevin on the coordinates ──────────────
        for _ in range(n_corrector):
            eps_f, _ = _denoise(frac, x_lat, t)
            score = eps_f / sigma_t
            step = step_lr * (sigma_t / sigmas[1]) ** 2
            noise = torch.randn_like(frac)
            frac = wrap_frac(
                frac + step * score + torch.sqrt(2.0 * step) * noise
            )

        # ── predictor ────────────────────────────────────────────────────
        eps_f, eps_l = _denoise(frac, x_lat, t)

        # Coordinates: reverse-SDE step for a VE process on the torus.
        score = eps_f / sigma_t
        d_sigma2 = sigma_t**2 - sigma_prev**2
        noise_scale = torch.sqrt(
            (sigma_prev**2 * d_sigma2 / sigma_t**2).clamp_min(0.0)
        )
        frac = wrap_frac(
            frac + d_sigma2 * score + noise_scale * torch.randn_like(frac)
        )

        # Lattice: DDPM ancestral step, written in posterior-mean form.
        # The equivalent "eps" form multiplies by 1/sqrt(alpha_t), which near
        # t=T (beta -> 0.999) is a ~30x per-step amplification and overflows
        # the moment the noise prediction is even slightly off. Reconstructing
        # x0 and clipping it keeps every quantity bounded instead.
        a_t = alphas[t - 1]
        b_t = betas[t - 1]
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1]
        x0_hat = ((x_lat - (1.0 - ab_t).sqrt() * eps_l) / ab_t.sqrt()).clamp(
            -x0_clip, x0_clip
        )
        coef_x0 = ab_prev.sqrt() * b_t / (1.0 - ab_t)
        coef_xt = a_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)
        mean = coef_x0 * x0_hat + coef_xt * x_lat
        if t > 1:
            var = b_t * (1.0 - ab_prev) / (1.0 - ab_t)
            x_lat = mean + var.sqrt() * torch.randn_like(x_lat)
        else:
            x_lat = mean

    lattice = vec6_to_lattice(normalizer.denorm_lattice(x_lat), natoms)
    return {"frac": wrap_frac(frac), "lattice": lattice, "natoms": natoms}


def to_jarvis_atoms(
    frac: torch.Tensor,
    lattice: torch.Tensor,
    atomic_numbers: torch.Tensor,
    natoms: torch.Tensor,
) -> List:
    """Split a flattened sample batch into jarvis ``Atoms`` objects."""
    from jarvis.core.atoms import Atoms
    from jarvis.core.specie import atomic_numbers_to_symbols

    out = []
    offset = 0
    frac_np = frac.detach().cpu().numpy()
    lat_np = lattice.detach().cpu().numpy()
    z_np = atomic_numbers.detach().cpu().numpy()
    for b, n in enumerate(natoms.tolist()):
        elements = list(
            atomic_numbers_to_symbols(
                [int(z) for z in z_np[offset : offset + n]]
            )
        )
        out.append(
            Atoms(
                lattice_mat=np.asarray(lat_np[b], dtype=float),
                coords=np.asarray(frac_np[offset : offset + n], dtype=float),
                elements=elements,
                cartesian=False,
            )
        )
        offset += n
    return out


def load_model(checkpoint_path, device, use_ema: bool = True):
    """Load an ALIGNN-CSP checkpoint into a ready-to-sample model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ALIGNNCSP(
        denoiser_config={
            "hidden_features": cfg["hidden_features"],
            "alignn_layers": cfg["alignn_layers"],
            "gcn_layers": cfg["gcn_layers"],
            "knn": cfg["knn"],
            "num_steps": cfg["num_steps"],
        },
        conditioner_spec=ckpt["conditioner_spec"],
    ).to(device)
    # Released checkpoints carry only the EMA weights, which are the ones
    # used for sampling; fall back to whichever of the two is present.
    state = ckpt.get("ema" if use_ema else "model")
    if state is None:
        state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    normalizer = Normalizer.from_dict(ckpt["normalizer"]).to(device)
    schedule = DiffusionSchedule(
        num_steps=cfg["num_steps"],
        sigma_min=cfg["sigma_min"],
        sigma_max=cfg["sigma_max"],
    ).to(device)
    return model, schedule, normalizer, cfg
