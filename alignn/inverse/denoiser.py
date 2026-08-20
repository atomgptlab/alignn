"""ALIGNN denoising network for crystal structure diffusion.

The network is a *scalar-in, scalar-out* model, which is what makes it simple
and stable to train:

* Fractional coordinates are expressed in the lattice basis, so they carry no
  rotational covariance — the only symmetry the coordinate head must respect
  is invariance to a global shift of all coordinates.  Feeding the network
  only pairwise differences ``Δf_ij`` (through Fourier features, which are
  additionally invariant to the choice of periodic image) buys that for free.
* The lattice is diffused in its rotation-invariant symmetric log
  representation, so the lattice head predicts 6 plain scalars.

Everything the network sees — RBFs of minimum-image distances, Fourier
features of fractional differences, bond-angle cosines — is invariant under
the full symmetry group, so no equivariant machinery is required.

The distinguishing ingredient relative to CSPNet / CDVAE / FlowMM denoisers is
the ALIGNN line graph: bond *angles* are propagated alongside bond lengths,
which is the three-body information that pins down coordination geometry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from alignn.models.alignn_atomwise_pure import (
    ALIGNNConvPure,
    EdgeGatedGraphConvPure,
    scatter_mean,
    scatter_sum,
)
from alignn.models.utils import MLPLayer, RBFExpansion
from alignn.torch_graph_builder import _line_graph_edges, torch_bond_cosines

from alignn.inverse.diffusion import wrap_diff


def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 1e4):
    """Standard transformer-style sinusoidal embedding of a scalar."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=x.device)
        / half
    )
    args = x.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def dense_pair_index(
    natoms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All ordered intra-crystal pairs, self-pairs included.

    Self-pairs ``i -> i`` are kept deliberately: resolved against a non-zero
    periodic image they encode how close an atom sits to its own translated
    copies, which is the only pair information a one-atom cell has at all.

    Returns ``(src, dst, edge_graph_id)`` with node indices already offset
    into the flattened batch.
    """
    device = natoms.device
    srcs, dsts, gids = [], [], []
    offset = 0
    for b, n in enumerate(natoms.tolist()):
        idx = torch.arange(n, device=device)
        s = idx.repeat_interleave(n)
        d = idx.repeat(n)
        srcs.append(s + offset)
        dsts.append(d + offset)
        gids.append(
            torch.full((s.numel(),), b, device=device, dtype=torch.long)
        )
        offset += n
    return torch.cat(srcs), torch.cat(dsts), torch.cat(gids)


def _knn_mask(
    dist: torch.Tensor, dst: torch.Tensor, num_nodes: int, k: int
) -> torch.Tensor:
    """Boolean mask keeping the ``k`` shortest edges incident on each dst."""
    if k <= 0:
        return torch.ones_like(dist, dtype=torch.bool)
    # Sort edges by (dst, distance) so each dst's edges are contiguous and
    # ordered; the within-group rank then gives the neighbour ranking.
    order = torch.argsort(dist, stable=True)
    order = order[torch.argsort(dst[order], stable=True)]
    sorted_dst = dst[order]
    counts = torch.bincount(sorted_dst, minlength=num_nodes)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(dist.numel(), device=dist.device) - starts[sorted_dst]
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask[order] = rank < k
    return mask


# 27 candidate periodic images for minimum-image resolution. Index 13 is the
# (0, 0, 0) offset, given the [-1, 0, 1] cartesian-product ordering.
_ZERO_OFFSET = 13


def _image_offsets(device, dtype):
    r = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    return torch.cartesian_prod(r, r, r)  # (27, 3)


class ALIGNNCSPDenoiser(nn.Module):
    """Predict (coordinate score, lattice noise) for a noised crystal."""

    def __init__(
        self,
        hidden_features: int = 256,
        embedding_features: int = 128,
        alignn_layers: int = 3,
        gcn_layers: int = 3,
        fourier_k: int = 10,
        rbf_bins: int = 64,
        triplet_bins: int = 40,
        rbf_cutoff: float = 10.0,
        knn: int = 12,
        num_species: int = 120,
        num_steps: int = 1000,
        score_channels: int = 32,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.fourier_k = fourier_k
        self.knn = knn
        self.num_steps = num_steps

        self.species_embedding = nn.Embedding(num_species, hidden_features)

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        # The lattice head has to see the current lattice state explicitly.
        self.lattice_mlp = nn.Sequential(
            nn.Linear(7, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.embedding_features = embedding_features

        # Edge features: RBF of minimum-image distance + Fourier of Δf.
        self.rbf = RBFExpansion(vmin=0.0, vmax=rbf_cutoff, bins=rbf_bins)
        edge_in = rbf_bins + 6 * fourier_k
        self.edge_embedding = nn.Sequential(
            MLPLayer(edge_in, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        # With no ALIGNN layers there is nothing to consume bond angles, so
        # the angle encoder is not built and the line graph is never
        # constructed. That keeps a no-line-graph ablation honest on
        # parameter count and free of the cost of building triplets it
        # would then discard.
        self.use_line_graph = alignn_layers > 0
        self.angle_embedding = (
            nn.Sequential(
                RBFExpansion(vmin=-1.0, vmax=1.0, bins=triplet_bins),
                MLPLayer(triplet_bins, embedding_features),
                MLPLayer(embedding_features, hidden_features),
            )
            if self.use_line_graph
            else None
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConvPure(hidden_features, hidden_features)
                for _ in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConvPure(hidden_features, hidden_features)
                for _ in range(gcn_layers)
            ]
        )

        # Coordinate score head.
        #
        # An MLP straight off the node feature cannot reliably emit a
        # *direction*: ALIGNN's edge-gated convolution lets the edge feature
        # only gate a source-node feature, so the aggregated message carries
        # magnitude far more readily than orientation. (ALIGNN-FF sidesteps
        # this entirely — it never emits a vector, it differentiates a scalar
        # energy.) So the score is assembled from the edge vectors themselves:
        # each edge contributes its fractional offset weighted by a learned
        # scalar, which is direction-correct by construction and stays
        # invariant to a global shift of all coordinates.
        self.score_channels = score_channels
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, score_channels),
        )
        self.score_combine = nn.Linear(score_channels, 1, bias=False)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 6),
        )
        # Start from a near-zero prediction: diffusion training is much better
        # behaved when the model does not begin by shouting.
        nn.init.zeros_(self.score_combine.weight)
        nn.init.zeros_(self.lattice_head[-1].weight)
        nn.init.zeros_(self.lattice_head[-1].bias)

    # ── geometry ─────────────────────────────────────────────────────────
    def _edge_geometry(
        self,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_graph_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return wrapped Δf, min-image Δf, min-image Cartesian vec, dist."""
        df = wrap_diff(frac[dst] - frac[src])
        offsets = _image_offsets(frac.device, frac.dtype)  # (27, 3)
        cand = df.unsqueeze(1) + offsets.unsqueeze(0)  # (E, 27, 3)
        lat_e = lattice[edge_graph_id]  # (E, 3, 3)
        cart = torch.einsum("eoi,eij->eoj", cand, lat_e)  # (E, 27, 3)
        d_all = cart.norm(dim=-1)  # (E, 27)
        # A self-pair must not resolve to the zero image (distance 0); force
        # it onto the nearest *translated* copy instead.
        self_edge = src == dst
        if bool(self_edge.any()):
            d_all = d_all.clone()
            d_all[self_edge, _ZERO_OFFSET] = float("inf")
        best = d_all.argmin(dim=1)  # (E,)
        idx = best.view(-1, 1, 1).expand(-1, 1, 3)
        r = cart.gather(1, idx).squeeze(1)  # (E, 3)
        # The fractional difference for the *same* image, which is what the
        # coordinate score head combines.
        df_min = cand.gather(1, idx).squeeze(1)  # (E, 3)
        return df, df_min, r, d_all.gather(1, best.view(-1, 1)).squeeze(1)

    def _fourier(self, df: torch.Tensor) -> torch.Tensor:
        """Fourier features of a fractional difference (periodic, signed)."""
        k = torch.arange(
            1, self.fourier_k + 1, device=df.device, dtype=df.dtype
        )
        ang = 2.0 * math.pi * df.unsqueeze(-1) * k  # (E, 3, K)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(
            df.shape[0], -1
        )

    # ── forward ──────────────────────────────────────────────────────────
    def forward(
        self,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        lattice_vec6: torch.Tensor,
        atomic_numbers: torch.Tensor,
        natoms: torch.Tensor,
        node_graph_id: torch.Tensor,
        t: torch.Tensor,
        cond_embedding: Optional[torch.Tensor] = None,
        pair_index: Optional[Tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict the coordinate score and the lattice noise.

        ``cond_embedding`` is a ``(B, hidden_features)`` vector produced by a
        :class:`~alignn.inverse.conditioners.MultiModalConditioner` — the
        denoiser is deliberately agnostic to which modalities went into it.
        """
        num_nodes = frac.shape[0]
        if pair_index is None:
            pair_index = dense_pair_index(natoms)
        src, dst, edge_graph_id = pair_index

        df, df_min, r, dist = self._edge_geometry(
            frac, lattice, src, dst, edge_graph_id
        )

        # Node features: species + timestep + conditioning + lattice state.
        h = self.species_embedding(atomic_numbers)
        t_emb = self.time_mlp(
            sinusoidal_embedding(
                t.float() / float(self.num_steps) * 1000.0,
                self.embedding_features,
            )
        )
        lat_feat = torch.cat(
            [lattice_vec6, natoms.to(lattice_vec6.dtype).view(-1, 1) / 20.0],
            dim=-1,
        )
        g_emb = t_emb + self.lattice_mlp(lat_feat)
        if cond_embedding is not None:
            g_emb = g_emb + cond_embedding
        h = h + g_emb[node_graph_id]

        # Edge / triplet features.
        y = self.edge_embedding(
            torch.cat([self.rbf(dist), self._fourier(df)], dim=-1)
        )
        if self.use_line_graph:
            allowed = _knn_mask(dist, dst, num_nodes, self.knn)
            lg_src, lg_dst = _line_graph_edges(
                src, dst, num_nodes, allowed=allowed
            )
            z = self.angle_embedding(torch_bond_cosines(r[lg_src], r[lg_dst]))
            for layer in self.alignn_layers:
                h, y, z = layer.forward_tensors(
                    src, dst, num_nodes, lg_src, lg_dst, y.shape[0], h, y, z
                )
        for layer in self.gcn_layers:
            h, y = layer.forward_tensors(src, dst, num_nodes, h, y)

        # Coordinate score: sum the fractional edge offsets into their
        # destination atom, each weighted by a learned per-edge scalar.
        w = self.edge_weight_mlp(
            torch.cat([h[src], h[dst], y], dim=-1)
        )  # (E, C)
        contrib = w.unsqueeze(-1) * df_min.unsqueeze(1)  # (E, C, 3)
        per_node = scatter_sum(contrib, dst, num_nodes)  # (N, C, 3)
        eps_frac = self.score_combine(per_node.transpose(1, 2)).squeeze(
            -1
        )  # (N, 3)

        pooled = scatter_mean(h, node_graph_id, int(natoms.shape[0]))
        eps_lattice = self.lattice_head(pooled)
        return {"eps_frac": eps_frac, "eps_lattice": eps_lattice}
