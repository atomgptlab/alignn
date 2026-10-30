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

Two optional extensions turn that three-body information from a *feature* into
a *generative channel*.  Both default to off, so the original model is
recovered exactly by the default configuration.

``angle_diffusion``
    Adds a per-triplet head predicting the angular displacement the forward
    process introduced, trained with FoldingDiff's wrapped smooth-L1
    objective.  The head reads the line-graph feature ``z`` of the shared
    backbone, and ``z`` reaches the coordinate and lattice heads through
    ALIGNN's ordinary ``angles -> bonds -> atoms`` path, so the angular
    channel is coupled rather than merely supervised.  ``angle_feedback=False``
    cuts that coupling for the control ablation.

``topology="radius"``
    Replaces the hard k-nearest-neighbour rule that decides which bonds may
    form triplets with a radius candidate set plus a DimeNet cutoff envelope
    and a ReaxFF-style product gate, so the line graph changes continuously as
    the coordinates denoise instead of jumping when two neighbours swap rank.

See :mod:`alignn.inverse.angles` for the literature these follow and for the
one place where this deviates from FoldingDiff.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from alignn.models.alignn_atomwise_pure import scatter_mean, scatter_sum
from alignn.models.alignn_atomwise_pure_smooth import (
    CutoffPolynomial,
    FourierAngular,
)
from alignn.models.utils import MLPLayer, RBFExpansion
from alignn.torch_graph_builder import _line_graph_edges, torch_bond_cosines

from alignn.inverse.angles import bond_angle, triplet_relevance
from alignn.inverse.diffusion import wrap_diff
from alignn.inverse.layers import (
    WeightedALIGNNConv,
    WeightedEdgeGatedGraphConv,
)

#: Line-graph topologies.  ``knn`` is the original hard neighbour-rank rule;
#: ``radius`` is the smooth construction of section 4 of the design brief.
TOPOLOGIES = ("knn", "radius")

#: Angular input bases.  ``cosine_rbf`` is ALIGNN's own; ``fourier`` is the
#: learnable Fourier basis on theta already shipped in this repository, and is
#: reserved for the A6 basis ablation.
ANGLE_BASES = ("cosine_rbf", "fourier")


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


def _angle_basis_layers(kind, triplet_bins, embedding_features, hidden):
    """Layers expanding a bond-angle cosine to a hidden-size feature.

    ``cosine_rbf`` is ALIGNN's own representation and is left byte-for-byte
    as it was; ``fourier`` swaps in the learnable Fourier basis on theta that
    this repository already carries, and exists only for the A6 ablation.
    DimeNet's joint spherical Fourier-Bessel distance-angle basis is *not*
    implemented here — see the ablation notes.
    """
    if kind == "fourier":
        order = max(1, (triplet_bins - 1) // 2)
        basis = FourierAngular(order=order)
        n_in = basis.out_features
    else:
        basis = RBFExpansion(vmin=-1.0, vmax=1.0, bins=triplet_bins)
        n_in = triplet_bins
    return [
        basis,
        MLPLayer(n_in, embedding_features),
        MLPLayer(embedding_features, hidden),
    ]


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
        angle_diffusion: bool = False,
        angle_feedback: bool = True,
        topology: str = "knn",
        radius_cutoff: float = 5.0,
        envelope_exponent: int = 5,
        gate_pair_messages: bool = False,
        angle_basis: str = "cosine_rbf",
    ):
        """Build the denoiser.

        Parameters beyond the original set, all defaulting to the original
        behaviour:

        angle_diffusion
            Emit an angular denoising prediction per triplet.  Requires
            ``alignn_layers > 0``, since the line graph is what carries
            angles.
        angle_feedback
            Whether the angular features are allowed to reach the bond (and
            hence atom, coordinate and lattice) representations.  ``False`` is
            ablation A4: angular supervision on a shared trunk with the
            architectural coupling removed.
        topology
            ``"knn"`` keeps the original rule — a bond may join a triplet if
            it is among the ``knn`` shortest bonds at its destination atom.
            ``"radius"`` replaces it with every bond shorter than
            ``radius_cutoff``, each weighted by the DimeNet envelope, with
            triplets weighted by the product of their two bonds' weights.
        radius_cutoff, envelope_exponent
            Cutoff radius and polynomial order of that envelope.  The default
            5 A sits between this repository's own three-body cutoff (3.5 A)
            and DimeNet's molecular cutoff (5 A), and is close to the radius
            the baseline's 12 nearest neighbours actually span in a crystal,
            which keeps the A1-vs-A3 comparison fair.
        gate_pair_messages
            Also weight the *pair* channel — the atom-graph messages and the
            per-edge terms of the coordinate score — by ``s_ij``.  The pair
            graph is dense rather than neighbour-ranked, so nothing is ever
            inserted or deleted there and this is not needed for continuity;
            it is the fuller reading of "smoothly vanishing pair
            interactions" and is switched on by the smooth-topology
            ablations.
        angle_basis
            ``"cosine_rbf"`` is ALIGNN's own angular representation and is
            what every primary experiment uses.  ``"fourier"`` is reserved
            for the A6 basis ablation.
        """
        super().__init__()
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {TOPOLOGIES}, got {topology!r}"
            )
        if angle_basis not in ANGLE_BASES:
            raise ValueError(
                f"angle_basis must be one of {ANGLE_BASES}, "
                f"got {angle_basis!r}"
            )
        if angle_diffusion and alignn_layers <= 0:
            raise ValueError(
                "angle_diffusion needs alignn_layers > 0: the angular "
                "channel lives on the line graph, which is not built when "
                "there are no ALIGNN layers"
            )
        if gate_pair_messages and topology != "radius":
            raise ValueError(
                "gate_pair_messages requires topology='radius'; the gate is "
                "the radius envelope"
            )
        self.hidden_features = hidden_features
        self.fourier_k = fourier_k
        self.knn = knn
        self.num_steps = num_steps
        self.angle_diffusion = angle_diffusion
        self.angle_feedback = angle_feedback
        self.topology = topology
        self.radius_cutoff = radius_cutoff
        self.gate_pair_messages = gate_pair_messages
        self.angle_basis = angle_basis

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
                *_angle_basis_layers(
                    angle_basis,
                    triplet_bins,
                    embedding_features,
                    hidden_features,
                )
            )
            if self.use_line_graph
            else None
        )
        # DimeNet's polynomial cutoff envelope, already implemented in this
        # repository for the smooth property model; u, u' and u'' all vanish
        # at the cutoff.
        self.envelope = (
            CutoffPolynomial(
                cutoff=radius_cutoff, coeff=float(envelope_exponent)
            )
            if topology == "radius"
            else None
        )

        self.alignn_layers = nn.ModuleList(
            [
                WeightedALIGNNConv(hidden_features, hidden_features)
                for _ in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                WeightedEdgeGatedGraphConv(hidden_features, hidden_features)
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
        # Angular denoising head. Reads the line-graph feature of the shared
        # backbone, so nothing about it is a second network: it is one more
        # output head on the representation that already denoises coordinates
        # and lattice.
        self.angle_head = (
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, 1),
            )
            if angle_diffusion
            else None
        )

        # Start from a near-zero prediction: diffusion training is much better
        # behaved when the model does not begin by shouting.
        nn.init.zeros_(self.score_combine.weight)
        nn.init.zeros_(self.lattice_head[-1].weight)
        nn.init.zeros_(self.lattice_head[-1].bias)
        if self.angle_head is not None:
            nn.init.zeros_(self.angle_head[-1].weight)
            nn.init.zeros_(self.angle_head[-1].bias)

    # ── geometry ─────────────────────────────────────────────────────────
    def _edge_geometry(
        self,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_graph_id: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return wrapped Δf, min-image Δf, Cartesian vec, dist, image.

        ``image`` is the integer cell offset ``n`` this resolution settled on,
        i.e. the one for which ``Δf = f[dst] - f[src] + n``.  The angular
        target needs it: re-applying the same ``n`` to the *clean* structure
        is what makes the target the corruption of one fixed triplet identity
        instead of a comparison between two different neighbours.
        """
        raw = frac[dst] - frac[src]
        df = wrap_diff(raw)
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
        # cand = (raw - round(raw)) + offset, so the total integer shift is:
        image = offsets[best] - torch.round(raw)  # (E, 3)
        return (
            df,
            df_min,
            r,
            d_all.gather(1, best.view(-1, 1)).squeeze(1),
            image,
        )

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

        With ``angle_diffusion`` on, the returned dict carries an extra
        ``"angle"`` entry holding the per-triplet prediction and everything
        the loss needs to build its target: the angles at the *noised*
        geometry, the triplet relevance weights, and the edge/triplet indices
        together with the periodic images the geometry was resolved against.
        """
        num_nodes = frac.shape[0]
        if pair_index is None:
            pair_index = dense_pair_index(natoms)
        src, dst, edge_graph_id = pair_index

        df, df_min, r, dist, image = self._edge_geometry(
            frac, lattice, src, dst, edge_graph_id
        )

        # Smooth pair relevance s_ij = u(r_ij; r_c). Recomputed from the
        # current coordinates and lattice on every call, which is what makes
        # the topology follow the geometry through reverse diffusion rather
        # than being fixed up front.
        s_edge = None if self.envelope is None else self.envelope(dist)
        edge_w = s_edge if self.gate_pair_messages else None

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
        angle_out: Optional[Dict[str, torch.Tensor]] = None
        if self.use_line_graph:
            if s_edge is None:
                # Original topology: a bond may join a triplet if it is among
                # the k shortest at its destination atom.
                allowed = _knn_mask(dist, dst, num_nodes, self.knn)
            else:
                # Radius candidate set. s_ij is exactly zero at and beyond
                # r_c, so dropping those bonds removes only terms that
                # already contributed nothing.
                allowed = s_edge > 0.0
            lg_src, lg_dst = _line_graph_edges(
                src, dst, num_nodes, allowed=allowed
            )
            # ReaxFF-style product gate: an angle fades out when either of
            # its two bonds does.
            tri_w = (
                None
                if s_edge is None
                else triplet_relevance(s_edge, lg_src, lg_dst)
            )
            cos_theta = torch_bond_cosines(r[lg_src], r[lg_dst])
            z = self.angle_embedding(cos_theta)
            # A4: keep the angular features evolving and supervised, but stop
            # them from reaching the bond representation.
            conv_tri_w = tri_w
            if not self.angle_feedback:
                conv_tri_w = torch.zeros_like(cos_theta)
            for layer in self.alignn_layers:
                h, y, z = layer.forward_tensors(
                    src,
                    dst,
                    num_nodes,
                    lg_src,
                    lg_dst,
                    y.shape[0],
                    h,
                    y,
                    z,
                    edge_w,
                    conv_tri_w,
                )
            if self.angle_head is not None:
                angle_out = {
                    "eps": self.angle_head(z).squeeze(-1),
                    "theta_t": bond_angle(r[lg_src], r[lg_dst]),
                    "weight": (
                        torch.ones_like(cos_theta) if tri_w is None else tri_w
                    ),
                    "lg_src": lg_src,
                    "lg_dst": lg_dst,
                    "src": src,
                    "dst": dst,
                    "image": image,
                    "edge_graph_id": edge_graph_id,
                }
        for layer in self.gcn_layers:
            h, y = layer.forward_tensors(src, dst, num_nodes, h, y, edge_w)

        # Coordinate score: sum the fractional edge offsets into their
        # destination atom, each weighted by a learned per-edge scalar.
        w = self.edge_weight_mlp(
            torch.cat([h[src], h[dst], y], dim=-1)
        )  # (E, C)
        if edge_w is not None:
            # Same continuity requirement as the messages: a pair leaving the
            # cutoff must stop contributing smoothly, not abruptly.
            w = w * edge_w.view(-1, 1)
        contrib = w.unsqueeze(-1) * df_min.unsqueeze(1)  # (E, C, 3)
        per_node = scatter_sum(contrib, dst, num_nodes)  # (N, C, 3)
        eps_frac = self.score_combine(per_node.transpose(1, 2)).squeeze(
            -1
        )  # (N, 3)

        pooled = scatter_mean(h, node_graph_id, int(natoms.shape[0]))
        eps_lattice = self.lattice_head(pooled)
        out: Dict = {"eps_frac": eps_frac, "eps_lattice": eps_lattice}
        if angle_out is not None:
            out["angle"] = angle_out
        return out
