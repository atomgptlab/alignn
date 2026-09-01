"""ALIGNN convolutions with an optional per-edge / per-triplet weight.

These are thin subclasses of the shared pure-torch ALIGNN layers in
:mod:`alignn.models.alignn_atomwise_pure`.  They exist so that the diffusion
denoiser can attenuate a message continuously instead of a graph edge simply
being present or absent, without touching the property-prediction models.

Parameter names and shapes are *identical* to the classes they subclass
(``node_update.*`` / ``edge_update.*``), so a checkpoint trained with the
stock layers loads into these and vice versa.

How the weight enters
---------------------
The edge-gated convolution aggregates a normalised, gated average

    h_i = sum_j sigma_ij * Bh_j / sum_j sigma_ij .

A weight ``w_ij`` is applied to ``sigma_ij`` *before both* sums.  That is the
only placement with the property we need: an edge with ``w = 0`` leaves ``h``
exactly as if the edge had never been in the list, so inserting or deleting it
at the cutoff produces no jump.  Scaling only the numerator would instead
renormalise the surviving messages and would not be continuous.  The
normalisation ``bn_nodes`` / ``bn_edges`` is ``LayerNorm``, computed per
element, so no cross-edge statistic can smuggle a discontinuity back in.

The same class is used in both roles ALIGNN gives it — over the atom graph the
weight is a per-pair relevance ``s_ij``; over the line graph the same code
receives a per-triplet relevance ``s_ijk`` — which is why no separate triplet
machinery is needed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from alignn.models.alignn_atomwise_pure import (
    EdgeGatedGraphConvPure,
    scatter_sum,
)

__all__ = ["WeightedEdgeGatedGraphConv", "WeightedALIGNNConv"]


class WeightedEdgeGatedGraphConv(EdgeGatedGraphConvPure):
    """:class:`EdgeGatedGraphConvPure` with an optional per-edge weight."""

    def forward_tensors(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identical to the base layer when ``edge_weight`` is ``None``."""
        e_src = self.src_gate(x)
        e_dst = self.dst_gate(x)
        m = e_src[src] + e_dst[dst] + self.edge_gate(y)
        sigma = torch.sigmoid(m)
        if edge_weight is not None:
            sigma = sigma * edge_weight.view(-1, 1)

        Bh = self.dst_update(x)
        msg_h = Bh[src] * sigma
        sum_sigma_h = scatter_sum(msg_h, dst, num_nodes)
        sum_sigma = scatter_sum(sigma, dst, num_nodes)
        h = sum_sigma_h / (sum_sigma + 1e-6)
        x_new = self.src_update(x) + h

        x_new = F.silu(self.bn_nodes(x_new))
        y_new = F.silu(self.bn_edges(m))

        if self.residual:
            x_new = x + x_new
            y_new = y + y_new
        return x_new, y_new


class WeightedALIGNNConv(nn.Module):
    """ALIGNN layer whose pair and triplet messages can both be weighted.

    Mirrors :class:`alignn.models.alignn_atomwise_pure.ALIGNNConvPure` exactly
    — the line-graph convolution updates bond features, which the atom-graph
    convolution then uses — with two extra optional arguments.

    ``triplet_weight = 0`` for every triplet is what ablation A4 uses: the
    angular features ``z`` still evolve and still feed the angle head, but they
    no longer reach the bond features, so the coordinate/lattice pathway sees
    no angular information.  The angular loss then acts as a pure auxiliary
    task on a shared trunk, which is the control the design brief asks for.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.node_update = WeightedEdgeGatedGraphConv(
            in_features, out_features
        )
        self.edge_update = WeightedEdgeGatedGraphConv(
            out_features, out_features
        )

    def forward_tensors(
        self,
        g_src: torch.Tensor,
        g_dst: torch.Tensor,
        g_num_nodes: int,
        lg_src: torch.Tensor,
        lg_dst: torch.Tensor,
        lg_num_nodes: int,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        triplet_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, m = self.node_update.forward_tensors(
            g_src, g_dst, g_num_nodes, x, y, edge_weight
        )
        y, z = self.edge_update.forward_tensors(
            lg_src, lg_dst, lg_num_nodes, m, z, triplet_weight
        )
        return x, y, z
