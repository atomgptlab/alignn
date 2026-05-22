"""Pure-PyTorch ALIGNN-atomwise (no DGL).

This is a line-for-line reimplementation of ``ALIGNNAtomWise`` with all
DGL ops (``dgl.function``, ``dgl.nn``, ``dgl.batch``, ``dgl.reverse``)
replaced by index/scatter primitives on ``TorchGraph`` tensors. Same
config fields, same forward output keys — drop-in for training and
inference without depending on DGL at model-evaluation time.

Accepts either:
  * a ``(TorchGraph, TorchGraph, lattice)`` triple (preferred), or
  * a ``(DGLGraph, DGLGraph, lattice)`` triple — converted at the
    boundary via ``torchgraph_from_dgl`` so you can flip models without
    touching the dataloader.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

from alignn.models.utils import MLPLayer, RBFExpansion
from alignn.torch_graph_builder import TorchGraph, torchgraph_from_dgl
from alignn.utils import BaseSettings

# =====================================================================
# Primitives
# =====================================================================


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets indexed by ``index``.

    Script-safe: builds output shape as an explicit ``List[int]`` and
    broadcasts ``index`` via repeated ``unsqueeze`` (avoids unpacking).
    """
    out_shape: List[int] = [dim_size]
    for i in range(1, src.dim()):
        out_shape.append(src.size(i))
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    total = scatter_sum(src, index, dim_size)
    ones = torch.ones(src.shape[0], device=src.device, dtype=src.dtype)
    count = torch.zeros(
        dim_size, device=src.device, dtype=src.dtype
    ).scatter_add_(0, index, ones)
    denom = count.clamp_min(1.0)
    while denom.dim() < total.dim():
        denom = denom.unsqueeze(-1)
    return total / denom


# =====================================================================
# Config
# =====================================================================


class ALIGNNAtomWisePureConfig(BaseSettings):
    """Hyperparameter schema — mirrors ALIGNNAtomWiseConfig."""

    name: Literal["alignn_atomwise_pure"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 1
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = True
    # Accepted for config-schema compatibility with ALIGNNAtomWiseConfig;
    # the pure-torch forward doesn't implement this branch.
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 3.0
    stress_multiplier: float = 1.0
    add_reverse_forces: bool = True
    lg_on_fly: bool = True
    batch_stress: bool = True
    multiply_cutoff: bool = False
    use_penalty: bool = True
    extra_features: int = 0
    exponent: int = 5
    penalty_factor: float = 0.5
    penalty_threshold: float = 1.0
    # penalty_factor: float = 0.1
    # penalty_threshold: float = 1.0
    additional_output_features: int = 0
    additional_output_weight: float = 0.0
    # Attention variants (Shao et al., Adv. Theory Simul. 2026):
    #   "alignn"   - original edge-gated conv
    #   "n_alignn" - Node-Attention Layer (NAL), per-node learnable weights
    #   "t_alignn" - Self-Attention Layer (SAL), Transformer-style Q/K/V
    conv_type: Literal["alignn", "n_alignn", "t_alignn"] = "alignn"
    num_heads: int = 1

    class Config:
        env_prefix = "jv_model"


# =====================================================================
# Conv layers
# =====================================================================


class EdgeGatedGraphConvPure(nn.Module):
    """Edge-gated graph convolution, pure torch / scatter-based."""

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward_tensors(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Script-friendly core: takes plain tensors instead of a graph."""
        # Edge update: m = e_src[src] + e_dst[dst] + edge_gate(y)
        e_src = self.src_gate(x)
        e_dst = self.dst_gate(x)
        m = e_src[src] + e_dst[dst] + self.edge_gate(y)
        sigma = torch.sigmoid(m)

        # Node update aggregated at dst:
        # h = sum_{(i,j)} sigma * Bh[src] / (sum sigma + eps).
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

    @torch.jit.ignore
    def forward(
        self, g: TorchGraph, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_tensors(g.src, g.dst, g.num_nodes, x, y)


def _scatter_softmax(
    logits: torch.Tensor, dst: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Per-destination softmax over edges (pure torch, scatter-based)."""
    max_per_dst = torch.full(
        (num_nodes,) + logits.shape[1:],
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    idx = dst
    while idx.dim() < logits.dim():
        idx = idx.unsqueeze(-1)
    idx_e = idx.expand_as(logits)
    max_per_dst.scatter_reduce_(
        0, idx_e, logits, reduce="amax", include_self=True
    )
    shifted = logits - max_per_dst[dst]
    exp_l = torch.exp(shifted)
    sum_exp = scatter_sum(exp_l, dst, num_nodes)
    return exp_l / (sum_exp[dst] + 1e-8)


class NodeAttentionGraphConvPure(nn.Module):
    """Node-Attention Layer (NAL) — pure torch.

    Paper Eq. (8):
        m_ij = A_src_i * L_src h_i + sum_j (A_dst_j * L_dst h_j + L_e e_ij)
    with A_src, A_dst as per-node learned sigmoids over node features.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.output_features = output_features
        assert output_features % num_heads == 0
        self.head_dim = output_features // num_heads

        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

        # Per-node attention heads: sigmoid(fc(h)) -> [N, num_heads]
        self.attn_src = nn.Linear(input_features, num_heads)
        self.attn_dst = nn.Linear(input_features, num_heads)

    def _apply_attn(self, proj: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # proj: [N, F]; a: [N, heads]
        N = proj.shape[0]
        ph = proj.view(N, self.num_heads, self.head_dim)
        return (a.unsqueeze(-1) * ph).view(N, self.output_features)

    @torch.jit.ignore
    def forward(
        self, g: TorchGraph, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst, N = g.src, g.dst, g.num_nodes

        a_src = torch.sigmoid(self.attn_src(x))
        a_dst = torch.sigmoid(self.attn_dst(x))
        e_src = self._apply_attn(self.src_gate(x), a_src)
        e_dst = self._apply_attn(self.dst_gate(x), a_dst)

        m = e_src[src] + e_dst[dst] + self.edge_gate(y)
        sigma = torch.sigmoid(m)

        Bh = self.dst_update(x)
        sum_sigma_h = scatter_sum(Bh[src] * sigma, dst, N)
        sum_sigma = scatter_sum(sigma, dst, N)
        h = sum_sigma_h / (sum_sigma + 1e-6)
        x_new = self.src_update(x) + h

        x_new = F.silu(self.bn_nodes(x_new))
        y_new = F.silu(self.bn_edges(m))
        if self.residual:
            x_new = x + x_new
            y_new = y + y_new
        return x_new, y_new


class SelfAttentionGraphConvPure(nn.Module):
    """Self-Attention Layer (SAL) — pure torch.

    Paper Eqs. (9)-(11):
        score  = softmax((q_i . k_j . k_e) / sqrt(d))
        h_i'   = fc(h_i) + SiLU(Norm(sum_j score . v_j))
        e_ij'  = fc(e_ij) + SiLU(Norm(score))
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.output_features = output_features
        assert output_features % num_heads == 0
        self.head_dim = output_features // num_heads
        self.scale = self.head_dim**0.5

        self.W_q = nn.Linear(input_features, output_features)
        self.W_k = nn.Linear(input_features, output_features)
        self.W_v = nn.Linear(input_features, output_features)
        self.W_ke = nn.Linear(input_features, output_features)

        self.fc_node = nn.Linear(input_features, output_features)
        self.fc_edge = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)
        self.bn_edges = nn.LayerNorm(output_features)

    @torch.jit.ignore
    def forward(
        self, g: TorchGraph, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst, N = g.src, g.dst, g.num_nodes
        E = y.shape[0]
        H, D = self.num_heads, self.head_dim

        q = self.W_q(x).view(N, H, D)
        k = self.W_k(x).view(N, H, D)
        v = self.W_v(x).view(N, H, D)
        k_e = self.W_ke(y).view(E, H, D)

        # Attention logits per edge, per head: [E, H, 1]
        attn_logits = (q[dst] * k[src] * k_e).sum(dim=-1, keepdim=True) / (
            self.scale
        )
        # Softmax over incoming edges per destination, per head.
        attn = _scatter_softmax(attn_logits.squeeze(-1), dst, N).unsqueeze(
            -1
        )  # [E, H, 1]

        # Weighted value aggregation per destination node.
        weighted = (attn * v[src]).view(E, self.output_features)
        h_agg = scatter_sum(weighted, dst, N)
        x_new = self.fc_node(x) + F.silu(self.bn_nodes(h_agg))

        # Edge update uses the attention scores themselves (paper Eq. 11).
        score_e = attn.expand(E, H, D).reshape(E, self.output_features)
        y_new = self.fc_edge(y) + F.silu(self.bn_edges(score_e))
        return x_new, y_new


def _make_bond_conv(
    in_features: int,
    out_features: int,
    conv_type: str,
    num_heads: int,
) -> nn.Module:
    if conv_type == "alignn":
        return EdgeGatedGraphConvPure(in_features, out_features)
    if conv_type == "n_alignn":
        return NodeAttentionGraphConvPure(
            in_features, out_features, num_heads=num_heads
        )
    if conv_type == "t_alignn":
        return SelfAttentionGraphConvPure(
            in_features, out_features, num_heads=num_heads
        )
    raise ValueError(f"Unknown conv_type: {conv_type!r}")


class ALIGNNConvPure(nn.Module):
    """Line-graph-aware ALIGNN update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_type: str = "alignn",
        num_heads: int = 1,
    ):
        super().__init__()
        self.node_update = _make_bond_conv(
            in_features, out_features, conv_type, num_heads
        )
        self.edge_update = _make_bond_conv(
            out_features, out_features, conv_type, num_heads
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, m = self.node_update.forward_tensors(
            g_src, g_dst, g_num_nodes, x, y
        )
        y, z = self.edge_update.forward_tensors(
            lg_src, lg_dst, lg_num_nodes, m, z
        )
        return x, y, z

    @torch.jit.ignore
    def forward(
        self,
        g: TorchGraph,
        lg: TorchGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention convs (NAL/SAL) don't have forward_tensors — dispatch
        # through the graph-object forward, which handles all conv types.
        if not isinstance(self.node_update, EdgeGatedGraphConvPure):
            x, m = self.node_update(g, x, y)
            y, z = self.edge_update(lg, m, z)
            return x, y, z
        return self.forward_tensors(
            g.src,
            g.dst,
            g.num_nodes,
            lg.src,
            lg.dst,
            lg.num_nodes,
            x,
            y,
            z,
        )


# =====================================================================
# Helpers
# =====================================================================


def cutoff_function_based_edges(
    r: torch.Tensor, inner_cutoff: float = 4.0, exponent: int = 3
) -> torch.Tensor:
    ratio = r / inner_cutoff
    c1 = -(exponent + 1) * (exponent + 2) / 2
    c2 = exponent * (exponent + 2)
    c3 = -exponent * (exponent + 1) / 2
    envelope = (
        1.0
        + c1 * ratio**exponent
        + c2 * ratio ** (exponent + 1)
        + c3 * ratio ** (exponent + 2)
    )
    return torch.where(r <= inner_cutoff, envelope, torch.zeros_like(r))


def _bond_cosines(r_ij: torch.Tensor, r_jk: torch.Tensor) -> torch.Tensor:
    num = -(r_ij * r_jk).sum(dim=-1)
    denom = r_ij.norm(dim=-1) * r_jk.norm(dim=-1)
    return (num / denom.clamp_min(1e-12)).clamp(-1.0, 1.0)


def _as_torchgraph(x):
    """Accept TorchGraph or DGLGraph transparently."""
    if isinstance(x, TorchGraph):
        return x
    return torchgraph_from_dgl(x)


# =====================================================================
# Model
# =====================================================================


class ALIGNNAtomWisePure(nn.Module):
    """DGL-free ALIGNN atomwise."""

    def __init__(
        self,
        config: ALIGNNAtomWisePureConfig = ALIGNNAtomWisePureConfig(
            name="alignn_atomwise_pure"
        ),
    ):
        super().__init__()
        self.config = config
        self.classification = config.classification
        if self.config.gradwise_weight == 0:
            self.config.calculate_gradient = False

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConvPure(
                    config.hidden_features,
                    config.hidden_features,
                    conv_type=config.conv_type,
                    num_heads=config.num_heads,
                )
                for _ in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                _make_bond_conv(
                    config.hidden_features,
                    config.hidden_features,
                    config.conv_type,
                    config.num_heads,
                )
                for _ in range(config.gcn_layers)
            ]
        )
        if config.atomwise_output_features > 0:
            self.fc_atomwise = nn.Linear(
                config.hidden_features, config.atomwise_output_features
            )
        if config.additional_output_features:
            self.fc_additional_output = nn.Linear(
                config.hidden_features, config.additional_output_features
            )
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        # Link is applied inline in the forward paths (kept as a string so
        # `torch.jit.script` doesn't choke on a Python lambda attribute).
        self.link_name: str = config.link
        if config.link == "log":
            self.fc.bias.data = torch.tensor(np.log(0.7), dtype=torch.float)

        # Species feature lookup for LAMMPS / TorchScript entry point.
        # Registered empty here; ``register_species_table`` populates it
        # before export. Stays unused on the training path.
        self.register_buffer(
            "_species_table",
            torch.zeros(120, config.atom_input_features, dtype=torch.float32),
        )

        # Expose scalar config fields that `forward_tensors` reads as
        # typed Python attributes — the pydantic ``self.config`` object
        # is not TorchScript-compatible.
        self.energy_mult_natoms: bool = bool(config.energy_mult_natoms)
        self.use_penalty: bool = bool(config.use_penalty)
        self.penalty_threshold: float = float(config.penalty_threshold)
        self.penalty_factor: float = float(config.penalty_factor)
        self.grad_multiplier: int = int(config.grad_multiplier)
        self.add_reverse_forces: bool = bool(config.add_reverse_forces)
        self.stress_multiplier: float = float(config.stress_multiplier)

    # ----- species lookup (LAMMPS hands us atomic numbers, not features) -----

    @torch.jit.ignore
    def register_species_table(
        self, atom_features: str = "cgcnn", max_z: int = 119
    ) -> None:
        """Build a (max_z+1, F) feature table indexed by atomic number.

        Once registered, ``forward_tensors_z`` can be used from TorchScript
        so callers (e.g. LAMMPS) don't need to ship the feature table
        alongside the model. Invalid / unknown atomic numbers map to
        zero vectors.
        """
        from ase.data import chemical_symbols
        from jarvis.core.specie import get_node_attributes

        F = int(self.config.atom_input_features)
        rows = np.zeros((max_z + 1, F), dtype=np.float32)
        for z in range(1, max_z + 1):
            if z >= len(chemical_symbols):
                continue
            symbol = chemical_symbols[z]
            try:
                feat = np.asarray(
                    get_node_attributes(symbol, atom_features=atom_features),
                    dtype=np.float32,
                )
                if feat.shape[0] != F:
                    continue
                rows[z] = feat
            except Exception:
                continue
        # Buffer was allocated in __init__; overwrite the tensor contents
        # (register_buffer would complain about duplicate registration).
        self._species_table.resize_(rows.shape).copy_(
            torch.as_tensor(rows, dtype=torch.float32)
        )

    # ----- tensor-only forward (LAMMPS / TorchScript path) -----

    @torch.jit.export
    def forward_tensors_z(
        self,
        positions: torch.Tensor,  # (N, 3), requires_grad=True for forces
        lattice: torch.Tensor,  # (3, 3)
        atomic_numbers: torch.Tensor,  # (N,) long
        src: torch.Tensor,  # (E,) long
        dst: torch.Tensor,  # (E,) long
        shift: torch.Tensor,  # (E, 3)
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward with internal atomic-number → feature lookup.

        Same as ``forward_tensors`` but accepts atomic numbers directly.
        Requires ``register_species_table`` to have been called
        beforehand. LAMMPS-style callers only ship atomic numbers.
        """
        atom_features = self._species_table.index_select(0, atomic_numbers)
        return self.forward_tensors(
            positions, lattice, atom_features, src, dst, shift, compute_stress
        )

    @torch.jit.export
    def forward_tensors(
        self,
        positions: torch.Tensor,  # (N, 3), requires_grad=True for forces
        lattice: torch.Tensor,  # (3, 3), requires_grad=True for stress
        atom_features: torch.Tensor,  # (N, atom_input_features)
        src: torch.Tensor,  # (E,) long
        dst: torch.Tensor,  # (E,) long
        shift: torch.Tensor,  # (E, 3) integer cell offsets (float dtype)
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Single-system forward driven by plain tensors.

        Intended for LAMMPS-style MD where each step supplies an
        atom list, lattice, and neighbor list. No batching, no class
        pooling machinery. Returns ``{"energy", "forces", ["stress"]}``.
        """
        num_nodes = positions.shape[0]

        # Differentiable edge vectors.
        r = (
            positions.index_select(0, dst)
            - positions.index_select(0, src)
            + torch.matmul(shift, lattice)
        )
        bondlength = torch.linalg.vector_norm(r, dim=1)

        # Inline line-graph: connect parent edge A=(u,v) to B=(v,w).
        E = src.shape[0]
        order = torch.argsort(src, stable=True)
        sorted_src = src.index_select(0, order)
        node_range = torch.arange(num_nodes, device=src.device)
        bucket_start = torch.searchsorted(sorted_src, node_range)
        bucket_end = torch.searchsorted(sorted_src, node_range, right=True)

        A_v = dst
        starts = bucket_start.index_select(0, A_v)
        ends = bucket_end.index_select(0, A_v)
        counts = ends - starts

        total = int(counts.sum().item())
        A_ids = torch.arange(E, device=src.device)
        lg_src = torch.repeat_interleave(A_ids, counts)
        cum = torch.cumsum(counts, dim=0)
        row_start = cum - counts
        offsets = torch.arange(
            total, device=src.device
        ) - torch.repeat_interleave(row_start, counts)
        pos_idx = torch.repeat_interleave(starts, counts) + offsets
        lg_dst = order.index_select(0, pos_idx)
        lg_num_nodes = E

        # Angle cosines (differentiable through r -> positions, lattice).
        r_ij = r.index_select(0, lg_src)
        r_jk = r.index_select(0, lg_dst)
        num = -(r_ij * r_jk).sum(dim=-1)
        denom = torch.linalg.vector_norm(r_ij, dim=-1).clamp_min(
            1e-12
        ) * torch.linalg.vector_norm(r_jk, dim=-1).clamp_min(1e-12)
        h_cos = (num / denom).clamp(-1.0, 1.0)

        # Embeddings.
        x = self.atom_embedding(atom_features)
        y = self.edge_embedding(bondlength)
        z = self.angle_embedding(h_cos)

        # ALIGNN and GCN layers via tensor-only paths.
        for layer in self.alignn_layers:
            x, y, z = layer.forward_tensors(
                src, dst, num_nodes, lg_src, lg_dst, lg_num_nodes, x, y, z
            )
        for layer in self.gcn_layers:
            x, y = layer.forward_tensors(src, dst, num_nodes, x, y)

        # Mean readout (single graph) and final projection.
        h_graph = x.mean(dim=0, keepdim=True)  # (1, hidden)
        out = self.fc(h_graph).squeeze(-1)  # (1,) for output_features=1

        en = out
        if self.energy_mult_natoms:
            en = out * float(num_nodes)

        if self.use_penalty:
            pen = torch.where(
                bondlength < self.penalty_threshold,
                self.penalty_factor * (self.penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            en = en + pen.sum()

        result: Dict[str, torch.Tensor] = {"energy": en.squeeze()}

        # Autograd forces through r -> positions.
        grad_outs = torch.autograd.grad(
            outputs=[en.sum()],
            inputs=[r],
            create_graph=self.training,
            retain_graph=True,
        )
        pair_forces = grad_outs[0]
        assert pair_forces is not None
        pair_forces = self.grad_multiplier * pair_forces

        # Atom forces = -dE/dx = sum of dE/dr over incoming edges
        # minus sum over outgoing (add_reverse_forces convention).
        forces_ji = scatter_sum(pair_forces, dst, num_nodes)
        if self.add_reverse_forces:
            forces_ij = scatter_sum(pair_forces, src, num_nodes)
            forces = forces_ji - forces_ij
        else:
            forces = forces_ji
        result["forces"] = forces

        if compute_stress:
            # Virial stress: -160.217 * r.T @ pair_forces / V.
            V = torch.abs(torch.det(lattice))
            stress = -160.21766208 * (torch.matmul(r.T, pair_forces) / V)
            result["stress"] = self.stress_multiplier * stress

        return result

    # ----- dataclass-based forward (training pipeline) -----

    @torch.jit.ignore
    def forward(self, g):
        # Unpack and normalize input.
        gg, lg, lat = g
        g = _as_torchgraph(gg)
        lg = _as_torchgraph(lg)

        r = g.edata["r"]
        if self.config.calculate_gradient:
            r.requires_grad_(True)

        bondlength = torch.norm(r, dim=1)

        # Recompute angle cosines inside the autograd graph when lg_on_fly.
        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            h = _bond_cosines(r[lg.src], r[lg.dst])
            z = self.angle_embedding(h)
        elif len(self.alignn_layers) > 0:
            z = self.angle_embedding(lg.edata["h"])
        else:
            z = None  # unused

        x = self.atom_embedding(g.ndata["atom_features"])

        if self.config.use_cutoff_function:
            if self.config.multiply_cutoff:
                c_off = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                ).unsqueeze(-1)
                y = self.edge_embedding(bondlength) * c_off
            else:
                bondlength_eff = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                )
                y = self.edge_embedding(bondlength_eff)
        else:
            y = self.edge_embedding(bondlength)

        for layer in self.alignn_layers:
            x, y, z = layer(g, lg, x, y, z)
        for layer in self.gcn_layers:
            x, y = layer(g, x, y)

        # Per-graph readout (mean over nodes).
        node_bid = g.node_batch_id
        h_graph = scatter_mean(x, node_bid, g.batch_size)
        out = self.fc(h_graph)
        if self.config.output_features == 1:
            out = out.squeeze(-1)

        natoms = (
            g.batch_num_nodes
            if g.batch_num_nodes is not None
            else torch.tensor([g.num_nodes], device=r.device)
        )
        en_out = out
        if self.config.energy_mult_natoms:
            en_out = out * natoms.to(out.dtype)

        if self.config.use_penalty:
            penalties = torch.where(
                bondlength < self.config.penalty_threshold,
                self.config.penalty_factor
                * (self.config.penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            en_out = en_out + penalties.sum()

        atomwise_pred = torch.empty(1, device=r.device)
        if (
            self.config.atomwise_output_features > 0
            and self.config.atomwise_weight != 0
        ):
            atomwise_pred = self.fc_atomwise(x)

        additional_out = torch.empty(1, device=r.device)
        if self.config.additional_output_features > 0:
            additional_out = self.fc_additional_output(h_graph)

        forces = torch.empty(1, device=r.device)
        stress = torch.empty(1, device=r.device)
        if self.config.calculate_gradient:
            pair_forces = (
                self.config.grad_multiplier
                * grad(
                    en_out.sum(),
                    r,
                    grad_outputs=torch.ones_like(en_out.sum()),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            if self.config.force_mult_natoms:
                pair_forces = pair_forces * g.num_nodes

            # force_i from incoming edges (dst == i): sum pair_forces at dst.
            forces_ji = scatter_sum(pair_forces, g.dst, g.num_nodes)
            if self.config.add_reverse_forces:
                forces_ij = scatter_sum(pair_forces, g.src, g.num_nodes)
                forces = (forces_ji - forces_ij).squeeze()
            else:
                forces = forces_ji.squeeze()

            if self.config.stresswise_weight != 0:
                # Per-graph stress = -160.21766208 * (r.T @ pair_forces) / V.
                # V is stored per-node; take V at the first node of each graph.
                B = g.batch_size
                V_per_node = g.ndata["V"]
                if g.batch_num_nodes is not None:
                    node_offsets = torch.zeros(
                        B, dtype=torch.long, device=r.device
                    )
                    node_offsets[1:] = torch.cumsum(
                        g.batch_num_nodes[:-1], dim=0
                    )
                    edge_offsets = torch.zeros(
                        B, dtype=torch.long, device=r.device
                    )
                    edge_offsets[1:] = torch.cumsum(
                        g.batch_num_edges[:-1], dim=0
                    )
                else:
                    node_offsets = torch.zeros(
                        1, dtype=torch.long, device=r.device
                    )
                    edge_offsets = torch.zeros(
                        1, dtype=torch.long, device=r.device
                    )

                stresses = []
                for b in range(B):
                    e0 = int(edge_offsets[b].item())
                    e1 = e0 + (
                        int(g.batch_num_edges[b].item())
                        if g.batch_num_edges is not None
                        else g.num_edges
                    )
                    n0 = int(node_offsets[b].item())
                    V_b = V_per_node[n0]
                    st = -160.21766208 * (
                        torch.matmul(r[e0:e1].T, pair_forces[e0:e1]) / V_b
                    )
                    stresses.append(st)
                stress = self.config.stress_multiplier * torch.stack(stresses)

        if self.link_name == "log":
            out = torch.exp(out)
        elif self.link_name == "logit":
            out = torch.sigmoid(out)
        if self.classification:
            if out.dim() == 1:
                out = out.unsqueeze(0)
            out = self.softmax(out)

        return {
            "out": out,
            "additional": additional_out,
            "grad": forces,
            "stresses": stress,
            "atomwise_pred": atomwise_pred,
        }
