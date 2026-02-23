"""Atomistic LIne Graph Neural Network with Attention Mechanisms."""

from typing import Tuple, Union, Literal, Optional
from torch.autograd import grad
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import AvgPooling
import torch
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import (
    RBFExpansion,
    compute_cartesian_coordinates,
    compute_pair_vector_and_distance,
    MLPLayer,
)
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for ALIGNN with attention mechanisms."""

    name: Literal["alignn_atomwise"]
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
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 3  # Angstrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = True
    lg_on_fly: bool = True
    batch_stress: bool = True
    multiply_cutoff: bool = False
    use_penalty: bool = True
    extra_features: int = 0
    exponent: int = 5
    penalty_factor: float = 0.1
    penalty_threshold: float = 1
    additional_output_features: int = 0
    additional_output_weight: float = 0
    conv_type: Literal["alignn", "n_alignn", "t_alignn"] = "alignn"
    # Number of attention heads for T-ALIGNN (SAL) multi-head attention
    # Also supported for N-ALIGNN
    num_heads: int = 1

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


# ---------------------------------------------------------------------------
# Cutoff functions (unchanged from original)
# ---------------------------------------------------------------------------


def cutoff_function_based_edges(r, inner_cutoff=4, exponent=3):
    """Apply smooth cutoff to pairwise interactions.

    Args:
        r: bond lengths
        inner_cutoff: cutoff radius
        exponent: polynomial exponent for envelope

    Returns:
        Smooth cutoff values, zero outside inner_cutoff.
    """
    ratio = r / inner_cutoff
    c1 = -(exponent + 1) * (exponent + 2) / 2
    c2 = exponent * (exponent + 2)
    c3 = -exponent * (exponent + 1) / 2
    envelope = (
        1
        + c1 * ratio**exponent
        + c2 * ratio ** (exponent + 1)
        + c3 * ratio ** (exponent + 2)
    )
    return torch.where(r <= inner_cutoff, envelope, torch.zeros_like(r))


# ---------------------------------------------------------------------------
# Utility: per-destination softmax over edges (DGL-version-agnostic)
# ---------------------------------------------------------------------------


def _edge_softmax(g, logits):
    """Compute softmax of edge logits grouped by destination node.

    Works across all DGL versions without relying on
    dgl.nn.functional.edge_softmax.

    Args:
        g: DGLGraph
        logits: Tensor of shape [num_edges, *] with raw scores.

    Returns:
        Tensor of same shape as logits with softmax applied per dst node.
    """
    try:
        from dgl.nn.functional import edge_softmax

        return edge_softmax(g, logits)
    except ImportError:
        pass

    # Manual fallback: group by destination node
    _, dst = g.edges()
    # Compute max per destination for numerical stability
    max_logits = torch.zeros(
        g.num_nodes(),
        *logits.shape[1:],
        device=logits.device,
        dtype=logits.dtype,
    )
    max_logits.scatter_reduce_(
        0,
        dst.unsqueeze(-1).expand_as(logits),
        logits,
        reduce="amax",
        include_self=True,
    )
    logits_shifted = logits - max_logits[dst]
    exp_logits = torch.exp(logits_shifted)

    sum_exp = torch.zeros(
        g.num_nodes(),
        *logits.shape[1:],
        device=logits.device,
        dtype=logits.dtype,
    )
    sum_exp.scatter_add_(
        0, dst.unsqueeze(-1).expand_as(exp_logits), exp_logits
    )

    return exp_logits / (sum_exp[dst] + 1e-8)


# ---------------------------------------------------------------------------
# Original ALIGNN: Edge-Gated Graph Convolution
# ---------------------------------------------------------------------------


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    This is the original convolution layer used in ALIGNN [1].
    Edge features go into the soft attention / edge gating function,
    and the primary node update function is W cat(u, v) + b.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # Gate computation: z_ij = cat(v_i, v_j, u_ij)
        # m_ij = sigma(z_ij W_f + b_f) * g_s(z_ij W_s + b_s)
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Edge-gated graph convolution.

        h_i^{l+1} = ReLU(U h_i + sum_{j->i} eta_{ij} . V h_j)
        """
        g = g.local_var()

        # Compute edge gates: Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # Node and edge updates with activation + normalization
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


# ---------------------------------------------------------------------------
# N-ALIGNN: Node-Attention Layer (NAL)
#   m_ij = A_src * L_i * h_i + sum_j(A_dst * L_j * h_j + L_e * e_ij)
# A_src and A_dst are learnable scalars that dynamically weight
# the importance of source and destination nodes.
# ---------------------------------------------------------------------------


class NodeAttentionGraphConv(nn.Module):
    """Node-Attention Layer (NAL) for N-ALIGNN.

    Extends EdgeGatedGraphConv by introducing learnable node-level
    attention parameters A_src and A_dst that dynamically weight the
    importance of different atoms during message aggregation.

    This allows the model to distinguish the contributions of different
    atoms in two-body interactions, consistent with the differences in
    atomic behaviors caused by different local chemical environments.

    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        num_heads: int = 1,
    ):
        """Initialize parameters for NAL update.

        Args:
            input_features: Input feature dimension.
            output_features: Output feature dimension.
            residual: Whether to use residual connections.
            num_heads: Number of attention heads. Paper shows NAL
                is relatively insensitive to this parameter.
        """
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.output_features = output_features

        if num_heads > 1:
            assert output_features % num_heads == 0, (
                f"output_features ({output_features}) must be divisible "
                f"by num_heads ({num_heads})"
            )
            self.head_dim = output_features // num_heads
        else:
            self.head_dim = output_features

        # Gate computation layers (same structure as EdgeGatedGraphConv)
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        # Node update layers
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

        # NEW: Learnable node-attention parameters (Eq. 8 in paper)
        # These scalar parameters weight the importance of source vs
        # destination nodes during message aggregation
        if num_heads > 1:
            # Per-head attention scalars
            self.src_attention = nn.Parameter(torch.ones(num_heads, 1))
            self.dst_attention = nn.Parameter(torch.ones(num_heads, 1))
        else:
            self.src_attention = nn.Parameter(torch.ones(1))
            self.dst_attention = nn.Parameter(torch.ones(1))

        # For multi-head: projection to combine heads back
        if num_heads > 1:
            self.fc_node = nn.Linear(output_features, output_features)
            self.fc_edge = nn.Linear(output_features, output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Node-Attention graph convolution.

        Eq. (8): m_ij = A_src * L_i * h_i
                      + sum(A_dst * L_j * h_j + L_e * e_ij)

        The key difference from EdgeGatedGraphConv is that A_src and
        A_dst scale the source and destination node contributions
        before gating.
        """
        g = g.local_var()

        # Compute gated edge messages with node attention
        src_proj = self.src_gate(node_feats)  # L_i * h_i
        dst_proj = self.dst_gate(node_feats)  # L_j * h_j

        if self.num_heads > 1:
            N = node_feats.shape[0]
            src_proj_mh = src_proj.view(N, self.num_heads, self.head_dim)
            dst_proj_mh = dst_proj.view(N, self.num_heads, self.head_dim)

            # Apply per-head attention scalars
            src_proj_mh = self.src_attention.unsqueeze(0) * src_proj_mh
            dst_proj_mh = self.dst_attention.unsqueeze(0) * dst_proj_mh

            g.ndata["e_src"] = src_proj_mh.view(N, self.output_features)
            g.ndata["e_dst"] = dst_proj_mh.view(N, self.output_features)
        else:
            # Single-head: simple scalar multiplication
            g.ndata["e_src"] = self.src_attention * src_proj
            g.ndata["e_dst"] = self.dst_attention * dst_proj

        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # Sigmoid gating and message passing (same as original)
        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # Activation + normalization
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        # Multi-head projection
        if self.num_heads > 1:
            x = self.fc_node(x)
            y = self.fc_edge(y)

        # Residual connection
        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


# ---------------------------------------------------------------------------
# T-ALIGNN: Self-Attention Layer (SAL)
#   score = Softmax((q_i . k_j . k_e) / sqrt(dim))
#   h_i^{l+1} = fc_i + SiLU(Norm(sum_j(score . v_j)))
#   e_ij^{l+1} = fc_e + SiLU(Norm(score))
# Uses Transformer-style self-attention to capture long-range interactions.
#
# NOTE: All edge-level operations use direct tensor indexing via
#   src, dst = g.edges()
# to avoid DGL version-specific APIs (copy_u, copy_v, etc.).
# ---------------------------------------------------------------------------


class SelfAttentionGraphConv(nn.Module):
    """Self-Attention Layer (SAL) for T-ALIGNN.

    Replaces the edge-gated convolution with a Transformer-style
    self-attention mechanism that generates query, key, and value
    vectors to compute attention scores between atoms.

    This enables capturing both local fine-grained interactions and
    global correlation features at the molecular graph level, making
    it better suited for properties sensitive to long-range interactions
    (e.g., dielectric constants, SLME, spillage).

    Supports multi-head attention, which the paper shows is critical
    for T-ALIGNN performance (heads=4 recommended).

    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        residual: bool = True,
        num_heads: int = 1,
    ):
        """Initialize parameters for SAL update.

        Args:
            input_features: Input feature dimension.
            output_features: Output feature dimension.
            residual: Whether to use residual connections.
            num_heads: Number of attention heads. Paper shows SAL
                benefits significantly from multi-head (heads=4 best).
        """
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.output_features = output_features

        assert output_features % num_heads == 0, (
            f"output_features ({output_features}) must be divisible "
            f"by num_heads ({num_heads})"
        )
        self.head_dim = output_features // num_heads

        # Query, Key, Value projections for nodes
        self.W_query = nn.Linear(input_features, output_features)
        self.W_key = nn.Linear(input_features, output_features)
        self.W_value = nn.Linear(input_features, output_features)

        # Key projection for edge features
        self.W_key_edge = nn.Linear(input_features, output_features)

        # Normalization layers
        self.bn_nodes = nn.LayerNorm(output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        # fc layers: dimension matching + residual connection (Eq. 10-11)
        self.fc_node = nn.Linear(input_features, output_features)
        self.fc_edge = nn.Linear(input_features, output_features)

        # Scaling factor for attention scores (Eq. 9)
        self.scale = self.head_dim**0.5

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-Attention graph convolution.

        Eq. (9):  score = Softmax((q_i . k_j . k_e) / sqrt(dim))
        Eq. (10): h_i^{l+1} = fc_i + SiLU(Norm(sum_j(score . v_j)))
        Eq. (11): e_ij^{l+1} = fc_e + SiLU(Norm(score))
        """
        g = g.local_var()
        num_nodes = node_feats.shape[0]
        num_edges = edge_feats.shape[0]

        # --- Get edge endpoint indices (DGL-version-agnostic) ---
        # src[e] = source node of edge e (neighbor j)
        # dst[e] = destination node of edge e (central node i)
        src, dst = g.edges()

        # --- Generate Q, K, V vectors ---
        q = self.W_query(node_feats)  # [N, output_features]
        k = self.W_key(node_feats)  # [N, output_features]
        v = self.W_value(node_feats)  # [N, output_features]
        k_e = self.W_key_edge(edge_feats)  # [E, output_features]

        # --- Reshape for multi-head attention ---
        # [N, output_features] -> [N, num_heads, head_dim]
        q = q.view(num_nodes, self.num_heads, self.head_dim)
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)
        k_e = k_e.view(num_edges, self.num_heads, self.head_dim)

        # --- Gather per-edge Q, K, V via direct indexing ---
        # No apply_edges / copy_u / copy_v needed
        q_dst = q[dst]  # [E, num_heads, head_dim] query from central i
        k_src = k[src]  # [E, num_heads, head_dim] key from neighbor j
        v_src = v[src]  # [E, num_heads, head_dim] value from neighbor j

        # --- Compute attention scores (Eq. 9) ---
        # score = Softmax((q_i . k_j . k_e) / sqrt(dim))
        # Element-wise triple product, then sum over head_dim
        attn_raw = (q_dst * k_src * k_e) / self.scale
        # [E, num_heads, head_dim] -> [E, num_heads, 1]
        attn_logits = attn_raw.sum(dim=-1, keepdim=True)

        # Softmax over incoming edges per destination node
        # Squeeze to [E, num_heads] for edge_softmax, then unsqueeze back
        attn_logits_2d = attn_logits.squeeze(-1)  # [E, num_heads]
        attn_scores_2d = _edge_softmax(g, attn_logits_2d)  # [E, num_heads]
        attn_scores = attn_scores_2d.unsqueeze(-1)  # [E, num_heads, 1]

        # Broadcast attention scores across head_dim
        score_full = attn_scores.expand_as(k_src)  # [E, heads, head_dim]

        # --- Node update (Eq. 10) ---
        # h_i^{l+1} = fc_i + SiLU(Norm(sum_j(score . v_j)))
        weighted_v = score_full * v_src  # [E, num_heads, head_dim]
        # Reshape to [E, output_features] for scatter aggregation
        weighted_v_flat = weighted_v.view(num_edges, self.output_features)

        # Scatter-add: aggregate weighted values per destination node
        h_agg = torch.zeros(
            num_nodes,
            self.output_features,
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        h_agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(weighted_v_flat),
            weighted_v_flat,
        )

        # Apply normalization, activation, and residual (fc)
        x = self.fc_node(node_feats) + F.silu(self.bn_nodes(h_agg))

        # --- Edge update (Eq. 11) ---
        # e_ij^{l+1} = fc_e + SiLU(Norm(score))
        score_for_edge = score_full.reshape(num_edges, self.output_features)
        y = self.fc_edge(edge_feats) + F.silu(self.bn_edges(score_for_edge))

        return x, y


# ---------------------------------------------------------------------------
# ALIGNN Convolution Layer (alternating line graph / bond graph updates)
# ---------------------------------------------------------------------------


class ALIGNNConv(nn.Module):
    """ALIGNN convolution: alternating updates on bond graph and line graph.

    Supports all three convolution types:
    - "alignn": Original EdgeGatedGraphConv
    - "n_alignn": Node-Attention Layer (NAL)
    - "t_alignn": Self-Attention Layer (SAL)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_type: str = "alignn",
        num_heads: int = 1,
    ):
        """Set up ALIGNN layer parameters.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            conv_type: One of "alignn", "n_alignn", "t_alignn".
            num_heads: Number of attention heads (for NAL/SAL).
        """
        super().__init__()

        if conv_type == "alignn":
            self.node_update = EdgeGatedGraphConv(in_features, out_features)
            self.edge_update = EdgeGatedGraphConv(out_features, out_features)
        elif conv_type == "n_alignn":
            #  Shao et al., Adv. Theory Simul. 2026
            self.node_update = NodeAttentionGraphConv(
                in_features, out_features, num_heads=num_heads
            )
            self.edge_update = NodeAttentionGraphConv(
                out_features, out_features, num_heads=num_heads
            )
        elif conv_type == "t_alignn":
            self.node_update = SelfAttentionGraphConv(
                in_features, out_features, num_heads=num_heads
            )
            self.edge_update = SelfAttentionGraphConv(
                out_features, out_features, num_heads=num_heads
            )
        else:
            raise ValueError(
                f"Unknown conv_type: {conv_type}. "
                f"Must be one of 'alignn', 'n_alignn', 't_alignn'."
            )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        First applies convolution on the line graph l(g) to update
        edge features using three-body (angle) information, then
        applies convolution on the bond graph g to update node features.
            m^l, t^{l+1}_{ijk} = Conv(L(g), e^l_{ij}, t^l_{ijk})
            h^{l+1}_i, e^{l+1}_{ij} = Conv(g, h^l_i, m^l)

        Args:
            g: Bond graph (atoms as nodes, bonds as edges).
            lg: Line graph (bonds as nodes, angles as edges).
            x: Node features [num_atoms, hidden_features].
            y: Edge features [num_bonds, hidden_features].
            z: Angle features [num_angles, hidden_features].
        """
        g = g.local_var()
        lg = lg.local_var()

        # Step 1: Line graph convolution
        # Updates edge features (y -> m) using angle info (z)
        y, z = self.edge_update(lg, y, z)

        # Step 2: Bond graph convolution
        # Updates node features (x) using updated edge features (m=y)
        x, y = self.node_update(g, x, y)

        return x, y, z


# ---------------------------------------------------------------------------
# Main Model: ALIGNNAtomWise with configurable attention mechanism
# ---------------------------------------------------------------------------


class ALIGNNAtomWise(nn.Module):
    """Atomistic Line Graph Neural Network with Attention Mechanisms.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph. Supports three convolution modes:

    1. Original ALIGNN (conv_type="alignn"):
       Standard edge-gated graph convolution.

    2. N-ALIGNN (conv_type="n_alignn"):
       Node-Attention Layer adds learnable attention parameters
       to weight atomic contributions during message passing.
       Better for properties dominated by local interactions.

    3. T-ALIGNN (conv_type="t_alignn"):
       Self-Attention Layer uses Transformer-style Q/K/V attention
       to capture long-range interactions. Better for properties
       sensitive to global molecular structure.
       Recommended: num_heads=4 for best performance.
    """

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(
            name="alignn_atomwise"
        ),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.classification = config.classification
        self.config = config
        if self.config.gradwise_weight == 0:
            self.config.calculate_gradient = False

        # --- Embedding layers ---
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        # --- ALIGNN layers (line graph + bond graph convolution) ---
        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                    conv_type=config.conv_type,
                    num_heads=config.num_heads,
                )
                for idx in range(config.alignn_layers)
            ]
        )

        # --- GCN layers (bond graph only, post-ALIGNN refinement) ---
        gcn_layers = []
        for idx in range(config.gcn_layers):
            if config.conv_type == "alignn":
                gcn_layers.append(
                    EdgeGatedGraphConv(
                        config.hidden_features, config.hidden_features
                    )
                )
            elif config.conv_type == "n_alignn":
                gcn_layers.append(
                    NodeAttentionGraphConv(
                        config.hidden_features,
                        config.hidden_features,
                        num_heads=config.num_heads,
                    )
                )
            elif config.conv_type == "t_alignn":
                gcn_layers.append(
                    SelfAttentionGraphConv(
                        config.hidden_features,
                        config.hidden_features,
                        num_heads=config.num_heads,
                    )
                )
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # --- Readout and output ---
        self.readout = AvgPooling()

        if config.extra_features != 0:
            self.readout_feat = AvgPooling()
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
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
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN forward pass.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            if len(g) == 3:
                g, lg, lat = g
                lg = lg.local_var()
                z = self.angle_embedding(lg.edata["h"])
            else:
                g, lat = g
                g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
                g.ndata["cart_coords"].requires_grad_(True)
                r, bondlength = compute_pair_vector_and_distance(g)
                lg = g.line_graph(shared=True)
                lg.ndata["r"] = r
                lg.apply_edges(compute_bond_cosines)
        else:
            g, lat = g

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)

        result = {}

        # Initial node features
        x = g.ndata["atom_features"]
        x = self.atom_embedding(x)

        # Bond displacement vectors
        r = g.edata["r"]
        if self.config.include_pos_deriv:
            g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
            g.ndata["cart_coords"].requires_grad_(True)
            r, bondlength = compute_pair_vector_and_distance(g)
            lg = g.line_graph(shared=True)
            lg.ndata["r"] = r
            lg.apply_edges(compute_bond_cosines)

        if (
            self.config.calculate_gradient
            and not self.config.include_pos_deriv
        ):
            r.requires_grad_(True)
        bondlength = torch.norm(r, dim=1)

        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            lg.ndata["r"] = r
            lg.apply_edges(compute_bond_cosines)
            z = self.angle_embedding(lg.edata["h"])

        # Edge embedding with optional cutoff
        if self.config.use_cutoff_function:
            if self.config.multiply_cutoff:
                c_off = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                ).unsqueeze(dim=1)
                y = self.edge_embedding(bondlength) * c_off
            else:
                bondlength = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                )
                y = self.edge_embedding(bondlength)
        else:
            y = self.edge_embedding(bondlength)

        # --- ALIGNN updates: line graph + bond graph convolution ---
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # --- GCN updates: bond graph only ---
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # --- Readout and prediction ---
        out = torch.empty(1)
        additional_out = torch.empty(1)
        if self.config.output_features is not None:
            h = self.readout(g, x)
            out = self.fc(h)
            if self.config.extra_features != 0:
                h_feat = self.readout_feat(g, features)
                h = torch.cat((h, h_feat), 1)
                h = self.fc1(h)
                h = self.fc2(h)
                out = self.fc3(h)
            else:
                out = torch.squeeze(out)
            if self.config.additional_output_features > 0:
                additional_out = self.fc_additional_output(h)

        atomwise_pred = torch.empty(1)
        if (
            self.config.atomwise_output_features > 0
            and self.config.atomwise_weight != 0
        ):
            atomwise_pred = self.fc_atomwise(x)

        forces = torch.empty(1)
        stress = torch.empty(1)
        natoms = torch.tensor([gg.num_nodes() for gg in dgl.unbatch(g)]).to(
            g.device
        )
        en_out = out
        if self.config.energy_mult_natoms:
            en_out = out * natoms

        if self.config.use_penalty:
            penalty_factor = self.config.penalty_factor
            penalty_threshold = self.config.penalty_threshold
            penalties = torch.where(
                bondlength < penalty_threshold,
                penalty_factor * (penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            total_penalty = torch.sum(penalties)
            en_out += total_penalty

        # --- Force / stress calculation via autograd ---
        if self.config.calculate_gradient:
            if self.config.include_pos_deriv:
                dx = [g.ndata["cart_coords"]]
                forces = (
                    self.config.grad_multiplier
                    * grad(
                        en_out * g.num_nodes(),
                        dx,
                        grad_outputs=torch.ones_like(en_out),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                )
            else:
                dx = r
                pair_forces = (
                    self.config.grad_multiplier
                    * grad(
                        en_out,
                        dx,
                        grad_outputs=torch.ones_like(en_out),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                )
                if self.config.force_mult_natoms:
                    pair_forces *= g.num_nodes()

                g.edata["pair_forces"] = pair_forces
                g.update_all(
                    fn.copy_e("pair_forces", "m"),
                    fn.sum("m", "forces_ji"),
                )
                if self.config.add_reverse_forces:
                    rg = dgl.reverse(g, copy_edata=True)
                    rg.update_all(
                        fn.copy_e("pair_forces", "m"),
                        fn.sum("m", "forces_ij"),
                    )
                    forces = torch.squeeze(
                        g.ndata["forces_ji"] - rg.ndata["forces_ij"]
                    )
                else:
                    forces = torch.squeeze(g.ndata["forces_ji"])

                if self.config.stresswise_weight != 0:
                    if not self.config.batch_stress:
                        g.ndata["cart_coords"] = compute_cartesian_coordinates(
                            g, lat
                        )
                        r, bondlength = compute_pair_vector_and_distance(g)
                        stress = (
                            -1
                            * 160.21766208
                            * (
                                torch.matmul(r.T, pair_forces)
                                / (2 * g.ndata["V"][0])
                            )
                        )
                    else:
                        stresses = []
                        count_edge = 0
                        count_node = 0
                        for graph_id in range(g.batch_size):
                            num_edges = g.batch_num_edges()[graph_id]
                            num_nodes = 0
                            st = -1 * (
                                160.21766208
                                * torch.matmul(
                                    r[count_edge : count_edge + num_edges].T,
                                    pair_forces[
                                        count_edge : count_edge + num_edges
                                    ],
                                )
                                / g.ndata["V"][count_node + num_nodes]
                            )
                            count_edge = count_edge + num_edges
                            num_nodes = g.batch_num_nodes()[graph_id]
                            count_node = count_node + num_nodes
                            stresses.append(st)
                        stress = self.config.stress_multiplier * torch.stack(
                            stresses
                        )

        if self.link:
            out = self.link(out)

        if self.classification:
            out = self.softmax(out)

        result["out"] = out
        result["additional"] = additional_out
        result["grad"] = forces
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred

        return result


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def get_alignn_model(
    conv_type: str = "alignn",
    num_heads: int = 1,
    **kwargs,
) -> ALIGNNAtomWise:
    """Create an ALIGNN model with the specified convolution type.

    Args:
        conv_type: One of "alignn", "n_alignn", "t_alignn".
        num_heads: Number of attention heads (recommended: 4 for t_alignn).
        **kwargs: Additional config parameters.

    Returns:
        Configured ALIGNNAtomWise model.

    Examples:
        # Original ALIGNN
        model = get_alignn_model("alignn")

        # N-ALIGNN with node attention
        model = get_alignn_model("n_alignn")

        # T-ALIGNN with 4-head self-attention (recommended)
        model = get_alignn_model("t_alignn", num_heads=4)
    """
    config = ALIGNNAtomWiseConfig(
        name="alignn_atomwise",
        conv_type=conv_type,
        num_heads=num_heads,
        **kwargs,
    )
    return ALIGNNAtomWise(config)
