"""Pure-PyTorch ALIGNN-atomwise with smooth radial/angular bases.

A drop-in cousin of ``alignn_atomwise_pure`` with:

  * **Learnable Bessel radial basis** (replaces fixed Gaussian RBF for
    bond distances).
  * **Learnable Fourier angular basis** (replaces cosine RBF for
    triplet angles).
  * **Polynomial smooth cutoff envelope** applied to the radial basis,
    so bond features (and forces) go continuously to zero at the cutoff
    (default coeff 5).
  * **AtomRef composition baseline**: a linear per-element offset is
    subtracted before the GNN and added back at the end, so the network
    only has to learn the residual chemistry.
  * **mlp_first / site-energy readout**: project per-atom features to a
    scalar site energy and sum (instead of mean-pool then project).
    Energy is extensive by construction and per-atom contributions are
    available for defect / segregation analysis.

Same forward output keys as ``ALIGNNAtomWisePure``; same TorchGraph
input contract; same TorchScript entry point.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

from alignn.models.alignn_atomwise_pure import (
    ALIGNNConvPure,
    EdgeGatedGraphConvPure,
    _bond_cosines,
    _make_bond_conv,
    scatter_mean,
    scatter_sum,
)
from alignn.models.utils import MLPLayer
from alignn.torch_graph_builder import TorchGraph, torchgraph_from_dgl
from alignn.utils import BaseSettings


# =====================================================================
# Smooth bases
# =====================================================================


class CutoffPolynomial(nn.Module):
    """Polynomial soft-cutoff envelope.

    Decays smoothly from 1 at r=0 to 0 at r=cutoff with continuous
    first two derivatives. ``coeff=0`` disables it (returns 1).
    """

    def __init__(self, cutoff: float = 8.0, coeff: float = 5.0) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.p = float(coeff)
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if self.p == 0:
            return torch.ones_like(r)
        rs = r / self.cutoff
        env = (
            1.0
            + self.a * rs**self.p
            + self.b * rs ** (self.p + 1)
            + self.c * rs ** (self.p + 2)
        )
        return torch.where(rs < 1.0, env, torch.zeros_like(rs))


class RadialBessel(nn.Module):
    """Learnable Bessel radial basis with smooth polynomial cutoff."""

    def __init__(
        self,
        num_radial: int = 80,
        cutoff: float = 8.0,
        learnable: bool = True,
        smooth_cutoff_coeff: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.inv_cutoff = 1.0 / float(cutoff)
        self.norm_const = (2.0 * self.inv_cutoff) ** 0.5

        freq = np.pi * torch.arange(1, num_radial + 1, dtype=torch.float)
        if learnable:
            self.frequencies = nn.Parameter(freq, requires_grad=True)
        else:
            self.register_buffer("frequencies", freq)

        self.smooth = (
            CutoffPolynomial(cutoff=cutoff, coeff=smooth_cutoff_coeff)
            if smooth_cutoff_coeff and smooth_cutoff_coeff > 0
            else None
        )

    @property
    def out_features(self) -> int:
        return self.num_radial

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (E,)
        d = dist.unsqueeze(-1).clamp_min(1e-8)
        out = self.norm_const * torch.sin(self.frequencies * d * self.inv_cutoff) / d
        if self.smooth is not None:
            out = out * self.smooth(d)
        return out


class FourierAngular(nn.Module):
    """Learnable Fourier angular basis on theta in [-pi, pi].

    Input is bond-angle cosine in [-1, 1]; converted to theta via acos.
    Output features = 1 + 2*order.
    """

    def __init__(self, order: int = 9, learnable: bool = True) -> None:
        super().__init__()
        self.order = order
        freq = torch.arange(1, order + 1, dtype=torch.float)
        if learnable:
            self.frequencies = nn.Parameter(freq, requires_grad=True)
        else:
            self.register_buffer("frequencies", freq)

    @property
    def out_features(self) -> int:
        return 1 + 2 * self.order

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        out = theta.new_zeros(theta.shape[0], 1 + 2 * self.order)
        out[:, 0] = 0.7071067811865475  # 1/sqrt(2)
        tmp = torch.outer(theta, self.frequencies)
        out[:, 1 : self.order + 1] = torch.sin(tmp)
        out[:, self.order + 1 :] = torch.cos(tmp)
        return out / 1.7724538509055159  # sqrt(pi)


# =====================================================================
# Config
# =====================================================================


class ALIGNNAtomWisePureSmoothConfig(BaseSettings):
    """Mirrors ALIGNNAtomWisePureConfig and adds smooth-basis options."""

    name: Literal["alignn_atomwise_pure_smooth"]
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
    # Site-energy readout (mlp_first) makes energy extensive natively, so
    # the multiplicative natoms is OFF by default in this variant.
    energy_mult_natoms: bool = False
    include_pos_deriv: bool = False
    inner_cutoff: float = 3.0
    stress_multiplier: float = 1.0
    add_reverse_forces: bool = True
    lg_on_fly: bool = True
    batch_stress: bool = True
    use_penalty: bool = True
    extra_features: int = 0
    exponent: int = 5
    penalty_factor: float = 0.5
    penalty_threshold: float = 1.0
    additional_output_features: int = 0
    additional_output_weight: float = 0.0
    conv_type: Literal["alignn", "n_alignn", "t_alignn"] = "alignn"
    num_heads: int = 1

    # --- smooth-basis additions ---
    # Smooth polynomial cutoff on the radial basis. Set coeff=0 to disable.
    radial_cutoff: float = 8.0
    smooth_cutoff_coeff: float = 5.0
    # "gaussian"/"cosine" fall back to the original alignn_atomwise_pure
    # behaviour.
    radial_basis: Literal["bessel", "gaussian"] = "bessel"
    angular_basis: Literal["fourier", "cosine"] = "fourier"
    radial_learnable: bool = True
    angular_learnable: bool = True
    # Linear per-element composition baseline (atomref). Learned jointly.
    use_atomref: bool = True
    # Per-atom site-energy readout (mlp_first=True semantics):
    # site_e = fc(x); energy = sum_i site_e_i.
    mlp_first: bool = True

    class Config:
        env_prefix = "jv_model"


# =====================================================================
# Model
# =====================================================================


def _as_torchgraph(x):
    if isinstance(x, TorchGraph):
        return x
    return torchgraph_from_dgl(x)


class ALIGNNAtomWisePureSmooth(nn.Module):
    """DGL-free ALIGNN-FF with smooth bases & site energies."""

    def __init__(
        self,
        config: ALIGNNAtomWisePureSmoothConfig = ALIGNNAtomWisePureSmoothConfig(
            name="alignn_atomwise_pure_smooth"
        ),
    ):
        super().__init__()
        self.config = config
        self.classification = config.classification
        if self.config.gradwise_weight == 0:
            self.config.calculate_gradient = False

        # --- atom embedding ---
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        # --- AtomRef (composition baseline) ---
        # Linear in atom_features, init to zero so it starts neutral and
        # learns a per-element offset by gradient descent.
        if config.use_atomref:
            self.atomref = nn.Linear(
                config.atom_input_features, 1, bias=False
            )
            nn.init.zeros_(self.atomref.weight)
        else:
            self.atomref = None

        # --- radial basis ---
        self.radial_envelope: Optional[CutoffPolynomial] = None
        if config.radial_basis == "bessel":
            self.radial = RadialBessel(
                num_radial=config.edge_input_features,
                cutoff=config.radial_cutoff,
                learnable=config.radial_learnable,
                smooth_cutoff_coeff=config.smooth_cutoff_coeff,
            )
            edge_in_dim = self.radial.out_features
            self._radial_kind = "bessel"
        else:
            from alignn.models.utils import RBFExpansion

            self.radial = RBFExpansion(
                vmin=0, vmax=config.radial_cutoff, bins=config.edge_input_features
            )
            edge_in_dim = config.edge_input_features
            self._radial_kind = "gaussian"
            # Optional standalone smooth envelope multiplied onto the
            # Gaussian expansion (off by default for this branch).
            if config.smooth_cutoff_coeff and config.smooth_cutoff_coeff > 0:
                self.radial_envelope = CutoffPolynomial(
                    cutoff=config.radial_cutoff,
                    coeff=config.smooth_cutoff_coeff,
                )
            else:
                self.radial_envelope = None

        self.edge_embedding = nn.Sequential(
            MLPLayer(edge_in_dim, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        # --- angular basis ---
        if config.angular_basis == "fourier":
            # Pick order so the basis has ~triplet_input_features dims.
            order = max(1, (config.triplet_input_features - 1) // 2)
            self.angular = FourierAngular(
                order=order, learnable=config.angular_learnable
            )
            angle_in_dim = self.angular.out_features
            self._angular_kind = "fourier"
        else:
            from alignn.models.utils import RBFExpansion

            self.angular = RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features
            )
            angle_in_dim = config.triplet_input_features
            self._angular_kind = "cosine"

        self.angle_embedding = nn.Sequential(
            MLPLayer(angle_in_dim, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        # --- conv stack ---
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

        # --- heads ---
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

        self.link_name: str = config.link
        if config.link == "log":
            self.fc.bias.data = torch.tensor(np.log(0.7), dtype=torch.float)

        self.register_buffer(
            "_species_table",
            torch.zeros(120, config.atom_input_features, dtype=torch.float32),
        )

        # Typed mirrors of pydantic config for TorchScript-friendly paths.
        self.energy_mult_natoms: bool = bool(config.energy_mult_natoms)
        self.use_penalty: bool = bool(config.use_penalty)
        self.penalty_threshold: float = float(config.penalty_threshold)
        self.penalty_factor: float = float(config.penalty_factor)
        self.grad_multiplier: int = int(config.grad_multiplier)
        self.add_reverse_forces: bool = bool(config.add_reverse_forces)
        self.stress_multiplier: float = float(config.stress_multiplier)
        self.mlp_first: bool = bool(config.mlp_first)

    # ----- species lookup (LAMMPS path) -----

    @torch.jit.ignore
    def register_species_table(
        self, atom_features: str = "cgcnn", max_z: int = 119
    ) -> None:
        from ase.data import chemical_symbols
        from jarvis.core.specie import get_node_attributes

        F_dim = int(self.config.atom_input_features)
        rows = np.zeros((max_z + 1, F_dim), dtype=np.float32)
        for z in range(1, max_z + 1):
            if z >= len(chemical_symbols):
                continue
            symbol = chemical_symbols[z]
            try:
                feat = np.asarray(
                    get_node_attributes(symbol, atom_features=atom_features),
                    dtype=np.float32,
                )
                if feat.shape[0] != F_dim:
                    continue
                rows[z] = feat
            except Exception:
                continue
        self._species_table.resize_(rows.shape).copy_(
            torch.as_tensor(rows, dtype=torch.float32)
        )

    # ----- internal embedding helpers -----

    def _embed_radial(self, bondlength: torch.Tensor) -> torch.Tensor:
        if self._radial_kind == "bessel":
            return self.edge_embedding(self.radial(bondlength))
        feats = self.radial(bondlength)
        if self.radial_envelope is not None:
            feats = feats * self.radial_envelope(bondlength).unsqueeze(-1)
        return self.edge_embedding(feats)

    def _embed_angular(self, cos_theta: torch.Tensor) -> torch.Tensor:
        return self.angle_embedding(self.angular(cos_theta))

    # ----- tensor-only forward (LAMMPS / TorchScript path) -----

    @torch.jit.export
    def forward_tensors_z(
        self,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        atomic_numbers: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        shift: torch.Tensor,
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        atom_features = self._species_table.index_select(0, atomic_numbers)
        return self.forward_tensors(
            positions, lattice, atom_features, src, dst, shift, compute_stress
        )

    @torch.jit.export
    def forward_tensors(
        self,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        atom_features: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        shift: torch.Tensor,
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = positions.shape[0]

        # Differentiable edge vectors.
        r = (
            positions.index_select(0, dst)
            - positions.index_select(0, src)
            + torch.matmul(shift, lattice)
        )
        bondlength = torch.linalg.vector_norm(r, dim=1)

        # Inline line graph: A=(u,v) -> B=(v,w).
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

        # Angle cosines (differentiable through r).
        r_ij = r.index_select(0, lg_src)
        r_jk = r.index_select(0, lg_dst)
        num = -(r_ij * r_jk).sum(dim=-1)
        denom = torch.linalg.vector_norm(r_ij, dim=-1).clamp_min(
            1e-12
        ) * torch.linalg.vector_norm(r_jk, dim=-1).clamp_min(1e-12)
        h_cos = (num / denom).clamp(-1.0, 1.0)

        # Embeddings (Bessel/Fourier with smooth cutoff if configured).
        x = self.atom_embedding(atom_features)
        y = self._embed_radial(bondlength)
        z = self._embed_angular(h_cos)

        for layer in self.alignn_layers:
            x, y, z = layer.forward_tensors(
                src, dst, num_nodes, lg_src, lg_dst, lg_num_nodes, x, y, z
            )
        for layer in self.gcn_layers:
            x, y = layer.forward_tensors(src, dst, num_nodes, x, y)

        # Site-energy readout: fc per atom then sum (extensive).
        if self.mlp_first:
            site_e = self.fc(x).squeeze(-1)  # (N,)
            en = site_e.sum().unsqueeze(0)   # (1,)
        else:
            h_graph = x.mean(dim=0, keepdim=True)
            en = self.fc(h_graph).squeeze(-1)
            if self.energy_mult_natoms:
                en = en * float(num_nodes)

        # AtomRef composition baseline added back as a per-element sum.
        if self.atomref is not None:
            en = en + self.atomref(atom_features).sum().unsqueeze(0)

        if self.use_penalty:
            pen = torch.where(
                bondlength < self.penalty_threshold,
                self.penalty_factor * (self.penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            en = en + pen.sum()

        result: Dict[str, torch.Tensor] = {"energy": en.squeeze()}

        grad_outs = torch.autograd.grad(
            outputs=[en.sum()],
            inputs=[r],
            create_graph=self.training,
            retain_graph=True,
        )
        pair_forces = grad_outs[0]
        assert pair_forces is not None
        pair_forces = self.grad_multiplier * pair_forces

        forces_ji = scatter_sum(pair_forces, dst, num_nodes)
        if self.add_reverse_forces:
            forces_ij = scatter_sum(pair_forces, src, num_nodes)
            forces = forces_ji - forces_ij
        else:
            forces = forces_ji
        result["forces"] = forces

        if compute_stress:
            V = torch.abs(torch.det(lattice))
            stress = -160.21766208 * (torch.matmul(r.T, pair_forces) / V)
            result["stress"] = self.stress_multiplier * stress

        return result

    # ----- dataclass-based forward (training pipeline) -----

    @torch.jit.ignore
    def forward(self, g):
        gg, lg, lat = g
        g = _as_torchgraph(gg)
        lg = _as_torchgraph(lg)

        r = g.edata["r"]
        if self.config.calculate_gradient:
            r.requires_grad_(True)

        bondlength = torch.norm(r, dim=1)

        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            h = _bond_cosines(r[lg.src], r[lg.dst])
            z = self._embed_angular(h)
        elif len(self.alignn_layers) > 0:
            z = self._embed_angular(lg.edata["h"])
        else:
            z = None

        atom_feats = g.ndata["atom_features"]
        x = self.atom_embedding(atom_feats)
        y = self._embed_radial(bondlength)

        for layer in self.alignn_layers:
            x, y, z = layer(g, lg, x, y, z)
        for layer in self.gcn_layers:
            x, y = layer(g, x, y)

        node_bid = g.node_batch_id
        B = g.batch_size
        natoms = (
            g.batch_num_nodes
            if g.batch_num_nodes is not None
            else torch.tensor([g.num_nodes], device=r.device)
        )

        # --- energy readout ---
        if self.config.mlp_first:
            site_e = self.fc(x)                       # (N, output_features)
            en_out = scatter_sum(site_e, node_bid, B) # (B, output_features)
            if self.config.output_features == 1:
                en_out = en_out.squeeze(-1)
            out = en_out  # for link-fn / classification compatibility below
        else:
            h_graph = scatter_mean(x, node_bid, B)
            out = self.fc(h_graph)
            if self.config.output_features == 1:
                out = out.squeeze(-1)
            en_out = out
            if self.config.energy_mult_natoms:
                en_out = out * natoms.to(out.dtype)

        # --- AtomRef composition baseline (per-graph sum) ---
        if self.atomref is not None:
            atomref_e = self.atomref(atom_feats).squeeze(-1)  # (N,)
            en_out = en_out + scatter_sum(atomref_e, node_bid, B)

        if self.config.use_penalty:
            # Penalty is per-edge; aggregate per-graph via edge_batch_id.
            penalties = torch.where(
                bondlength < self.config.penalty_threshold,
                self.config.penalty_factor
                * (self.config.penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            if g.edge_batch_id is not None:
                en_out = en_out + scatter_sum(penalties, g.edge_batch_id, B)
            else:
                en_out = en_out + penalties.sum()

        atomwise_pred = torch.empty(1, device=r.device)
        if (
            self.config.atomwise_output_features > 0
            and self.config.atomwise_weight != 0
        ):
            atomwise_pred = self.fc_atomwise(x)

        additional_out = torch.empty(1, device=r.device)
        if self.config.additional_output_features > 0:
            additional_out = self.fc_additional_output(
                scatter_mean(x, node_bid, B)
            )

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

            forces_ji = scatter_sum(pair_forces, g.dst, g.num_nodes)
            if self.config.add_reverse_forces:
                forces_ij = scatter_sum(pair_forces, g.src, g.num_nodes)
                forces = (forces_ji - forces_ij).squeeze()
            else:
                forces = forces_ji.squeeze()

            if self.config.stresswise_weight != 0:
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

        # Keep the same return contract as ALIGNNAtomWisePure.
        return {
            "out": en_out if self.config.mlp_first else out,
            "additional": additional_out,
            "grad": forces,
            "stresses": stress,
            "atomwise_pred": atomwise_pred,
        }
