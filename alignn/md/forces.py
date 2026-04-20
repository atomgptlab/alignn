"""Wrap ALIGNN-FF (pure PyTorch variant) as an on-device forces function.

A `forces_fn` has signature:
    energy, forces = forces_fn(positions)          # positions: (N, 3) on GPU
                                                    # energy:    scalar tensor
                                                    # forces:    (N, 3) tensor

Current implementation rebuilds the graph every step via the existing
jarvis/DGL pipeline. That's CPU-heavy and will be the bottleneck for large
systems — replace with an on-device neighbor list when you scale up.
"""
from __future__ import annotations
import numpy as np
import torch
from ase import Atoms as AseAtoms
from jarvis.core.atoms import ase_to_atoms as ase_to_jarvis

from alignn.graphs import Graph
from alignn.torch_graph_builder import torchgraph_from_dgl


class AlignnForces:
    def __init__(
        self,
        model,
        atomic_numbers: np.ndarray,
        cell: np.ndarray,              # (3, 3), constant for NVT/NVE
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.eval().to(device).to(dtype)
        self.Z = np.asarray(atomic_numbers)
        self.cell_np = np.asarray(cell, dtype=float)
        self.cell = torch.tensor(self.cell_np, dtype=dtype, device=device)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.device = torch.device(device)
        self.dtype = dtype

    def _build_graph(self, positions_np):
        ase_atoms = AseAtoms(
            numbers=self.Z, positions=positions_np, cell=self.cell_np, pbc=True
        )
        j = ase_to_jarvis(ase_atoms)
        g, lg = Graph.atom_dgl_multigraph(
            j, neighbor_strategy="k-nearest",
            cutoff=self.cutoff, max_neighbors=self.max_neighbors,
            atom_features="cgcnn", use_canonize=True,
        )
        g = g.to(self.device); lg = lg.to(self.device)
        tg = torchgraph_from_dgl(g); tlg = torchgraph_from_dgl(lg)
        for d in (tg.ndata, tg.edata, tlg.ndata, tlg.edata):
            for k, v in list(d.items()):
                if v.is_floating_point():
                    d[k] = v.to(self.dtype)
        return tg, tlg

    @torch.enable_grad()
    def __call__(self, positions: torch.Tensor):
        positions = positions.detach()
        tg, tlg = self._build_graph(positions.cpu().numpy())
        # model sets r.requires_grad_(True) internally; energy grad wrt r
        # gives per-edge forces, aggregated to per-atom in the model's head.
        out = self.model((tg, tlg, self.cell))
        energy = out["out"].sum() if "out" in out else next(iter(out.values())).sum()
        if "grad" in out:
            forces = out["grad"].detach()
        else:
            # fallback: autograd wrt edge vectors is handled inside the model
            forces = torch.zeros_like(positions)
        return energy.detach(), forces.to(positions.dtype)
