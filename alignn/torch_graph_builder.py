"""Pure-torch graph and line-graph builder (no DGL).

Produces a crystal neighbor graph together with its atomistic line
graph as plain torch data, with separate two-body and three-body
cutoffs. Edge displacement vectors and bond angle cosines are torch
functions of atomic positions and the lattice, so autograd flows back
to both (forces via -dE/dx, stress via dE/dL).

Typical use
-----------

    g, lg = build_pure_torch_graph(
        atoms=jarvis_atoms,
        two_body_cutoff=5.0,
        three_body_cutoff=4.0,   # defaults to two_body_cutoff if None
        max_neighbors=12,
    )
    # g.edata["r"]     : (E, 3) differentiable displacements
    # g.edata["images"]: (E, 3) integer cell offsets
    # g.ndata["atom_features"], g.ndata["frac_coords"], g.ndata["V"]
    # lg.src / lg.dst  : (T,) indices into g's edges (i.e. line-graph
    #                    nodes *are* parent edges)
    # lg.edata["h"]    : (T,) bond-angle cosines at the shared atom
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------
# Neighbor-list primitives (moved here from alignn.graphs so importing
# this module stays DGL-free).
# ---------------------------------------------------------------------


def _torch_periodic_shifts(
    lattice: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Integer shift vectors (K, 3) covering a ``cutoff`` sphere."""
    with torch.no_grad():
        recip = 2 * math.pi * torch.linalg.inv(lattice).T
        recip_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(cutoff * recip_len / (2 * math.pi)).to(torch.long)
    ranges = [
        torch.arange(-int(n.item()), int(n.item()) + 1, device=lattice.device)
        for n in n_max
    ]
    return torch.cartesian_prod(*ranges).to(lattice.dtype)


def _topk_per_source(
    src: torch.Tensor,
    keys: torch.Tensor,
    max_neighbors: int,
    num_nodes: int,
) -> torch.Tensor:
    """Return indices keeping the K smallest ``keys`` per src node."""
    device = src.device
    order_key = torch.argsort(keys)
    src_k = src[order_key]
    order_src = torch.argsort(src_k, stable=True)
    perm = order_key[order_src]
    src_sorted = src[perm]
    E = perm.numel()
    starts = torch.searchsorted(
        src_sorted, torch.arange(num_nodes, device=device)
    )
    within = torch.arange(E, device=device) - starts[src_sorted]
    return perm[within < max_neighbors]


def torch_neighbor_list(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    max_neighbors: Optional[int] = None,
    atoms=None,
    use_matscipy_topology: bool = False,
    self_tol: float = 1e-8,
    chunk_size: int = 512,
):
    """Torch-native periodic neighbor list with differentiable edges.

    Memory-chunked over source atoms: peak memory is O(K * chunk * N)
    instead of O(K * N^2), which also sidesteps torch's INT_MAX limit
    on torch.where for very large boolean tensors.
    """
    dtype = positions.dtype
    device = positions.device
    num_nodes = int(positions.shape[0])

    used_matscipy = False
    if use_matscipy_topology and atoms is not None:
        try:
            from matscipy.neighbours import neighbour_list as _mnl

            i_np, j_np, S_np = _mnl(
                "ijS", atoms.ase_converter(), float(cutoff)
            )
            src = torch.from_numpy(np.ascontiguousarray(i_np)).to(
                device=device, dtype=torch.long
            )
            dst = torch.from_numpy(np.ascontiguousarray(j_np)).to(
                device=device, dtype=torch.long
            )
            shift = torch.from_numpy(np.ascontiguousarray(S_np)).to(
                device=device, dtype=dtype
            )
            used_matscipy = True
        except ImportError:
            pass

    if not used_matscipy:
        shifts = _torch_periodic_shifts(lattice, cutoff)  # (K, 3)
        with torch.no_grad():
            offs = shifts @ lattice  # (K, 3) cartesian
            c2 = float(cutoff) * float(cutoff)

            # Dynamically shrink chunk for very large systems so that
            # (K * chunk * N) bool tensor stays well under INT_MAX.
            K = int(shifts.shape[0])
            max_elems = 2**30  # ~1.07e9, safe
            max_chunk_by_int = max(1, max_elems // max(K * num_nodes, 1))
            eff_chunk = max(1, min(chunk_size, max_chunk_by_int))

            src_chunks, dst_chunks, shift_chunks = [], [], []
            for i0 in range(0, num_nodes, eff_chunk):
                i1 = min(i0 + eff_chunk, num_nodes)
                # (K, chunk, N, 3)
                rvec = (
                    positions[None, None, :, :]
                    + offs[:, None, None, :]
                    - positions[None, i0:i1, None, :]
                )
                dist2 = rvec.pow(2).sum(-1)  # (K, chunk, N)
                mask = (dist2 <= c2) & (dist2 > self_tol)
                del rvec, dist2
                k_idx, i_local, j_idx = torch.where(mask)
                del mask
                src_chunks.append((i_local + i0).to(torch.long))
                dst_chunks.append(j_idx.to(torch.long))
                shift_chunks.append(shifts[k_idx])
                del k_idx, i_local, j_idx

            src = (
                torch.cat(src_chunks)
                if src_chunks
                else torch.empty(0, dtype=torch.long, device=device)
            )
            dst = (
                torch.cat(dst_chunks)
                if dst_chunks
                else torch.empty(0, dtype=torch.long, device=device)
            )
            shift = (
                torch.cat(shift_chunks)
                if shift_chunks
                else torch.empty((0, 3), dtype=dtype, device=device)
            )

    # Differentiable displacement vectors — this is the autograd bridge
    r = positions[dst] - positions[src] + shift @ lattice

    if max_neighbors is not None and max_neighbors > 0 and src.numel() > 0:
        with torch.no_grad():
            dist = r.norm(dim=1)
        keep = _topk_per_source(src, dist, int(max_neighbors), num_nodes)
        src, dst, shift, r = src[keep], dst[keep], shift[keep], r[keep]

    return src, dst, shift, r


def torch_neighbor_list_old(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    max_neighbors: Optional[int] = None,
    atoms=None,
    use_matscipy_topology: bool = True,
    self_tol: float = 1e-8,
):
    """Torch-native periodic neighbor list with differentiable edges.

    Topology (src, dst, integer shift) is computed without gradient
    tracking (matscipy when available, pure-torch fallback otherwise).
    Displacement vectors are then a torch function of positions and
    lattice: ``r = positions[dst] - positions[src] + shift @ lattice``.
    """
    dtype = positions.dtype
    device = positions.device
    num_nodes = positions.shape[0]

    used_matscipy = False
    if use_matscipy_topology and atoms is not None:
        try:
            from matscipy.neighbours import neighbour_list as _mnl

            i_np, j_np, S_np = _mnl(
                "ijS", atoms.ase_converter(), float(cutoff)
            )
            src = torch.from_numpy(np.ascontiguousarray(i_np)).to(
                device=device, dtype=torch.long
            )
            dst = torch.from_numpy(np.ascontiguousarray(j_np)).to(
                device=device, dtype=torch.long
            )
            shift = torch.from_numpy(np.ascontiguousarray(S_np)).to(
                device=device, dtype=dtype
            )
            used_matscipy = True
        except ImportError:
            pass

    if not used_matscipy:
        shifts = _torch_periodic_shifts(lattice, cutoff)
        with torch.no_grad():
            offs = shifts @ lattice
            rvec_full = (
                positions[None, None, :, :]
                + offs[:, None, None, :]
                - positions[None, :, None, :]
            )
            dist2 = rvec_full.pow(2).sum(-1)
            mask = (dist2 <= cutoff * cutoff) & (dist2 > self_tol)
            k_idx, i_idx, j_idx = torch.where(mask)
        src = i_idx.to(torch.long)
        dst = j_idx.to(torch.long)
        shift = shifts[k_idx]

    r = positions[dst] - positions[src] + shift @ lattice

    if max_neighbors is not None and max_neighbors > 0 and src.numel() > 0:
        with torch.no_grad():
            dist = r.norm(dim=1)
        keep = _topk_per_source(src, dist, int(max_neighbors), num_nodes)
        src, dst, shift, r = src[keep], dst[keep], shift[keep], r[keep]

    return src, dst, shift, r


@dataclass
class TorchGraph:
    """Dict-based graph container. Edge list is directed.

    When a graph is the result of batching, ``batch_num_nodes`` /
    ``batch_num_edges`` carry the per-subgraph counts so downstream
    code (pooling, per-graph stress, etc.) can segment correctly.
    """

    num_nodes: int
    src: torch.Tensor
    dst: torch.Tensor
    ndata: Dict[str, torch.Tensor] = field(default_factory=dict)
    edata: Dict[str, torch.Tensor] = field(default_factory=dict)
    batch_num_nodes: Optional[torch.Tensor] = None
    batch_num_edges: Optional[torch.Tensor] = None

    @property
    def num_edges(self) -> int:
        """Return the number of directed edges."""
        return int(self.src.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying tensors."""
        return self.src.device

    @property
    def batch_size(self) -> int:
        """Return the number of subgraphs in this (possibly batched) graph."""
        if self.batch_num_nodes is None:
            return 1
        return int(self.batch_num_nodes.shape[0])

    @property
    def node_batch_id(self) -> torch.Tensor:
        """(N,) long tensor mapping each node to its subgraph index."""
        dev = self.src.device
        if self.batch_num_nodes is None:
            return torch.zeros(self.num_nodes, dtype=torch.long, device=dev)
        return torch.repeat_interleave(
            torch.arange(self.batch_size, device=dev),
            self.batch_num_nodes,
        )

    @property
    def edge_batch_id(self) -> torch.Tensor:
        """(E,) long tensor mapping each edge to its subgraph index."""
        dev = self.src.device
        if self.batch_num_edges is None:
            return torch.zeros(self.num_edges, dtype=torch.long, device=dev)
        return torch.repeat_interleave(
            torch.arange(self.batch_size, device=dev),
            self.batch_num_edges,
        )

    def to(self, device) -> "TorchGraph":
        """Return a copy of this graph moved to ``device``."""
        return TorchGraph(
            num_nodes=self.num_nodes,
            src=self.src.to(device),
            dst=self.dst.to(device),
            ndata={k: v.to(device) for k, v in self.ndata.items()},
            edata={k: v.to(device) for k, v in self.edata.items()},
            batch_num_nodes=(
                self.batch_num_nodes.to(device)
                if self.batch_num_nodes is not None
                else None
            ),
            batch_num_edges=(
                self.batch_num_edges.to(device)
                if self.batch_num_edges is not None
                else None
            ),
        )

    def to_dgl(self):
        """Materialize as a ``dgl.DGLGraph`` preserving ndata / edata.

        Autograd-safe: tensor values are assigned by reference, so
        gradients on ``edata['r']`` continue to flow back to any leaves
        it was derived from (e.g. positions / lattice).
        """
        import dgl as _dgl

        g = _dgl.graph((self.src, self.dst), num_nodes=self.num_nodes)
        for k, v in self.ndata.items():
            g.ndata[k] = v
        for k, v in self.edata.items():
            g.edata[k] = v
        return g


def torch_bond_cosines(
    r_ij: torch.Tensor, r_jk: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Cosine of the bond angle at atom j for triplets i -> j -> k.

    Matches ALIGNN's convention: negate the first bond so vectors
    point away from j, i.e. cos = (-r_ij) . r_jk / (|r_ij| |r_jk|).
    """
    num = -(r_ij * r_jk).sum(dim=-1)
    denom = r_ij.norm(dim=-1).clamp_min(eps) * r_jk.norm(dim=-1).clamp_min(eps)
    return (num / denom).clamp(-1.0, 1.0)


def _line_graph_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    allowed: Optional[torch.Tensor] = None,
):
    """Build line-graph edges from directed parent edges.

    For each parent edge A = (u, v), emit a line-graph edge (A, B) for
    every parent edge B = (v, w). When ``allowed`` is given (bool mask
    of length E), only edges with allowed=True may act as A *or* B.

    Returns
    -------
    lg_src, lg_dst : long tensors, indices into the parent edge list.
    """
    E = int(src.shape[0])
    device = src.device

    if allowed is None:
        allowed_ids = torch.arange(E, device=device)
    else:
        allowed_ids = torch.nonzero(allowed, as_tuple=False).squeeze(-1)

    # Sort allowed parent edges by their src node so we can find all
    # outgoing edges from a node v as a contiguous slice.
    sub_src = src[allowed_ids]
    order = torch.argsort(sub_src, stable=True)
    sorted_edge_ids = allowed_ids[order]
    sorted_src = sub_src[order]

    node_range = torch.arange(num_nodes, device=device)
    bucket_start = torch.searchsorted(sorted_src, node_range)
    bucket_end = torch.searchsorted(sorted_src, node_range, right=True)

    # For each allowed A, count successors = (# allowed edges out of A.dst).
    A_ids = allowed_ids
    A_v = dst[A_ids]
    starts = bucket_start[A_v]
    ends = bucket_end[A_v]
    counts = ends - starts

    total = int(counts.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    lg_src = torch.repeat_interleave(A_ids, counts)

    # Expand per-A ranges [start_A, start_A + count_A) into flat indices.
    cum = torch.cumsum(counts, dim=0)
    row_start = cum - counts
    offsets = torch.arange(total, device=device) - torch.repeat_interleave(
        row_start, counts
    )
    pos = torch.repeat_interleave(starts, counts) + offsets
    lg_dst = sorted_edge_ids[pos]
    return lg_src, lg_dst


def build_pure_torch_graph(
    atoms=None,
    two_body_cutoff: float = 5.0,
    three_body_cutoff: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    atom_features: str = "cgcnn",
    use_lattice_prop: bool = False,
    positions: Optional[torch.Tensor] = None,
    lattice: Optional[torch.Tensor] = None,
    device=None,
    use_matscipy_topology: bool = True,
    compute_line_graph: bool = True,
):
    """Build (g, lg) as pure-torch TorchGraph objects.

    Parameters
    ----------
    two_body_cutoff : float
        Cutoff for the main neighbor graph (edges / pair distances).
    three_body_cutoff : float or None
        Cutoff for triplets in the line graph. Defaults to
        ``two_body_cutoff`` when None. If smaller, only edges with
        ``|r| <= three_body_cutoff`` are allowed to take part in
        triplets; the main graph still carries all edges up to
        ``two_body_cutoff``.
    max_neighbors : int or None
        Per-source cap on the pair graph (keeps K closest).
    positions, lattice : optional torch leaves; when ``requires_grad``
        is set, gradients flow through ``g.edata['r']`` and
        ``lg.edata['h']`` back to these.

    Returns
    -------
    (g, lg) if compute_line_graph else g.
    """
    from jarvis.core.specie import get_node_attributes

    if three_body_cutoff is None:
        three_body_cutoff = two_body_cutoff
    if three_body_cutoff > two_body_cutoff:
        raise ValueError(
            f"three_body_cutoff ({three_body_cutoff}) must be <= "
            f"two_body_cutoff ({two_body_cutoff})."
        )

    dtype = torch.get_default_dtype()
    if positions is None:
        positions = torch.as_tensor(np.asarray(atoms.cart_coords), dtype=dtype)
    if lattice is None:
        lattice = torch.as_tensor(np.asarray(atoms.lattice_mat), dtype=dtype)
    if device is not None:
        positions = positions.to(device)
        lattice = lattice.to(device)
    device = positions.device
    n_atoms = int(positions.shape[0])

    src, dst, shift, r = torch_neighbor_list(
        positions=positions,
        lattice=lattice,
        cutoff=float(two_body_cutoff),
        max_neighbors=max_neighbors,
        atoms=atoms,
        use_matscipy_topology=use_matscipy_topology,
    )

    sps = np.array(
        [
            list(get_node_attributes(s, atom_features=atom_features))
            for s in atoms.elements
        ]
    )
    node_features = torch.as_tensor(sps, dtype=dtype, device=device)
    frac = torch.as_tensor(
        np.asarray(atoms.frac_coords), dtype=dtype, device=device
    )
    vol = torch.abs(torch.det(lattice))

    g = TorchGraph(
        num_nodes=n_atoms,
        src=src,
        dst=dst,
        ndata={
            "atom_features": node_features,
            "frac_coords": frac,
            "V": vol.expand(n_atoms),
            "Z": torch.as_tensor(
                np.asarray(atoms.atomic_numbers),
                dtype=torch.long,
                device=device,
            ),
        },
        edata={"r": r, "images": shift},
    )
    if use_lattice_prop:
        lp = np.array(
            [atoms.lattice.lat_lengths(), atoms.lattice.lat_angles()]
        ).flatten()
        g.ndata["extra_features"] = (
            torch.as_tensor(lp, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(n_atoms, -1)
            .contiguous()
        )

    if not compute_line_graph:
        return g

    # Three-body cutoff filter: only edges short enough join triplets.
    if three_body_cutoff < two_body_cutoff:
        with torch.no_grad():
            allowed = r.norm(dim=1) <= three_body_cutoff
    else:
        allowed = None

    lg_src, lg_dst = _line_graph_edges(src, dst, n_atoms, allowed=allowed)
    # Angle cosines at the shared atom, differentiable through r.
    h = torch_bond_cosines(r[lg_src], r[lg_dst])

    lg = TorchGraph(
        num_nodes=g.num_edges,  # line-graph nodes == parent edges
        src=lg_src,
        dst=lg_dst,
        # Share parent edata as ndata (like DGL's shared line graph).
        ndata={"r": r, "images": shift},
        edata={"h": h},
    )
    return g, lg


# =====================================================================
# Batching and DGL interop
# =====================================================================


def _batch_concat(
    graphs: List[TorchGraph],
    node_offset_source: Optional[List[int]] = None,
) -> TorchGraph:
    """Concatenate graphs into one, offsetting edge indices.

    ``node_offset_source`` is a list whose cumulative sum gives the
    offset applied to ``src`` / ``dst`` for each subgraph. When None,
    uses each subgraph's ``num_nodes`` (standard pair-graph batching).
    Set it to parent-graph edge counts when batching *line graphs*.
    """
    if len(graphs) == 0:
        raise ValueError("batch_torch_graphs: empty graph list.")
    device = graphs[0].src.device
    if node_offset_source is None:
        node_offset_source = [g.num_nodes for g in graphs]

    offsets = torch.zeros(len(graphs), dtype=torch.long, device=device)
    if len(graphs) > 1:
        offsets[1:] = torch.cumsum(
            torch.tensor(
                node_offset_source[:-1], dtype=torch.long, device=device
            ),
            dim=0,
        )

    srcs, dsts = [], []
    for g, off in zip(graphs, offsets):
        srcs.append(g.src + off)
        dsts.append(g.dst + off)

    # Only keep keys present in *all* subgraphs (avoid partial batches).
    ndata_keys = set(graphs[0].ndata)
    edata_keys = set(graphs[0].edata)
    for g in graphs[1:]:
        ndata_keys &= set(g.ndata)
        edata_keys &= set(g.edata)
    ndata = {
        k: torch.cat([g.ndata[k] for g in graphs], dim=0) for k in ndata_keys
    }
    edata = {
        k: torch.cat([g.edata[k] for g in graphs], dim=0) for k in edata_keys
    }

    return TorchGraph(
        num_nodes=sum(g.num_nodes for g in graphs),
        src=torch.cat(srcs),
        dst=torch.cat(dsts),
        ndata=ndata,
        edata=edata,
        batch_num_nodes=torch.tensor(
            [g.num_nodes for g in graphs], dtype=torch.long, device=device
        ),
        batch_num_edges=torch.tensor(
            [g.num_edges for g in graphs], dtype=torch.long, device=device
        ),
    )


def batch_torch_graphs(graphs: List[TorchGraph]) -> TorchGraph:
    """Batch pair graphs with offsets = cumulative ``num_nodes``."""
    return _batch_concat(graphs, node_offset_source=None)


def batch_torch_graph_pairs(
    pairs: List,
) -> "tuple[TorchGraph, TorchGraph]":
    """Batch a list of ``(g, lg)`` pairs with consistent offsetting.

    The line graph's src/dst index into the *parent* edge list, so its
    node-offset source is the list of parent num_edges, not num_nodes.
    """
    parents = [p[0] for p in pairs]
    lgs = [p[1] for p in pairs]
    g_batched = batch_torch_graphs(parents)
    lg_batched = _batch_concat(
        lgs, node_offset_source=[p.num_edges for p in parents]
    )
    return g_batched, lg_batched


# ---- DGL adapter (so the pure-torch model can consume DGL input) ----


def unbatch(g):
    """Polymorphic unbatch — works for TorchGraph *and* DGL graph.

    For a batched ``TorchGraph``, returns a list of single-graph
    ``TorchGraph`` views with ``src`` / ``dst`` reindexed to the local
    node range and ndata / edata sliced accordingly. For a DGL graph,
    delegates to ``dgl.unbatch``.
    """
    if isinstance(g, TorchGraph):
        if g.batch_num_nodes is None:
            return [g]
        dev = g.src.device
        B = g.batch_size
        node_off = torch.zeros(B + 1, dtype=torch.long, device=dev)
        node_off[1:] = torch.cumsum(g.batch_num_nodes, dim=0)
        edge_off = torch.zeros(B + 1, dtype=torch.long, device=dev)
        edge_off[1:] = torch.cumsum(g.batch_num_edges, dim=0)
        out = []
        for b in range(B):
            n0 = int(node_off[b].item())
            n1 = int(node_off[b + 1].item())
            e0 = int(edge_off[b].item())
            e1 = int(edge_off[b + 1].item())
            out.append(
                TorchGraph(
                    num_nodes=n1 - n0,
                    src=g.src[e0:e1] - n0,
                    dst=g.dst[e0:e1] - n0,
                    ndata={k: v[n0:n1] for k, v in g.ndata.items()},
                    edata={k: v[e0:e1] for k, v in g.edata.items()},
                )
            )
        return out
    import dgl as _dgl

    return _dgl.unbatch(g)


def torchgraph_from_dgl(g) -> TorchGraph:
    """Wrap a ``dgl.DGLGraph`` as a ``TorchGraph`` (zero-copy where possible).

    Batching metadata is preserved when the DGL graph is itself a batch.
    """
    src, dst = g.edges()
    ndata = {k: v for k, v in g.ndata.items()}
    edata = {k: v for k, v in g.edata.items()}
    bnn = None
    bne = None
    try:
        bnn_t = g.batch_num_nodes()
        bne_t = g.batch_num_edges()
        if bnn_t.numel() > 1:
            bnn = bnn_t.to(torch.long)
            bne = bne_t.to(torch.long)
    except Exception:
        pass
    return TorchGraph(
        num_nodes=int(g.num_nodes()),
        src=src.to(torch.long),
        dst=dst.to(torch.long),
        ndata=ndata,
        edata=edata,
        batch_num_nodes=bnn,
        batch_num_edges=bne,
    )
