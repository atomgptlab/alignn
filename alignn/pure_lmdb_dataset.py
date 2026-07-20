"""LMDB-backed dataset yielding TorchGraph objects (no DGL).

Mirrors the API of ``alignn.lmdb_dataset`` but serializes
``TorchGraph`` dataclasses via pickle, and batches them with
``batch_torch_graph_pairs`` at the DataLoader boundary.
"""

from __future__ import annotations

import os
import pickle as pk
from typing import List, Tuple

import lmdb
import numpy as np
import torch
from jarvis.core.atoms import Atoms
from torch.utils.data import Dataset
from tqdm import tqdm

from alignn.graphs import Graph
from alignn.torch_graph_builder import (
    TorchGraph,
    batch_torch_graph_pairs,
    batch_torch_graphs,
    torchgraph_from_dgl,
)

# How often to commit the LMDB write transaction during a fresh build.
# Holding a single transaction across all entries keeps every dirty page
# resident in RAM until commit; flushing periodically caps peak memory.
LMDB_COMMIT_EVERY = 100000


def _mem_gb():
    """Resident memory of the current process in GB (for diagnostics)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float("nan")


def prepare_pure_batch(batch, device=None, non_blocking=False):
    """Move a pure-torch batch to device; mirrors prepare_line_graph_batch."""
    g, lg, t, id = batch
    return (
        (g.to(device), lg.to(device)),
        t.to(device, non_blocking=non_blocking),
    )


def _graph_to_torchgraph(g, lg=None):
    """Accept DGL or TorchGraph; always return TorchGraph (pair)."""
    if isinstance(g, TorchGraph):
        return g, (lg if lg is not None else None)
    tg = torchgraph_from_dgl(g)
    tlg = torchgraph_from_dgl(lg) if lg is not None else None
    return tg, tlg


class PureTorchLMDBDataset(Dataset):
    """Dataset of crystal TorchGraphs using LMDB."""

    def __init__(self, lmdb_path: str = "", line_graph: bool = True, ids=None):
        """Open the LMDB at ``lmdb_path`` read-only."""
        super().__init__()
        self.lmdb_path = lmdb_path
        self.ids = ids or []
        self.line_graph = line_graph
        # Open lazily: an lmdb.Environment cannot be pickled, so holding one
        # here breaks DataLoader(num_workers>0), which pickles the dataset to
        # each worker. Each process opens its own handle on first access.
        self.env = None
        _env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with _env.begin() as txn:
            self.length = txn.stat()["entries"]
        _env.close()
        self.prepare_batch = prepare_pure_batch

    def _get_env(self):
        """Return this process's LMDB handle, opening it on first use."""
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        return self.env

    def __getstate__(self):
        """Drop the unpicklable LMDB handle when sent to a worker."""
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __len__(self):
        """Return the number of records in the LMDB."""
        return self.length

    def __getitem__(self, idx):
        """Load and unpickle the ``idx``-th record."""
        with self._get_env().begin() as txn:
            serialized = txn.get(f"{idx}".encode())
        if self.line_graph:
            g, lg, lattice, label = pk.loads(serialized)
            return g, lg, lattice, label
        g, lattice, label = pk.loads(serialized)
        return g, lattice, label

    def close(self):
        """Close the LMDB environment (idempotent)."""
        try:
            self.env.close()
        except Exception:
            pass

    def __del__(self):
        """Ensure the LMDB handle is released on GC."""
        self.close()

    @staticmethod
    def collate(samples: List[Tuple[TorchGraph, torch.Tensor, torch.Tensor]]):
        """Collate ``(g, lattice, label)`` samples into a batched triple."""
        graphs, lattices, labels = map(list, zip(*samples))
        batched = batch_torch_graphs(graphs)
        if labels[0].dim() > 0:
            return batched, torch.stack(lattices), torch.stack(labels)
        return batched, torch.stack(lattices), torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[
            Tuple[TorchGraph, TorchGraph, torch.Tensor, torch.Tensor]
        ],
    ):
        """Collate ``(g, lg, lattice, label)`` samples into a batched tuple."""
        graphs, line_graphs, lattices, labels = map(list, zip(*samples))
        g_b, lg_b = batch_torch_graph_pairs(list(zip(graphs, line_graphs)))
        if labels[0].dim() > 0:
            return g_b, lg_b, torch.stack(lattices), torch.stack(labels)
        return g_b, lg_b, torch.stack(lattices), torch.tensor(labels)


def _attach_node_payload(
    g: TorchGraph, key: str, value: np.ndarray, natoms: int
):
    """Tile a per-structure (global) tensor across nodes, like the DGL loader.

    All callers pass a per-structure quantity (stress 3x3, extra_features,
    additional_output) that must be broadcast to every node; genuine per-node
    arrays (forces, atomwise targets) are assigned to ``g.ndata`` directly.
    We therefore always tile. A previous ``shape[0] == natoms`` early-return
    mis-fired for a 3x3 stress on a 3-atom cell (3 == 3), storing it as a
    2-D per-node array and crashing batch collation (``got 2 and 3``) once
    the batch mixed 3-atom and non-3-atom structures.
    """
    dtype = torch.get_default_dtype()
    arr = np.asarray(value)
    tiled = np.broadcast_to(arr, (natoms,) + arr.shape).copy()
    g.ndata[key] = torch.as_tensor(tiled, dtype=dtype)


def _lmdb_is_usable(tmp_name: str) -> bool:
    """Return True if the LMDB at ``tmp_name`` contains at least one entry.

    Guards against the empty-stub left behind when a previous build was
    interrupted (OOM, Ctrl-C, etc.). Such stubs have an 8KB data.mdb but
    zero records, and the read_existing fast path would silently return
    a zero-length dataset if not caught.
    """
    if not os.path.exists(tmp_name):
        return False
    try:
        env = lmdb.open(tmp_name, readonly=True, lock=False)
        with env.begin() as txn:
            n_entries = txn.stat()["entries"]
        env.close()
        return n_entries > 0
    except Exception:
        return False


def get_torch_dataset(
    dataset=None,
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    target_additional_output="",
    neighbor_strategy="pure_torch",
    atom_features="cgcnn",
    use_canonize="",
    name="",
    line_graph=True,
    cutoff=8.0,
    cutoff_extra=3.0,
    max_neighbors=12,
    three_body_cutoff=None,
    classification=False,
    sampler=None,
    output_dir=".",
    tmp_name="dataset",
    map_size=1e12,
    read_existing=False,
    dtype="float32",
):
    """Build or load a PureTorchLMDBDataset from a list of records."""
    dataset = dataset or []
    vals = np.array([ii[target] for ii in dataset])
    print("data range", np.max(vals), np.min(vals))
    print("line_graph", line_graph)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, tmp_name + "_data_range"), "w") as f:
        f.write(f"Max={np.max(vals)}\nMin={np.min(vals)}\n")

    print(f"[MEM] {tmp_name}: entry: {_mem_gb():.2f} GB")

    # Fast path: reuse a previously built cache, but only if it actually
    # contains records. An empty stub (8KB data.mdb) from an interrupted
    # build would otherwise be returned as a zero-length dataset.
    if read_existing and _lmdb_is_usable(tmp_name):
        # Validate the cache was built for the pure-torch backend.
        _env = lmdb.open(tmp_name, readonly=True, lock=False)
        with _env.begin() as _txn:
            _probe = _txn.get(b"0")
        _env.close()
        if _probe is not None:
            _sample = pk.loads(_probe)
            if not isinstance(_sample[0], TorchGraph):
                raise RuntimeError(
                    f"LMDB cache at '{tmp_name}' contains "
                    f"{type(_sample[0]).__name__} records, not "
                    "TorchGraph. Delete the stale cache (e.g. "
                    f"`rm -rf {tmp_name}`) and rerun — it was built "
                    "for a different model backend."
                )
        ids = [d[id_tag] for d in dataset]
        print("Reading dataset", tmp_name)
        ds = PureTorchLMDBDataset(
            lmdb_path=tmp_name, line_graph=line_graph, ids=ids
        )
        print(
            f"[MEM] {tmp_name}: after PureTorchLMDBDataset (cached): "
            f"{_mem_gb():.2f} GB"
        )
        return ds

    # If read_existing was requested but the cache is unusable (empty
    # stub or missing), warn so the user knows we're rebuilding.
    if read_existing and os.path.exists(tmp_name):
        print(
            f"[WARN] {tmp_name}: read_existing=True but cache is "
            "empty or unreadable; rebuilding from scratch."
        )

    ids = []
    # Fresh build: wipe any pre-existing cache dir to avoid mixing old
    # records (possibly from a different model backend) with new writes.
    if os.path.exists(tmp_name):
        import shutil

        shutil.rmtree(tmp_name)
    env = lmdb.open(tmp_name, map_size=int(map_size))

    print(f"[MEM] {tmp_name}: before write loop: {_mem_gb():.2f} GB")

    # Use periodic commits instead of a single giant write transaction.
    # A single txn around millions of put() calls keeps every dirty page
    # in RAM until commit and was the cause of OOM on memory-tight hosts.
    txn = env.begin(write=True)
    try:
        for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            ids.append(d[id_tag])
            atoms = Atoms.from_dict(d["atoms"])
            raw = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=float(cutoff),
                max_neighbors=max_neighbors,
                atom_features=atom_features,
                compute_line_graph=line_graph,
                use_canonize=use_canonize,
                cutoff_extra=cutoff_extra,
                neighbor_strategy=neighbor_strategy,
                three_body_cutoff=three_body_cutoff,
                dtype=dtype,
            )
            if line_graph:
                g, lg = raw
                g, lg = _graph_to_torchgraph(g, lg)
            else:
                g = raw
                g, _ = _graph_to_torchgraph(g)

            lattice = torch.as_tensor(
                atoms.lattice_mat, dtype=torch.get_default_dtype()
            )
            label = torch.as_tensor(d[target], dtype=torch.get_default_dtype())
            natoms = len(d["atoms"]["elements"])
            if classification:
                label = label.long()
            if "extra_features" in d:
                _attach_node_payload(
                    g,
                    "extra_features",
                    np.asarray(d["extra_features"]),
                    natoms,
                )
            if target_atomwise:
                g.ndata[target_atomwise] = torch.as_tensor(
                    np.asarray(d[target_atomwise]),
                    dtype=torch.get_default_dtype(),
                )
            if target_grad:
                arr = np.asarray(d[target_grad])
                g.ndata[target_grad] = torch.as_tensor(
                    arr, dtype=torch.get_default_dtype()
                )
            if target_stress:
                stress = np.asarray(d[target_stress])
                _attach_node_payload(g, target_stress, stress, natoms)
            if target_additional_output:
                _attach_node_payload(
                    g,
                    target_additional_output,
                    np.asarray(d[target_additional_output]),
                    natoms,
                )

            if line_graph:
                txn.put(f"{idx}".encode(), pk.dumps((g, lg, lattice, label)))
            else:
                txn.put(f"{idx}".encode(), pk.dumps((g, lattice, label)))

            # Drop this entry's full atomic-structure dict now that it's
            # serialized into LMDB. The caller's list survives but its
            # contents are released, freeing ~10-30 KB per entry. For
            # 1.5M-entry datasets this is the difference between sitting
            # at 24+ GB during the loop and staying at 3-4 GB.
            dataset[idx] = None

            # Flush dirty pages to disk every LMDB_COMMIT_EVERY entries.
            if (idx + 1) % LMDB_COMMIT_EVERY == 0:
                txn.commit()
                txn = env.begin(write=True)
                if (idx + 1) % (LMDB_COMMIT_EVERY * 5) == 0:
                    print(
                        f"[MEM] {tmp_name}: after {idx + 1} entries: "
                        f"{_mem_gb():.2f} GB"
                    )
    finally:
        # Commit whatever is left in the final partial batch.
        txn.commit()

    env.close()

    print(f"[MEM] {tmp_name}: after write loop: {_mem_gb():.2f} GB")

    ds = PureTorchLMDBDataset(
        lmdb_path=tmp_name, line_graph=line_graph, ids=ids
    )

    print(
        f"[MEM] {tmp_name}: after PureTorchLMDBDataset wrap: "
        f"{_mem_gb():.2f} GB"
    )

    return ds
