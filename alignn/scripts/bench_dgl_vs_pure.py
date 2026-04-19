"""Scaling benchmark: DGL ALIGNN vs pure-torch ALIGNN.

Builds Cu-FCC supercells of increasing size, runs one forward + backward
(energy + forces + stress) through each backend, and reports time and
peak memory.

Both backends receive the *same* graph topology: we build a single DGL
graph via matscipy (``neighbor_strategy="fast_graph"``) and feed it to
``ALIGNNAtomWise`` directly; the pure-torch model gets the same graph
converted to ``TorchGraph`` at the model boundary. So the numbers
below isolate **model-level** cost (message passing, autograd,
readout) — not the neighbor-list step.

Usage
-----
    python alignn/scripts/bench_dgl_vs_pure.py
    python alignn/scripts/bench_dgl_vs_pure.py \\
        --sizes 1,2,3,4,5 --cutoff 4.0 --hidden 64 \\
        --alignn-layers 1 --gcn-layers 1 --repeat 3

Writes a printed table; with ``--output foo`` also saves ``foo.npz``
and, if matplotlib is available, ``foo.png``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ase.build.supercells import make_supercell
from jarvis.io.vasp.inputs import Poscar

from alignn.graphs import Graph
from alignn.models.alignn_atomwise import (
    ALIGNNAtomWise,
    ALIGNNAtomWiseConfig,
)
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)
from alignn.torch_graph_builder import torchgraph_from_dgl


CU_POSCAR = """Cu
1.0
3.6 0.0 0.0
0.0 3.6 0.0
0.0 0.0 3.6
Cu
4
direct
0.0 0.0 0.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
"""


def _build_models(args, device) -> "tuple[torch.nn.Module, torch.nn.Module]":
    """Construct both backends from the same hyperparameters and weights."""
    common = dict(
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        atom_input_features=92,
        hidden_features=args.hidden,
        output_features=1,
        calculate_gradient=True,
        graphwise_weight=0.85,
        gradwise_weight=0.05,
        stresswise_weight=0.05,
        atomwise_weight=0.0,
        add_reverse_forces=True,
        lg_on_fly=True,
        use_penalty=True,
    )
    torch.manual_seed(0)
    m_dgl = ALIGNNAtomWise(
        ALIGNNAtomWiseConfig(name="alignn_atomwise", **common)
    ).to(device)
    torch.manual_seed(0)
    m_pure = ALIGNNAtomWisePure(
        ALIGNNAtomWisePureConfig(name="alignn_atomwise_pure", **common)
    ).to(device)
    # Copy DGL weights into pure so any difference is purely mechanical.
    sd_dgl = m_dgl.state_dict()
    sd_pure = m_pure.state_dict()
    for k in sd_pure:
        if k in sd_dgl and sd_dgl[k].shape == sd_pure[k].shape:
            sd_pure[k] = sd_dgl[k].clone()
    m_pure.load_state_dict(sd_pure, strict=False)
    m_dgl.train()
    m_pure.train()
    return m_dgl, m_pure


def _build_graph(atoms, cutoff: float, max_neighbors: int):
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        neighbor_strategy="fast_graph",
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="cgcnn",
    )
    return g, lg


def _run_once(
    model,
    graphs,
    lat,
    device: torch.device,
    is_pure: bool,
):
    """One forward + backward. Returns elapsed seconds and peak memory (bytes)."""
    g, lg = graphs
    if is_pure:
        g_in, lg_in = torchgraph_from_dgl(g), torchgraph_from_dgl(lg)
    else:
        g_in, lg_in = g, lg
    lat_in = lat

    # Sync + reset peak memory.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    out = model((g_in, lg_in, lat_in))
    loss = (
        out["out"].sum()
        + out["grad"].pow(2).sum()
        + out["stresses"].pow(2).sum()
    )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    # Free autograd graph / grads before next iter.
    model.zero_grad(set_to_none=True)
    del out, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dt, int(peak)


def _bench_one_size(
    model,
    graphs,
    lat,
    device,
    is_pure: bool,
    warmup: int,
    repeat: int,
):
    """Warm up then time ``repeat`` iterations; return median time + peak mem."""
    for _ in range(warmup):
        _run_once(model, graphs, lat, device, is_pure)
    times, peaks = [], []
    for _ in range(repeat):
        t, p = _run_once(model, graphs, lat, device, is_pure)
        times.append(t)
        peaks.append(p)
    return float(np.median(times)), max(peaks)


def run(args) -> Dict[str, np.ndarray]:
    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    m_dgl, m_pure = _build_models(args, device)
    n_params = sum(p.numel() for p in m_dgl.parameters())
    print(f"model parameters: {n_params}")

    base = Poscar.from_string(CU_POSCAR).atoms.ase_converter()
    sizes = [int(s) for s in args.sizes.split(",")]

    rows: List[dict] = []
    print()
    hdr = (
        f"{'N':>3}  {'atoms':>6}  {'edges':>8}  "
        f"{'DGL ms':>10}  {'pure ms':>10}  {'speedup':>8}  "
        f"{'DGL MB':>10}  {'pure MB':>10}  {'mem ratio':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in sizes:
        sc = make_supercell(base, [[n, 0, 0], [0, n, 0], [0, 0, n]])
        # Convert to jarvis Atoms for the neighbor-list pipeline.
        from jarvis.core.atoms import Atoms as JAtoms

        jatoms = JAtoms(
            lattice_mat=np.asarray(sc.cell),
            coords=np.asarray(sc.positions),
            elements=list(sc.get_chemical_symbols()),
            cartesian=True,
        )
        try:
            g, lg = _build_graph(jatoms, args.cutoff, args.max_neighbors)
        except Exception as exc:
            print(f"  graph build failed at N={n}: {exc}")
            break
        lat = torch.as_tensor(
            np.asarray(jatoms.lattice_mat), dtype=torch.float32, device=device
        )
        g = g.to(device)
        lg = lg.to(device)

        try:
            t_dgl, m_dgl_peak = _bench_one_size(
                m_dgl, (g, lg), lat, device, False, args.warmup, args.repeat
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  DGL OOM/error at N={n}: {type(exc).__name__}")
            break
        try:
            t_pure, m_pure_peak = _bench_one_size(
                m_pure, (g, lg), lat, device, True, args.warmup, args.repeat
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  pure OOM/error at N={n}: {type(exc).__name__}")
            break

        n_atoms = int(sc.positions.shape[0])
        n_edges = int(g.num_edges())
        speedup = t_dgl / t_pure if t_pure > 0 else float("nan")
        mem_ratio = (
            m_pure_peak / m_dgl_peak if m_dgl_peak > 0 else float("nan")
        )
        print(
            f"{n:>3}  {n_atoms:>6}  {n_edges:>8}  "
            f"{t_dgl * 1000:>10.2f}  {t_pure * 1000:>10.2f}  "
            f"{speedup:>8.2f}x  "
            f"{m_dgl_peak / 1024**2:>10.1f}  "
            f"{m_pure_peak / 1024**2:>10.1f}  "
            f"{mem_ratio:>10.2f}x"
        )
        rows.append(
            dict(
                n=n,
                atoms=n_atoms,
                edges=n_edges,
                dgl_time_s=t_dgl,
                pure_time_s=t_pure,
                dgl_peak_bytes=m_dgl_peak,
                pure_peak_bytes=m_pure_peak,
            )
        )

    result = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}

    if args.output and rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out.with_suffix(".npz")), **result)
        print(f"\nSaved data to {out.with_suffix('.npz')}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            n_arr = result["atoms"]
            ax1.plot(
                n_arr, result["dgl_time_s"] * 1000, "-o", label="DGL"
            )
            ax1.plot(
                n_arr, result["pure_time_s"] * 1000, "-s", label="pure-torch"
            )
            ax1.set(
                xlabel="# atoms", ylabel="time / iter (ms)",
                title="Forward + backward (E+F+σ)",
            )
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend()

            ax2.plot(n_arr, result["dgl_peak_bytes"] / 1024**2, "-o", label="DGL")
            ax2.plot(
                n_arr, result["pure_peak_bytes"] / 1024**2, "-s",
                label="pure-torch",
            )
            ax2.set(
                xlabel="# atoms", ylabel="peak memory (MB)",
                title="Peak GPU memory",
            )
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid(True, which="both", alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            png = out.with_suffix(".png")
            plt.savefig(str(png), dpi=160)
            print(f"Saved plot to {png}")
        except ImportError:
            pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", default="1,2,3,4,5",
        help="Comma-separated supercell factors (N -> 4*N^3 atoms).",
    )
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--max-neighbors", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--alignn-layers", type=int, default=1)
    ap.add_argument("--gcn-layers", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path stem; writes <stem>.npz and (if matplotlib) <stem>.png.",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
