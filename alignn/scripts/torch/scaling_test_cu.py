"""ALIGNN-FF (pure PyTorch) scaling test on a Cu FCC supercell.

Sweeps supercell size 1x1x1 .. NxNxN across fp32 / bf16-autocast / fp16-autocast.
Records peak GPU memory and wall time for a single forward+backward (E, F).

Usage:
    python scaling_test_cu.py --max 10 --model-dir alignn/ff/alignnff_wt01
"""

from __future__ import annotations
import argparse, gc, json, os, time, traceback
import numpy as np
import torch
from ase.build import bulk
from jarvis.core.atoms import ase_to_atoms as ase_to_jarvis

from alignn.graphs import Graph
from alignn.torch_graph_builder import torchgraph_from_dgl
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)


def build_cu_supercell(n: int):
    a = bulk("Cu", "fcc", a=3.615, cubic=True)  # 4 atoms / cubic cell
    return a.repeat((n, n, n))


def build_graphs(ase_atoms, cutoff=8.0, max_neighbors=12, device="cuda"):
    j = ase_to_jarvis(ase_atoms)
    g, lg = Graph.atom_dgl_multigraph(
        j, neighbor_strategy="k-nearest",
        cutoff=cutoff, max_neighbors=max_neighbors,
        atom_features="cgcnn", use_canonize=True,
    )
    g = g.to(device); lg = lg.to(device)
    tg = torchgraph_from_dgl(g)
    tlg = torchgraph_from_dgl(lg)
    return tg, tlg


def make_model(device):
    cfg = ALIGNNAtomWisePureConfig(
        name="alignn_atomwise_pure",
        calculate_gradient=True,
        atomwise_output_features=0,
        atom_input_features=92,
    )
    m = ALIGNNAtomWisePure(cfg).to(device)
    m.eval()
    return m


def _cast_graph(tg, dtype):
    for k, v in list(tg.ndata.items()):
        if v.is_floating_point():
            tg.ndata[k] = v.to(dtype)
    for k, v in list(tg.edata.items()):
        if v.is_floating_point():
            tg.edata[k] = v.to(dtype)


def run_one(model, tg, tlg, cell, device, precision, model_fp32_state, fp32_snapshot):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(torch.float32)
    model.load_state_dict(model_fp32_state)
    # restore graph tensors from fp32 snapshot
    ndata0, edata0, lndata0, ledata0 = fp32_snapshot
    tg.ndata = {k: v.detach().clone() for k, v in ndata0.items()}
    tg.edata = {k: v.detach().clone() for k, v in edata0.items()}
    tlg.ndata = {k: v.detach().clone() for k, v in lndata0.items()}
    tlg.edata = {k: v.detach().clone() for k, v in ledata0.items()}
    cell = cell.to(torch.float32)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        if precision == "fp32":
            out = model((tg, tlg, cell))
        elif precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model((tg, tlg, cell))
        elif precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model((tg, tlg, cell))
        elif precision == "bf16_pure":
            model.to(torch.bfloat16)
            _cast_graph(tg, torch.bfloat16)
            _cast_graph(tlg, torch.bfloat16)
            out = model((tg, tlg, cell.to(torch.bfloat16)))
        elif precision == "fp16_pure":
            model.to(torch.float16)
            _cast_graph(tg, torch.float16)
            _cast_graph(tlg, torch.float16)
            out = model((tg, tlg, cell.to(torch.float16)))
        else:
            raise ValueError(precision)
        # Force backward to capture gradient memory
        e = out["out"].sum() if "out" in out else out.get("total_energy", None)
        if e is None:  # fallback
            e = next(iter(out.values())).sum()
        e.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return {"ok": True, "time_s": dt, "peak_gb": peak}
    except torch.cuda.OutOfMemoryError as ex:
        return {"ok": False, "err": "OOM"}
    except Exception as ex:
        return {"ok": False, "err": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--out", default="scaling_cu_results.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}  total={total_gb:.1f} GB")

    results = {"gpu": gpu, "total_gb": total_gb, "runs": []}
    model = make_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f} M")
    import copy
    model_fp32_state = copy.deepcopy(model.state_dict())

    for n in range(args.min, args.max + 1):
        atoms = build_cu_supercell(n)
        N = len(atoms)
        try:
            tg, tlg = build_graphs(atoms, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"[{n}^3 N={N}] graph build OOM"); break
        except Exception as ex:
            print(f"[{n}^3 N={N}] graph build failed: {ex}"); break
        n_edges = tg.src.numel(); n_triplets = tlg.src.numel()
        cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32, device=device)
        print(f"\n=== {n}x{n}x{n}  N={N}  edges={n_edges}  triplets={n_triplets} ===")
        row = {"n": n, "N": N, "edges": n_edges, "triplets": n_triplets}
        snap = (
            {k: v.detach().clone() for k, v in tg.ndata.items()},
            {k: v.detach().clone() for k, v in tg.edata.items()},
            {k: v.detach().clone() for k, v in tlg.ndata.items()},
            {k: v.detach().clone() for k, v in tlg.edata.items()},
        )
        for prec in ["fp32", "bf16", "fp16", "bf16_pure", "fp16_pure"]:
            r = run_one(model, tg, tlg, cell, device, prec, model_fp32_state, snap)
            row[prec] = r
            tag = f"{r['peak_gb']:.2f} GB  {r['time_s']*1000:.0f} ms" if r["ok"] else r.get("err","?")
            print(f"  {prec:5s}: {tag}")
        results["runs"].append(row)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        # stop if all precisions OOM
        if not any(row[p]["ok"] for p in ["fp32","bf16","fp16","bf16_pure","fp16_pure"]):
            print("all precisions OOM — stopping"); break
        del tg, tlg, cell; torch.cuda.empty_cache(); gc.collect()

    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
