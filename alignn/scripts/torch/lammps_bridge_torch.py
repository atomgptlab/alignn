"""Drive LAMMPS with ALIGNN-FF via direct torch calls (no ASE wrapper).

Loads `ALIGNNAtomWisePure` from OutputDir/ and registers a LAMMPS
fix-external callback that:
  1. Takes positions/cell from LAMMPS each step
  2. Builds the atom + line graph via jarvis (CPU, one-shot per step)
  3. Runs the pure-PyTorch model on GPU
  4. Returns forces (+ virial) to LAMMPS

This bypasses AlignnAtomwiseCalculator + ASE entirely. Next stop for
speed: torch.jit.script the model (see `jit_compile()` below) or write
a C++ pair style.

Usage:
    python scripts/torch/lammps_bridge_torch.py \\
        --model-dir OutputDir --input melt_quench.in --types Si
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch

from ase import Atoms as AseAtoms
from jarvis.core.atoms import ase_to_atoms as ase_to_jarvis

from alignn.graphs import Graph
from alignn.torch_graph_builder import torchgraph_from_dgl
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)

# eV/Å³ -> bar (LAMMPS 'metal' units)
EV_PER_A3_TO_BAR = 1.602176634e6


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pure_model(model_dir: str, device, dtype=torch.float32):
    cfg_raw = json.load(open(f"{model_dir}/config.json"))
    mcfg = dict(cfg_raw["model"])
    mcfg["name"] = "alignn_atomwise_pure"
    # make sure forces + stress autograd paths are on
    mcfg["calculate_gradient"] = True
    if "stresswise_weight" in mcfg and mcfg["stresswise_weight"] == 0:
        mcfg["stresswise_weight"] = 0.01
    cfg = ALIGNNAtomWisePureConfig(**mcfg)
    model = ALIGNNAtomWisePure(cfg)
    sd = torch.load(f"{model_dir}/best_model.pt", map_location=device,
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).to(dtype).eval()
    return model, cfg_raw


def jit_compile(model):
    """Optional: torch.jit.script the model for Path B."""
    try:
        return torch.jit.script(model)
    except Exception as e:
        print(f"[warn] jit.script failed: {e}\n       falling back to eager")
        return model


# ---------------------------------------------------------------------------
# Graph build — currently jarvis/DGL, the CPU bottleneck.
# For large systems, replace with an on-device neighbor list.
# ---------------------------------------------------------------------------
def build_graph_on_device(symbols, positions, cell, cutoff, max_neighbors,
                          device, dtype):
    ase_atoms = AseAtoms(symbols=symbols, positions=positions,
                         cell=cell, pbc=True)
    j = ase_to_jarvis(ase_atoms)
    g, lg = Graph.atom_dgl_multigraph(
        j, neighbor_strategy="k-nearest",
        cutoff=cutoff, max_neighbors=max_neighbors,
        atom_features="cgcnn", use_canonize=True,
    )
    g = g.to(device); lg = lg.to(device)
    tg = torchgraph_from_dgl(g); tlg = torchgraph_from_dgl(lg)
    for d in (tg.ndata, tg.edata, tlg.ndata, tlg.edata):
        for k, v in list(d.items()):
            if v.is_floating_point():
                d[k] = v.to(dtype)
    return tg, tlg


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------
def make_callback(model, cfg, symbols_by_type, device, dtype, compute_virial=True):
    cutoff = float(cfg.get("cutoff", 8.0))
    max_neighbors = int(cfg.get("max_neighbors", 12))
    step_counter = [0]

    def callback(lmp, ntimestep, nlocal, tag, x, f):
        # --- unpack box from LAMMPS ---
        boxlo, boxhi, xy, yz, xz, *_ = lmp.extract_box()
        cell = np.array([
            [boxhi[0] - boxlo[0], 0.0, 0.0],
            [xy, boxhi[1] - boxlo[1], 0.0],
            [xz, yz, boxhi[2] - boxlo[2]],
        ])
        # --- unpack atom types + positions ---
        types = np.ctypeslib.as_array(lmp.extract_atom("type"), shape=(nlocal,))
        symbols = [symbols_by_type[t] for t in types]
        positions = np.array(x, dtype=np.float64)

        # --- build graph ---
        tg, tlg = build_graph_on_device(symbols, positions, cell,
                                        cutoff, max_neighbors, device, dtype)
        cell_t = torch.tensor(cell, dtype=dtype, device=device)

        # --- forward ---
        with torch.enable_grad():
            out = model((tg, tlg, cell_t))

        # energy + forces
        energy = out["out"].sum().item()
        forces = out["grad"].detach().cpu().numpy()
        f[:] = forces

        # virial (optional)
        if compute_virial and "stresses" in out:
            stress_voigt = out["stresses"].detach().cpu().numpy().reshape(-1)[:6]
            # stresses returned in eV/Å³, Voigt order xx yy zz yz xz xy
            volume = float(np.linalg.det(cell))
            virial_bar = -stress_voigt * volume * EV_PER_A3_TO_BAR
            v6 = [
                float(virial_bar[0]), float(virial_bar[1]), float(virial_bar[2]),
                float(virial_bar[5]), float(virial_bar[4]), float(virial_bar[3]),
            ]
            try:
                lmp.fix_external_set_virial_global("alignn", v6)
            except Exception:
                pass

        step_counter[0] += 1
        if step_counter[0] % 50 == 0:
            fmax = float(np.max(np.abs(forces)))
            print(f"[torch-bridge] step={step_counter[0]:6d}  "
                  f"E={energy: .4f} eV  |F|max={fmax:.3f} eV/Å")

    return callback


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(args):
    from lammps import lammps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model, cfg_raw = load_pure_model(args.model_dir, device, dtype)
    if args.jit:
        model = jit_compile(model)

    symbols_by_type = {i + 1: s for i, s in enumerate(args.types.split(","))}

    lmp = lammps()
    if args.input:
        with open(args.input) as fh:
            lines = fh.readlines()
        cb = make_callback(model, cfg_raw, symbols_by_type, device, dtype,
                          compute_virial=not args.no_virial)
        registered = False
        for line in lines:
            stripped = line.strip()
            if (not registered and stripped.startswith("run ")
                and "alignn" in "".join(lines[: lines.index(line)])):
                lmp.set_fix_external_callback("alignn", cb, lmp)
                registered = True
            lmp.command(line.rstrip("\n"))
    else:
        raise SystemExit("Provide --input with a LAMMPS script")
    lmp.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="dir with best_model.pt + config.json")
    p.add_argument("--input", required=True, help="LAMMPS input script")
    p.add_argument("--types", required=True, help="comma-separated element symbols by LAMMPS type")
    p.add_argument("--jit", action="store_true", help="try torch.jit.script the model")
    p.add_argument("--no-virial", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
