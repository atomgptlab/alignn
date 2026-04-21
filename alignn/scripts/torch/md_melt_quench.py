"""Melt-quench MD using pure-PyTorch ALIGNN-FF + in-house NVTBerendsen.

Drop-in replacement for the ASE NVTBerendsen workflow. Same two-stage
schedule: heat at T0 (melt), cool to T1 (quench), write final POSCAR.

Usage (defaults to a small Si supercell so it runs anywhere):
    python md_melt_quench.py

For a trained model, pass --model-dir pointing to a directory with
`best_model.pt` + `config.json` that was trained with the
`alignn_atomwise` model. The state_dict is loaded into the pure-PyTorch
variant (layer names match by construction).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from ase import Atoms as AseAtoms
from ase.build import bulk
from ase.data import atomic_masses
from ase.io import write as ase_write

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)
from alignn.md import AlignnForces, NVTBerendsen, run, maxwell_boltzmann
from alignn.md.integrators import wrap_pbc, instantaneous_temperature


def ensure_cell_size(ase_atoms: AseAtoms, min_size: float):
    L = ase_atoms.get_cell().lengths()
    return [max(1, int(np.ceil(min_size / Li))) for Li in L]


def build_model(model_dir: str | None, device, dtype):
    if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
        cfg_raw = json.load(open(os.path.join(model_dir, "config.json")))
        mcfg = cfg_raw["model"]
        # force pure variant name so the config validates
        mcfg = {**mcfg, "name": "alignn_atomwise_pure"}
        model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**mcfg))
        sd_path = os.path.join(model_dir, "best_model.pt")
        sd = torch.load(sd_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] missing={len(missing)}  unexpected={len(unexpected)}")
        print(f"loaded trained weights from {sd_path}")
    else:
        print("[note] no model_dir given — using random-init model")
        model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(
            name="alignn_atomwise_pure",
            calculate_gradient=True,
            atomwise_output_features=0,
            atom_input_features=92,
        ))
    return model.to(device).to(dtype)


def make_starting_atoms(args) -> AseAtoms:
    # Try JARVIS figshare if jid given; otherwise a built-in Si cubic cell.
    if args.jid:
        from jarvis.db.figshare import get_jid_data
        from jarvis.core.atoms import Atoms as JAtoms
        d = get_jid_data(jid=args.jid, dataset="dft_3d")
        ja = JAtoms.from_dict(d["atoms"]).get_conventional_atoms
        ase_atoms = ja.ase_converter()
    else:
        ase_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    dims = ensure_cell_size(ase_atoms, args.min_size)
    ase_atoms = ase_atoms.repeat(dims)
    print(f"starting cell: {len(ase_atoms)} atoms, dims={dims}")
    return ase_atoms


def write_poscar(positions_t, Z, cell_np, path):
    ase_atoms = AseAtoms(numbers=Z, positions=positions_t.detach().cpu().numpy(),
                        cell=cell_np, pbc=True)
    ase_write(path, ase_atoms, format="vasp")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jid", default=None, help="JARVIS id, e.g. JVASP-1002")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--outdir", default="output_melt")
    ap.add_argument("--min-size", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1.0, help="fs")
    ap.add_argument("--temp0", type=float, default=3500.0)
    ap.add_argument("--nsteps0", type=int, default=200)
    ap.add_argument("--temp1", type=float, default=300.0)
    ap.add_argument("--nsteps1", type=int, default=400)
    ap.add_argument("--taut", type=float, default=20.0, help="fs")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ase_atoms = make_starting_atoms(args)
    Z = ase_atoms.get_atomic_numbers()
    cell_np = np.array(ase_atoms.cell)
    cell = torch.tensor(cell_np, dtype=dtype, device=device)
    positions = torch.tensor(ase_atoms.get_positions(), dtype=dtype, device=device)
    masses = torch.tensor([atomic_masses[z] for z in Z], dtype=dtype, device=device)

    model = build_model(args.model_dir, device, dtype)
    forces_fn = AlignnForces(model, Z, cell_np, device=device, dtype=dtype)

    # Initialize velocities at T0
    gen = torch.Generator(device=device).manual_seed(0)
    velocities = maxwell_boltzmann(masses, T=args.temp0, generator=gen)

    # Wrap integrator step to also PBC-wrap positions (the model's graph
    # builder uses the ASE atoms object, which handles PBC internally; we
    # wrap here mostly to keep output POSCARs clean).
    integ = NVTBerendsen(forces_fn=forces_fn, masses=masses, dt=args.dt,
                        T=args.temp0, taut=args.taut)

    t_start = time.time()
    jid_tag = args.jid or "Si"

    # --- Stage 1: melt at T0 ---
    print(f"\n--- MELT @ {args.temp0} K for {args.nsteps0} steps ---")
    def cb(i, row): print(f"[melt]  step={row['step']:5d}  t={row['time_fs']:7.1f} fs  "
                         f"T={row['T_K']:7.1f} K  wall={row['wall_s']:6.1f} s")
    positions, velocities, _ = run(integ, positions, velocities,
                                   nsteps=args.nsteps0,
                                   log_every=args.log_every, callback=cb)
    positions = wrap_pbc(positions, cell)

    # --- Stage 2: quench to T1 ---
    integ.set_temperature(args.temp1)
    print(f"\n--- QUENCH to {args.temp1} K for {args.nsteps1} steps ---")
    def cb2(i, row): print(f"[quench] step={row['step']:5d}  t={row['time_fs']:7.1f} fs  "
                          f"T={row['T_K']:7.1f} K  wall={row['wall_s']:6.1f} s")
    positions, velocities, _ = run(integ, positions, velocities,
                                   nsteps=args.nsteps1,
                                   log_every=args.log_every, callback=cb2)
    positions = wrap_pbc(positions, cell)

    T_final = instantaneous_temperature(masses, velocities).item()
    print(f"\nDone. final T = {T_final:.1f} K   total wall = {time.time()-t_start:.1f} s")

    poscar = os.path.join(args.outdir, f"POSCAR_{jid_tag}_quenched_alignn.vasp")
    write_poscar(positions, Z, cell_np, poscar)


if __name__ == "__main__":
    main()
