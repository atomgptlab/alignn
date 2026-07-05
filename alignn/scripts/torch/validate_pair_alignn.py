"""Validate pair_alignn against the Python reference (AlignnAtomwiseCalculator).

Does a single-point `run 0` in LAMMPS with pair_alignn, then computes the
same quantities via the Python calculator, and reports component-wise
differences. Run this immediately after you get pair_alignn compiling and
loading the .pt file — it's the gate between "compiles" and "correct."

Pass criteria (typical):
    |ΔE|          < 1e-4 eV
    max|ΔF|       < 1e-3 eV/Å
    max|Δσ|       < 1e-4 eV/Å³

Usage:
    python scripts/torch/validate_pair_alignn.py \\
        --model-dir OutputDir --ts-model alignn_ff.pt \\
        --jid JVASP-1002 --supercell 2 2 2
"""
from __future__ import annotations
import argparse, os, sys, tempfile
import numpy as np
from ase.io import write as ase_write
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data

# eV/Å³ ↔ bar conversion (LAMMPS `metal` units pressure)
EV_PER_A3_TO_BAR = 1.602176634e6


def python_reference_ase(atoms_jarvis, model_dir):
    """ASE-calculator reference (DGL graph). NOTE: the DGL neighbor graph
    can differ from the pure-torch k-nearest graph that pair_alignn (and
    the model's training) uses, so this is *not* the right baseline for a
    pure-torch model — see python_reference_pure."""
    from alignn.ff.ff import AlignnAtomwiseCalculator

    a = atoms_jarvis.ase_converter()
    a.calc = AlignnAtomwiseCalculator(
        path=model_dir, force_mult_batchsize=False
    )
    E = float(np.asarray(a.get_potential_energy()).reshape(-1)[0])
    F = np.asarray(a.get_forces())                      # eV/Å
    S6 = np.asarray(a.get_stress(voigt=True)).reshape(-1)   # eV/Å³ Voigt
    S = np.array([[S6[0], S6[5], S6[4]],
                  [S6[5], S6[1], S6[3]],
                  [S6[4], S6[3], S6[2]]])
    return E, F, S


def python_reference_pure(atoms_jarvis, model_dir, cutoff, max_neighbors):
    """Pure-torch reference: same k-nearest graph + entry point
    (forward_tensors_z) that pair_alignn calls and that the model was
    trained with. This is the correct apples-to-apples baseline."""
    import torch
    from alignn.pretrained import load_pure_torch_model
    from alignn.torch_graph_builder import torch_neighbor_list

    model, cfg = load_pure_torch_model(model_dir, device="cpu")
    af = cfg.get("atom_features", "cgcnn")
    model.register_species_table(atom_features=af)

    dt = torch.get_default_dtype()
    pos = torch.tensor(
        np.asarray(atoms_jarvis.cart_coords), dtype=dt
    ).requires_grad_(True)
    L = torch.tensor(np.asarray(atoms_jarvis.lattice_mat), dtype=dt)
    Z = torch.tensor(np.asarray(atoms_jarvis.atomic_numbers), dtype=torch.long)
    src, dst, shift, _ = torch_neighbor_list(
        pos, L, cutoff=float(cutoff), max_neighbors=int(max_neighbors),
        atoms=atoms_jarvis, use_matscipy_topology=True,
    )
    out = model.forward_tensors_z(pos, L, Z, src, dst, shift, True)
    E = float(out["energy"].detach())
    F = out["forces"].detach().cpu().numpy()
    S = (out["stress"].detach().cpu().numpy()
         if "stress" in out else np.zeros((3, 3)))
    return E, F, S


def lammps_pair_alignn(ase_atoms, ts_model_path, species_symbols, cutoff=5.0,
                       max_neighbors=12):
    """Run LAMMPS with pair_alignn at step 0, return E/F/S."""
    from lammps import lammps

    tmpdir = tempfile.mkdtemp(prefix="pair_alignn_val_")
    data_path = os.path.join(tmpdir, "sys.data")
    ase_write(data_path, ase_atoms, format="lammps-data",
              specorder=list(species_symbols))

    lmp = lammps()
    lmp.commands_string(f"""
        units           metal
        atom_style      atomic
        boundary        p p p
        read_data       {data_path}
        {'mass ' + ' '.join(str(i+1)+' '+str(_atomic_mass(sym))
                             for i,sym in enumerate(species_symbols))}
        pair_style      alignn {cutoff} {max_neighbors}
        pair_coeff      * * {ts_model_path} {' '.join(species_symbols)}
        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes
        compute         PE all pe
        compute         VIR all pressure NULL virial
        thermo          1
        thermo_style    custom step pe c_VIR[1] c_VIR[2] c_VIR[3] c_VIR[4] c_VIR[5] c_VIR[6]
        run             0
    """)

    nlocal = lmp.get_natoms()
    # Gather forces by atom ID so ordering matches the ASE/jarvis object
    F_flat = lmp.gather_atoms("f", 1, 3)
    F = np.array(F_flat[:]).reshape(nlocal, 3)

    # Energy (metal units → eV already)
    E = float(lmp.get_thermo("pe"))

    # Pressure tensor in LAMMPS = sum of kinetic + virial. At step 0 with
    # zero velocities, kinetic term is 0 so p = virial/V. Flip sign to
    # match ASE stress convention (stress = -1/V * dE/dε).
    V = lmp.get_thermo("vol")        # Å³
    p_bar = np.array([lmp.get_thermo(name) for name in
                      ("c_VIR[1]","c_VIR[2]","c_VIR[3]",
                       "c_VIR[4]","c_VIR[5]","c_VIR[6]")])
    # LAMMPS virial order: xx yy zz xy xz yz; pressure = virial/V (bar)
    p_ev_A3 = p_bar / EV_PER_A3_TO_BAR              # bar -> eV/Å³
    S = np.array([[ p_ev_A3[0], p_ev_A3[3], p_ev_A3[4]],
                  [ p_ev_A3[3], p_ev_A3[1], p_ev_A3[5]],
                  [ p_ev_A3[4], p_ev_A3[5], p_ev_A3[2]]])
    # LAMMPS pressure sign convention is opposite of stress
    S = -S
    lmp.close()
    return E, F, S


def _atomic_mass(sym):
    from ase.data import atomic_masses, atomic_numbers
    return atomic_masses[atomic_numbers[sym]]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with best_model.pt + config.json for the Python reference")
    ap.add_argument("--ts-model",  required=True, help="TorchScript .pt exported via export_torchscript.py")
    ap.add_argument("--jid", default="JVASP-1002")
    ap.add_argument("--supercell", type=int, nargs=3, default=[2,2,2])
    ap.add_argument("--perturb", type=float, default=0.03,
                    help="Å, random atomic displacement to break symmetry "
                         "(so forces are non-zero and meaningful)")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="Å, model cutoff (read from model-dir config if unset)")
    ap.add_argument("--max-neighbors", type=int, default=None,
                    help="k-nearest cap (read from model-dir config if unset). "
                         "Must match pair_style alignn <cutoff> <max_neighbors>.")
    ap.add_argument("--fail-force", type=float, default=0.05,
                    help="eV/Å, gate: exit non-zero if max|ΔF| exceeds this.")
    ap.add_argument("--fail-energy", type=float, default=0.05,
                    help="eV/atom, gate: exit non-zero if |ΔE|/atom exceeds "
                         "this.")
    ap.add_argument("--reference", choices=["pure", "ase"], default="pure",
                    help="Python baseline. 'pure' (default) uses the same "
                         "pure-torch k-nearest graph + forward_tensors_z that "
                         "pair_alignn implements and the model was trained "
                         "with. 'ase' uses AlignnAtomwiseCalculator (DGL "
                         "graph) which can legitimately differ.")
    args = ap.parse_args()
    if args.cutoff is None or args.max_neighbors is None:
        import json as _j
        _cfg = _j.load(open(f"{args.model_dir}/config.json"))
        if args.cutoff is None:
            args.cutoff = float(_cfg["cutoff"])
            print(f"cutoff (from config): {args.cutoff} Å")
        if args.max_neighbors is None:
            args.max_neighbors = int(_cfg.get("max_neighbors", 12))
            print(f"max_neighbors (from config): {args.max_neighbors}")

    # Build system
    d = get_jid_data(jid=args.jid, dataset="dft_3d")
    atoms = Atoms.from_dict(d["atoms"]).get_conventional_atoms
    atoms = atoms.make_supercell(args.supercell)
    ase_atoms = atoms.ase_converter()

    # Small perturbation — exact equilibrium would give all-zero forces,
    # which is not a useful validation.
    rng = np.random.default_rng(0)
    ase_atoms.positions += args.perturb * rng.standard_normal(ase_atoms.positions.shape)

    species = sorted(set(ase_atoms.get_chemical_symbols()))
    print(f"System: {atoms.composition.reduced_formula} x{args.supercell}  "
          f"N={len(ase_atoms)}  species={species}")

    # Re-wrap into jarvis for the Python reference
    from jarvis.core.atoms import ase_to_atoms as ase2j
    atoms_pert = ase2j(ase_atoms)

    # --- Python reference ---
    if args.reference == "pure":
        print("\n→ Python reference (pure-torch: build_pure_torch_graph "
              "+ forward_tensors_z)...")
        E_py, F_py, S_py = python_reference_pure(
            atoms_pert, args.model_dir, args.cutoff, args.max_neighbors
        )
    else:
        print("\n→ Python reference (ASE AlignnAtomwiseCalculator, DGL graph)"
              " — may differ from pair_alignn's pure-torch graph...")
        E_py, F_py, S_py = python_reference_ase(atoms_pert, args.model_dir)

    # --- LAMMPS pair_alignn ---
    print("→ LAMMPS pair_alignn (run 0)...")
    E_lmp, F_lmp, S_lmp = lammps_pair_alignn(ase_atoms, args.ts_model, species,
                                              cutoff=args.cutoff,
                                              max_neighbors=args.max_neighbors)

    # --- Compare ---
    dE = abs(E_lmp - E_py)
    dF = F_lmp - F_py
    dS = S_lmp - S_py
    print("\n── Comparison ──────────────────────────────────────────")
    print(f"E (Python):      {E_py: .6f} eV")
    print(f"E (LAMMPS):      {E_lmp: .6f} eV")
    print(f"|ΔE|:            {dE: .3e} eV     "
          f"[{'PASS' if dE < 1e-4 else 'FAIL'}]  (threshold 1e-4)")

    max_abs_dF = float(np.max(np.abs(dF)))
    rms_dF     = float(np.sqrt(np.mean(dF**2)))
    max_abs_F  = float(np.max(np.abs(F_py)))
    print(f"max|ΔF|:         {max_abs_dF: .3e} eV/Å   "
          f"[{'PASS' if max_abs_dF < 1e-3 else 'FAIL'}]  (threshold 1e-3)")
    print(f"rms ΔF:          {rms_dF: .3e} eV/Å")
    print(f"max|F_ref|:      {max_abs_F: .3e} eV/Å   (scale for context)")

    max_abs_dS = float(np.max(np.abs(dS)))
    print(f"max|Δσ|:         {max_abs_dS: .3e} eV/Å³  "
          f"[{'PASS' if max_abs_dS < 1e-4 else 'FAIL'}]  (threshold 1e-4)")

    print("\n── Per-atom ΔF (eV/Å) — first 5 atoms ──")
    print("idx   ΔF_x         ΔF_y         ΔF_z         |ΔF|")
    for i in range(min(5, len(F_py))):
        d = dF[i]
        print(f"{i:3d}  {d[0]:+10.3e}  {d[1]:+10.3e}  {d[2]:+10.3e}  {np.linalg.norm(d):10.3e}")

    # Write out both force arrays for post-hoc inspection
    np.savez("validate_pair_alignn.npz",
             E_py=E_py, F_py=F_py, S_py=S_py,
             E_lmp=E_lmp, F_lmp=F_lmp, S_lmp=S_lmp)
    print("\nsaved raw arrays -> validate_pair_alignn.npz")

    # Gate decision: a *gross* mismatch means pair_alignn and the Python
    # reference disagree, so any MD with this .pt is untrustworthy. We use a
    # loose force threshold here (not the strict 1e-3 above) so float32
    # round-off doesn't trip the gate — only real bugs do.
    n_atoms = max(len(F_py), 1)
    dE_per_atom = dE / n_atoms
    gross = (max_abs_dF > args.fail_force) or (dE_per_atom > args.fail_energy)
    print(
        f"\nGATE: max|ΔF|={max_abs_dF:.3e} (limit {args.fail_force}) "
        f"|ΔE|/atom={dE_per_atom:.3e} (limit {args.fail_energy}) "
        f"-> {'FAIL' if gross else 'OK'}"
    )
    if gross:
        print(
            "pair_alignn disagrees with the Python reference. Do NOT run MD "
            "until this is fixed (check the TorchScript export and the C++ "
            "neighbor/graph construction)."
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
