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
import argparse, os, tempfile
import numpy as np
from ase.io import write as ase_write
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import get_jid_data

# Reference: Python ASE calculator path
from alignn.ff.ff import AlignnAtomwiseCalculator


# eV/Å³ ↔ bar conversion (LAMMPS `metal` units pressure)
EV_PER_A3_TO_BAR = 1.602176634e6


def python_reference(atoms_jarvis, calc):
    a = atoms_jarvis.ase_converter()
    a.calc = calc
    E_raw = a.get_potential_energy()
    E = float(np.asarray(E_raw).reshape(-1)[0])        # calc may return shape-(1,)
    F = np.asarray(a.get_forces())                     # eV/Å
    S6 = np.asarray(a.get_stress(voigt=True)).reshape(-1)   # eV/Å³ (Voigt)
    # Reorder ASE Voigt (xx,yy,zz,yz,xz,xy) -> plain 3x3
    S = np.array([[S6[0], S6[5], S6[4]],
                  [S6[5], S6[1], S6[3]],
                  [S6[4], S6[3], S6[2]]])
    return E, F, S


def lammps_pair_alignn(ase_atoms, ts_model_path, species_symbols, cutoff=5.0):
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
        pair_style      alignn {cutoff}
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
    args = ap.parse_args()
    if args.cutoff is None:
        import json as _j
        args.cutoff = float(_j.load(open(f"{args.model_dir}/config.json"))["cutoff"])
        print(f"cutoff (from config): {args.cutoff} Å")

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
    print("\n→ Python reference (AlignnAtomwiseCalculator)...")
    calc = AlignnAtomwiseCalculator(path=args.model_dir, force_mult_batchsize=False)
    E_py, F_py, S_py = python_reference(atoms_pert, calc)

    # --- LAMMPS pair_alignn ---
    print("→ LAMMPS pair_alignn (run 0)...")
    E_lmp, F_lmp, S_lmp = lammps_pair_alignn(ase_atoms, args.ts_model, species,
                                              cutoff=args.cutoff)

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


if __name__ == "__main__":
    main()
