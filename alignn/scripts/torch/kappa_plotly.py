"""Lattice thermal conductivity κ(T) and 3-phonon properties from ALIGNN-FF.

Uses phono3py for anharmonic (3rd-order) force constants, then solves the
linearized Boltzmann transport equation under the relaxation-time
approximation (RTA).

WARNING: this is expensive. For Si primitive + 2x2x2 supercell (16 atoms),
phono3py generates ~111 displacements for fc3 + ~1 for fc2 ~~ hundreds
of force evaluations. Allow 10-30 min on a single GPU.

Usage:
    python kappa_plotly.py --model-dir OutputDir --jid JVASP-1002 \\
        --dim 2 2 2 --mesh 11 11 11
"""
from __future__ import annotations
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ase import Atoms as AseAtoms
from ase.filters import ExpCellFilter
from ase.optimize import FIRE
from phono3py import Phono3py
from phonopy.structure.atoms import PhonopyAtoms

from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.db.figshare import get_jid_data

from alignn.ff.ff import AlignnAtomwiseCalculator


def relax(atoms, calc, fmax=0.05, steps=100):
    a = atoms.ase_converter()
    a.calc = calc
    FIRE(ExpCellFilter(a), logfile=None).run(fmax=fmax, steps=steps)
    return ase_to_atoms(a)


def forces_on_supercell(sc, calc):
    a = AseAtoms(
        symbols=list(sc.symbols),
        scaled_positions=np.array(sc.scaled_positions),
        cell=np.array(sc.cell),
        pbc=True,
    )
    a.calc = calc
    F = np.array(a.get_forces())
    F -= F.mean(axis=0)
    return F


def run_phono3py(atoms, calc, dim=(2, 2, 2), dim_fc2=None, distance=0.03):
    """Build Phono3py, compute fc3 (and fc2) via ALIGNN-FF."""
    bulk = PhonopyAtoms(
        symbols=atoms.elements,
        scaled_positions=atoms.frac_coords,
        cell=atoms.lattice_mat,
    )
    # fc2 supercell can be larger than fc3 supercell for a better harmonic part
    if dim_fc2 is None:
        dim_fc2 = dim
    ph3 = Phono3py(
        bulk,
        supercell_matrix=[[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]],
        phonon_supercell_matrix=[[dim_fc2[0], 0, 0], [0, dim_fc2[1], 0], [0, 0, dim_fc2[2]]],
    )
    ph3.generate_displacements(distance=distance)

    # fc3 supercells
    sc3 = ph3.supercells_with_displacements
    n3 = len(sc3)
    print(f"fc3 displacements: {n3}")
    fset3 = []
    for i, sc in enumerate(sc3):
        if sc is None:                                  # symmetry-reduced skip
            fset3.append(None); continue
        F = forces_on_supercell(sc, calc)
        fset3.append(F)
        if (i + 1) % 10 == 0 or i == n3 - 1:
            print(f"  fc3 {i+1}/{n3}")
    ph3.forces = fset3

    # fc2 supercells (phonon supercells)
    sc2 = ph3.phonon_supercells_with_displacements
    n2 = len(sc2)
    print(f"fc2 displacements: {n2}")
    fset2 = []
    for i, sc in enumerate(sc2):
        if sc is None:
            fset2.append(None); continue
        F = forces_on_supercell(sc, calc)
        fset2.append(F)
    ph3.phonon_forces = fset2

    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.produce_fc3(symmetrize_fc3r=True)
    return ph3


def compute_kappa(ph3, mesh=(11, 11, 11), t_min=100, t_max=1000, t_step=50,
                  boundary_mfp=None):
    """Run RTA thermal conductivity. Returns (T, kappa_xx_yy_zz_xy_xz_yz)."""
    ph3.mesh_numbers = list(mesh)
    ph3.init_phph_interaction()
    temperatures = np.arange(t_min, t_max + 1, t_step)
    ph3.run_thermal_conductivity(
        temperatures=temperatures,
        is_isotope=False,
        boundary_mfp=boundary_mfp,      # µm; None = infinite sample
        write_kappa=False,
    )
    tc = ph3.thermal_conductivity
    T = np.asarray(tc.temperatures)
    kappa = np.asarray(tc.kappa)   # shape (1, nT, 6): xx yy zz yz xz xy
    return T, kappa[0]


def plot_kappa(T, kappa, thermo=None, title=""):
    # trace of kappa = average of diagonal components
    kappa_avg = kappa[:, :3].mean(axis=1)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Thermal conductivity κ(T)",
                        "Log-log view"),
        horizontal_spacing=0.12,
    )
    for i, lbl in enumerate(["κ_xx", "κ_yy", "κ_zz"]):
        fig.add_trace(go.Scatter(x=T, y=kappa[:, i], mode="lines+markers",
                                 name=lbl, line=dict(width=1.5)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=T, y=kappa_avg, mode="lines",
                             name="κ_avg", line=dict(color="black", dash="dash")),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=T, y=kappa_avg, mode="lines+markers",
                             name="κ_avg (log-log)",
                             line=dict(color="steelblue", width=1.5),
                             showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(type="log", title_text="T (K)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="κ (W/m·K)", row=1, col=2)

    fig.update_xaxes(title_text="T (K)", row=1, col=1)
    fig.update_yaxes(title_text="κ (W/m·K)", row=1, col=1)
    fig.update_layout(template="plotly_white", title=title,
                     width=1100, height=450)
    return fig


def print_summary(T, kappa, name=""):
    kappa_avg = kappa[:, :3].mean(axis=1)
    i300 = int(np.argmin(abs(T - 300)))
    print(f"\n── κ summary {name} ──")
    print(f"κ_xx, κ_yy, κ_zz (300 K) = {kappa[i300,0]:.2f}, "
          f"{kappa[i300,1]:.2f}, {kappa[i300,2]:.2f} W/m·K")
    print(f"κ_avg(300 K) = {kappa_avg[i300]:.2f} W/m·K")
    # T-scaling: fit log(κ) = a - p*log(T) in high-T regime
    hi = T > 300
    if hi.sum() > 2:
        slope = np.polyfit(np.log(T[hi]), np.log(kappa_avg[hi]), 1)[0]
        print(f"high-T exponent p (κ ~ T^-p): p ≈ {-slope:.2f}  "
              f"(3-phonon expects ~1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="OutputDir")
    ap.add_argument("--jid", default="JVASP-1002")
    ap.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2],
                    help="fc3 supercell")
    ap.add_argument("--dim-fc2", type=int, nargs=3, default=None,
                    help="fc2 supercell (default = fc3 supercell)")
    ap.add_argument("--mesh", type=int, nargs=3, default=[11, 11, 11])
    ap.add_argument("--t-max", type=float, default=1000.0)
    ap.add_argument("--out", default="kappa.html")
    args = ap.parse_args()

    calc = AlignnAtomwiseCalculator(path=args.model_dir, force_mult_batchsize=False)
    d = get_jid_data(jid=args.jid, dataset="dft_3d")
    atoms = Atoms.from_dict(d["atoms"]).get_conventional_atoms
    formula = atoms.composition.reduced_formula
    print(f"loaded {args.jid}  {formula}  N={atoms.num_atoms}")

    relaxed = relax(atoms, calc)
    prim = relaxed.get_primitive_atoms
    print(f"relaxed → primitive N={prim.num_atoms}")

    ph3 = run_phono3py(
        prim, calc,
        dim=tuple(args.dim),
        dim_fc2=tuple(args.dim_fc2) if args.dim_fc2 else None,
    )
    T, kappa = compute_kappa(ph3, mesh=tuple(args.mesh), t_max=args.t_max)
    print_summary(T, kappa, formula)

    fig = plot_kappa(T, kappa,
                    title=f"{formula} ({args.jid}) — κ(T) via phono3py+ALIGNN-FF "
                          f"[fc3 {args.dim}, mesh {args.mesh}]")
    fig.write_html(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
