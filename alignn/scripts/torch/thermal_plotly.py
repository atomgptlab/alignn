"""Phonon + thermal properties from trained ALIGNN-FF, plotted with Plotly.

Computes:
  * phonon bandstructure
  * phonon DOS
  * harmonic thermodynamics: Cv(T), S(T), F(T), ZPE
  * Dulong-Petit comparison

Outputs a single HTML with 4 panels.

Usage:
    python thermal_plotly.py --model-dir OutputDir --jid JVASP-1002
"""
from __future__ import annotations
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ase import Atoms as AseAtoms
from ase.filters import ExpCellFilter
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.db.figshare import get_jid_data

from alignn.ff.ff import AlignnAtomwiseCalculator


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
def relax(atoms, calc, fmax=0.05, steps=100):
    a = atoms.ase_converter()
    a.calc = calc
    FIRE(ExpCellFilter(a), logfile=None).run(fmax=fmax, steps=steps)
    return ase_to_atoms(a)


def run_phonopy(atoms, calc, dim=(2, 2, 2), distance=0.03):
    bulk = PhonopyAtoms(
        symbols=atoms.elements,
        scaled_positions=atoms.frac_coords,
        cell=atoms.lattice_mat,
    )
    ph = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
    ph.generate_displacements(distance=distance)

    forces_set = []
    for sc in ph.supercells_with_displacements:
        a = AseAtoms(
            symbols=list(sc.symbols),
            scaled_positions=np.array(sc.scaled_positions),
            cell=np.array(sc.cell),
            pbc=True,
        )
        a.calc = calc
        F = np.array(a.get_forces())
        F -= F.mean(axis=0)
        forces_set.append(F)
    ph.produce_force_constants(forces=forces_set)
    return ph


def bandstructure(ph, atoms, line_density=20):
    kp = Kpoints().kpath(atoms, line_density=line_density)
    q_points = np.array(kp.kpts)
    labels_raw = kp.labels

    freqs, distances, tick_pos, tick_labels = [], [], [], []
    last_kstr = None
    d = 0.0
    for i, (q, lbl) in enumerate(zip(q_points, labels_raw)):
        if i > 0:
            d += np.linalg.norm(q_points[i] - q_points[i - 1])
        distances.append(d)
        freqs.append(ph.get_frequencies(q))
        kstr = ",".join(f"{x:.6f}" for x in q)
        if lbl and kstr != last_kstr:
            tick_pos.append(d)
            tick_labels.append(lbl)
            last_kstr = kstr
    return np.array(distances), np.array(freqs), tick_pos, tick_labels


def thermal_properties(ph, mesh=(20, 20, 20), t_min=0, t_max=1000, t_step=10):
    ph.run_mesh(list(mesh))
    ph.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    td = ph.get_thermal_properties_dict()
    N = len(ph.primitive)
    return {
        "T":   td["temperatures"],
        "Cv":  td["heat_capacity"] / N,   # J/K/mol-atom
        "S":   td["entropy"] / N,         # J/K/mol-atom
        "F":   td["free_energy"] / N,     # kJ/mol-atom
        "ZPE_per_cell_kJmol": float(td["free_energy"][0]),
        "ZPE_per_atom_eV": float(td["free_energy"][0]) * 1000.0 / 96.485 / N,
    }


def dos(ph):
    ph.run_total_dos()
    d = ph.get_total_dos_dict()
    return np.array(d["frequency_points"]), np.array(d["total_dos"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(bands, dos_data, thermo, title=""):
    distances, freqs, tick_pos, tick_labels = bands
    dos_freq, dos_val = dos_data

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"colspan": 2}, None, {}],
            [{}, {}, {}],
        ],
        subplot_titles=(
            "Phonon bandstructure", "Phonon DOS",
            "Heat capacity Cv", "Entropy S", "Free energy F",
        ),
        horizontal_spacing=0.08, vertical_spacing=0.15,
    )

    # Bandstructure (top-left, 2 cols wide)
    for b in range(freqs.shape[1]):
        fig.add_trace(
            go.Scatter(x=distances, y=freqs[:, b], mode="lines",
                       line=dict(color="steelblue", width=1.2),
                       showlegend=False,
                       hovertemplate="d=%{x:.2f}<br>ν=%{y:.2f} THz"),
            row=1, col=1,
        )
    for xp in tick_pos[1:-1]:
        fig.add_vline(x=xp, line=dict(color="lightgray", width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="black", dash="dot", width=1), row=1, col=1)

    # DOS (top-right)
    fig.add_trace(
        go.Scatter(x=dos_val, y=dos_freq, mode="lines",
                   fill="tozerox", line=dict(color="firebrick"),
                   showlegend=False,
                   hovertemplate="DOS=%{x:.2f}<br>ν=%{y:.2f} THz"),
        row=1, col=3,
    )

    # Cv, S, F (bottom row)
    T = thermo["T"]
    fig.add_trace(go.Scatter(x=T, y=thermo["Cv"], mode="lines",
                             line=dict(color="seagreen"), showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=24.94, line=dict(dash="dot", color="gray"),
                  annotation_text="3R (Dulong-Petit)",
                  annotation_position="top right", row=2, col=1)
    fig.add_trace(go.Scatter(x=T, y=thermo["S"], mode="lines",
                             line=dict(color="darkorange"), showlegend=False),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=T, y=thermo["F"], mode="lines",
                             line=dict(color="mediumpurple"), showlegend=False),
                  row=2, col=3)

    # Axes labels
    ticktext = [l.replace("\\Gamma", "Γ") for l in tick_labels]
    fig.update_xaxes(title_text="Wave vector",
                     tickmode="array", tickvals=tick_pos, ticktext=ticktext,
                     showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Frequency (THz)", row=1, col=1)
    fig.update_xaxes(title_text="DOS (states/THz/cell)", row=1, col=3)
    fig.update_yaxes(title_text="Frequency (THz)", row=1, col=3)

    for col, ylab in [(1, "Cv (J/K/mol-atom)"),
                      (2, "S (J/K/mol-atom)"),
                      (3, "F (kJ/mol-atom)")]:
        fig.update_xaxes(title_text="T (K)", row=2, col=col)
        fig.update_yaxes(title_text=ylab, row=2, col=col)

    fig.update_layout(template="plotly_white", title=title,
                     width=1250, height=800)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="OutputDir")
    ap.add_argument("--jid", default="JVASP-1002")
    ap.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--mesh", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--t-max", type=float, default=1000.0)
    ap.add_argument("--out", default="thermal.html")
    args = ap.parse_args()

    calc = AlignnAtomwiseCalculator(path=args.model_dir, force_mult_batchsize=False)

    d = get_jid_data(jid=args.jid, dataset="dft_3d")
    atoms = Atoms.from_dict(d["atoms"]).get_conventional_atoms
    formula = atoms.composition.reduced_formula
    print(f"loaded {args.jid}  {formula}  N={atoms.num_atoms}")

    relaxed = relax(atoms, calc)
    prim = relaxed.get_primitive_atoms
    print(f"relaxed → primitive N={prim.num_atoms}")

    ph = run_phonopy(prim, calc, dim=tuple(args.dim))
    bands = bandstructure(ph, prim)
    dos_data = dos(ph)
    thermo = thermal_properties(ph, mesh=tuple(args.mesh), t_max=args.t_max)

    # Summary
    n_imag = int((bands[1] < -0.05).sum())
    T = thermo["T"]
    i300 = int(np.argmin(abs(T - 300)))
    print(f"\n── Results ────────────────────────────────────────────")
    print(f"bands           : {bands[1].shape[1]}   imag modes: {n_imag}")
    print(f"ZPE             : {thermo['ZPE_per_cell_kJmol']:.3f} kJ/mol-cell"
          f"   ({thermo['ZPE_per_atom_eV']:.4f} eV/atom)")
    print(f"Cv(300 K)       : {thermo['Cv'][i300]:.3f} J/K/mol-atom  "
          f"(Dulong-Petit: 24.94)")
    print(f"S(300 K)        : {thermo['S'][i300]:.3f} J/K/mol-atom")
    print(f"F(300 K) - F(0) : {thermo['F'][i300] - thermo['F'][0]:.3f} kJ/mol-atom")

    fig = plot_all(bands, dos_data, thermo,
                  title=f"{formula} ({args.jid}) — ALIGNN-FF phonons "
                        f"[supercell {args.dim}, mesh {args.mesh}]")
    fig.write_html(args.out)
    print(f"\nwrote {args.out}")

    # In a Jupyter cell, uncomment:
    # fig.show()


if __name__ == "__main__":
    main()
