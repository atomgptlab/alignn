"""Phonon bandstructure from a trained ALIGNN-FF model, plotted with Plotly.

Usage:
    python phonons_plotly.py --model-dir OutputDir --jid JVASP-1002
"""
from __future__ import annotations
import argparse
import numpy as np
import plotly.graph_objects as go
from ase import Atoms as AseAtoms
from ase.filters import ExpCellFilter
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from jarvis.core.atoms import Atoms, ase_to_atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.db.figshare import get_jid_data

from alignn.ff.ff import AlignnAtomwiseCalculator


def relax(atoms: Atoms, calc, fmax=0.05, steps=100):
    a = atoms.ase_converter()
    a.calc = calc
    FIRE(ExpCellFilter(a), logfile=None).run(fmax=fmax, steps=steps)
    return ase_to_atoms(a)


def compute_phonons(atoms: Atoms, calc, dim=(2, 2, 2), distance=0.03, line_density=20):
    bulk = PhonopyAtoms(
        symbols=atoms.elements,
        scaled_positions=atoms.frac_coords,
        cell=atoms.lattice_mat,
    )
    ph = Phonopy(bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]])
    ph.generate_displacements(distance=distance)
    # evaluate forces on each displaced supercell
    forces_set = []
    for sc in ph.get_supercells_with_displacements():
        a = AseAtoms(symbols=sc.get_chemical_symbols(),
                     scaled_positions=sc.get_scaled_positions(),
                     cell=sc.get_cell(), pbc=True)
        a.calc = calc
        F = np.array(a.get_forces())
        F -= F.mean(axis=0)       # remove drift
        forces_set.append(F)
    ph.produce_force_constants(forces=forces_set)

    # walk the standard k-path (jarvis builds it from the lattice)
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
        freqs.append(ph.get_frequencies(q))   # THz
        kstr = ",".join(f"{x:.6f}" for x in q)
        if lbl and kstr != last_kstr:
            tick_pos.append(d)
            tick_labels.append(lbl)
            last_kstr = kstr
    return np.array(distances), np.array(freqs), tick_pos, tick_labels


def plot(distances, freqs, tick_pos, tick_labels, title=""):
    fig = go.Figure()
    nbands = freqs.shape[1]
    for b in range(nbands):
        fig.add_trace(go.Scatter(
            x=distances, y=freqs[:, b], mode="lines",
            line=dict(color="steelblue", width=1.5),
            showlegend=False, hovertemplate="d=%{x:.3f}<br>ν=%{y:.3f} THz",
        ))
    fig.add_hline(y=0, line=dict(color="black", dash="dot", width=1))
    for xp in tick_pos[1:-1]:
        fig.add_vline(x=xp, line=dict(color="lightgray", width=1))
    fig.update_layout(
        title=title, width=800, height=500,
        xaxis=dict(
            title="Wave vector",
            tickmode="array",
            tickvals=tick_pos,
            ticktext=[l.replace("\\Gamma", "Γ") for l in tick_labels],
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(title="Frequency (THz)", gridcolor="lightgray"),
        template="plotly_white",
    )
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="OutputDir")
    ap.add_argument("--jid", default="JVASP-1002")
    ap.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--out", default="phonons.html")
    args = ap.parse_args()

    calc = AlignnAtomwiseCalculator(path=args.model_dir, force_mult_batchsize=False)

    d = get_jid_data(jid=args.jid, dataset="dft_3d")
    atoms = Atoms.from_dict(d["atoms"]).get_conventional_atoms
    print(f"loaded {args.jid}  {atoms.composition.reduced_formula}  N={atoms.num_atoms}")

    relaxed = relax(atoms, calc)
    prim = relaxed.get_primitive_atoms
    print(f"relaxed → primitive N={prim.num_atoms}")

    distances, freqs, tick_pos, tick_labels = compute_phonons(
        prim, calc, dim=tuple(args.dim)
    )
    n_imag = int((freqs < -0.05).sum())
    print(f"bands computed: {freqs.shape[1]}  "
          f"imag modes (<-0.05 THz): {n_imag}")

    fig = plot(distances, freqs, tick_pos, tick_labels,
               title=f"Phonon bands — {atoms.composition.reduced_formula} "
                     f"({args.jid})  supercell {args.dim}")
    fig.write_html(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
