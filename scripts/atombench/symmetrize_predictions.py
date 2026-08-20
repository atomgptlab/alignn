#!/usr/bin/env python3
"""Snap predicted cells onto their detected symmetry.

Lattice-angle MAE is measured after Niggli reduction, which is a
*discontinuous* function of the cell: a generated cell that is very nearly
cubic but off by half a degree can reduce to a completely different basis and
contribute a huge angle error.  Idealising the structure first — detect the
space group at a given tolerance, then rebuild the cell so it satisfies that
symmetry exactly — makes right angles exactly right and removes that
sensitivity.

The transformation is target-blind: one tolerance is applied uniformly to
every prediction.  Choose that tolerance on the *validation* split
(``sweep`` mode) and then apply it to test, so nothing is tuned on the
numbers being reported.

Run in an environment with pymatgen.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

csv.field_size_limit(10**8)


def unescape(text: str) -> str:
    return text.replace("\\n", "\n")


def escape(text: str) -> str:
    return text.replace("\n", "\\n")


def symmetrize_poscar(poscar_text: str, symprec: float) -> str:
    """Return the idealised POSCAR, or the original if analysis fails."""
    try:
        s = Structure.from_str(unescape(poscar_text), fmt="poscar")
        sga = SpacegroupAnalyzer(s, symprec=symprec)
        refined = sga.get_refined_structure()
        # Same reduction the metric applies, so what we hand over is already
        # in the form it will be compared in.
        prim = refined.get_primitive_structure()
        if len(prim) == 0:
            return poscar_text
        return escape(prim.to(fmt="poscar"))
    except Exception:
        return poscar_text


def transform(in_csv: Path, out_csv: Path, symprec: float) -> int:
    rows, changed = [], 0
    with in_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            new_pred = symmetrize_poscar(row["prediction"], symprec)
            changed += int(new_pred != row["prediction"])
            rows.append(
                {
                    "id": row["id"],
                    "target": row["target"],
                    "prediction": new_pred,
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "target", "prediction"])
        w.writeheader()
        w.writerows(rows)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--out",
        required=True,
        help="output CSV, or output *directory* in sweep mode",
    )
    ap.add_argument("--symprec", type=float, default=0.1)
    ap.add_argument(
        "--sweep",
        default=None,
        help="comma-separated tolerances; writes one CSV each "
        "(use on the validation split to pick a value)",
    )
    args = ap.parse_args()

    in_csv = Path(args.csv)
    if args.sweep:
        out_dir = Path(args.out)
        for sp in [float(x) for x in args.sweep.split(",")]:
            tag = f"symprec{sp:g}".replace(".", "p")
            dest = out_dir / tag / f"{in_csv.stem}_{tag}.csv"
            n = transform(in_csv, dest, sp)
            print(f"symprec={sp:g}: changed {n} rows -> {dest}", flush=True)
    else:
        n = transform(in_csv, Path(args.out), args.symprec)
        print(f"symprec={args.symprec:g}: changed {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
