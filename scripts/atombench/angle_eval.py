#!/usr/bin/env python3
"""Mechanism metrics for a generated benchmark CSV.

Complements ``score.sh``, which reports the AtomBench benchmark numbers.  This
adds the two diagnostics that say whether an *angular* channel is doing
anything:

    bond-angle distribution   generated vs. held-out real structures, the
                              comparison FoldingDiff uses
    relaxation displacement   how far a sample has to move to reach the
                              nearest ALIGNN-FF local minimum, the proximity
                              MatterGen evaluates

Both sides of the angle comparison come from the same file: the CSV written by
``generate_benchmark.py`` carries the generated structure in ``prediction`` and
the held-out reference in ``target``.

    python scripts/atombench/angle_eval.py runs/bench/alignn_csp.csv \
        --relax --limit 50

Writes ``angle_metrics.json`` next to the CSV.  Relaxation is off by default
because it costs about a second per structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignn.inverse.evaluate import (
    DEFAULT_ANGLE_CUTOFF,
    DEFAULT_BINS,
    DEFAULT_MAX_NEIGHBORS,
    collect_bond_angles,
    compare_angle_distributions,
    relaxation_displacement,
    structures_from_benchmark_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="benchmark CSV from generate_benchmark.py")
    ap.add_argument(
        "--reference-csv",
        default=None,
        help="take the reference structures from a different CSV's "
        "'target' column (default: the same file)",
    )
    ap.add_argument("--cutoff", type=float, default=DEFAULT_ANGLE_CUTOFF)
    ap.add_argument("--max-neighbors", type=int, default=DEFAULT_MAX_NEIGHBORS)
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument(
        "--relax",
        action="store_true",
        help="also relax each generated structure with ALIGNN-FF and report "
        "how far it moved",
    )
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--relax-fmax", type=float, default=0.05)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only use the first N rows (0 = all); relaxation is the slow "
        "part, so this mostly matters with --relax",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    generated = structures_from_benchmark_csv(csv_path, "prediction")
    reference = structures_from_benchmark_csv(
        args.reference_csv or csv_path, "target"
    )
    if args.limit:
        generated = generated[: args.limit]
        reference = reference[: args.limit]
    print(f"{len(generated)} generated, {len(reference)} reference structures")

    kw = {"cutoff": args.cutoff, "max_neighbors": args.max_neighbors}
    gen_angles = collect_bond_angles(generated, **kw)
    ref_angles = collect_bond_angles(reference, **kw)
    metrics = {
        "angle_distribution": compare_angle_distributions(
            gen_angles, ref_angles, bins=args.bins
        ),
        "angle_cutoff": args.cutoff,
        "angle_max_neighbors": args.max_neighbors,
        "generated_angle_mean_deg": (
            float(np.mean(gen_angles)) if gen_angles.size else None
        ),
        "reference_angle_mean_deg": (
            float(np.mean(ref_angles)) if ref_angles.size else None
        ),
    }
    d = metrics["angle_distribution"]
    print(
        f"  bond angles: KL {d['kl']:.4f}  JS {d['js']:.4f}  "
        f"Wasserstein {d['wasserstein_deg']:.3f} deg  "
        f"({d['n_generated']} vs {d['n_reference']} angles)"
    )

    if args.relax:
        from alignn.inverse.relax_rank import AlignnFFRelaxer

        relaxer = AlignnFFRelaxer(
            relax_cell=True, fmax=args.relax_fmax, steps=args.relax_steps
        )
        rows = []
        for i, atoms in enumerate(generated):
            e0 = None
            try:
                e0 = relaxer.energy(atoms)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] single point failed: {exc}")
            res = relaxer.relax(atoms)
            if res.error:
                print(f"  [{i}] relaxation failed: {res.error}")
                continue
            rows.append(
                relaxation_displacement(
                    atoms, res.atoms, e0, res.energy_per_atom
                )
            )
        if rows:
            keys = sorted({k for r in rows for k in r})
            summary = {
                k: float(np.nanmean([r[k] for r in rows if k in r]))
                for k in keys
            }
            summary["n_relaxed"] = len(rows)
            metrics["relaxation"] = summary
            print(
                f"  relaxation: RMSD {summary['rmsd_angstrom']:.4f} A  "
                f"|dV|/V {abs(summary['volume_change_frac']):.4f}  "
                f"over {len(rows)} structures"
            )

    out = Path(args.output or csv_path.with_name("angle_metrics.json"))
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
