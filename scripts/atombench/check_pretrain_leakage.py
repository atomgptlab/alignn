#!/usr/bin/env python3
"""Measure structural overlap between a pretraining corpus and the test set.

Holding out material ids is not enough: the same crystal can appear in a
database under more than one id, so a pretraining corpus drawn from the same
database may still contain structures that a matcher would call identical to a
benchmark target.  Any model pretrained on such a corpus can reach those
targets by recall rather than by generation, and a match rate that includes
them is not comparable to a model trained only on the benchmark split.

This reports the fraction of test targets that have a StructureMatcher-equal
counterpart in the pretraining corpus, using the *same* matcher settings the
benchmark scores with.  Candidates are pre-filtered by reduced formula, which
is exact (a matcher can only match same-composition structures) and turns the
sweep from hours into seconds.

Run in an environment with pymatgen.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from tqdm import tqdm


def to_structure(rec) -> Structure:
    return Structure(
        Lattice(rec["lattice_mat"]),
        [int(z) for z in rec["atomic_numbers"]],
        rec["frac_coords"],
    )


def reduced(struct: Structure) -> Structure:
    return struct.get_primitive_structure().get_reduced_structure(
        reduction_algo="niggli"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-dir", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    corpus = json.loads((Path(args.pretrain_dir) / "train.json").read_text())
    test = json.loads(Path(args.test_json).read_text())

    by_formula = defaultdict(list)
    for rec in corpus:
        by_formula[rec["formula"]].append(rec)
    print(f"corpus={len(corpus)} formulas={len(by_formula)} test={len(test)}")

    # Same settings compute_metrics.py uses for the benchmark match rate.
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    hits = []
    for rec in tqdm(test, desc="checking"):
        try:
            target = reduced(to_structure(rec))
        except Exception:
            continue
        found = None
        for cand in by_formula.get(rec["formula"], []):
            try:
                if matcher.fit(target, reduced(to_structure(cand))):
                    found = cand["material_id"]
                    break
            except Exception:
                continue
        if found:
            hits.append(
                {
                    "test_id": rec["material_id"],
                    "formula": rec["formula"],
                    "corpus_id": found,
                }
            )

    frac = len(hits) / max(len(test), 1)
    print(
        f"\n{len(hits)}/{len(test)} test targets ({frac:.1%}) have a "
        f"structure-matched counterpart in the pretraining corpus"
    )
    for h in hits[:20]:
        print(f"  {h['test_id']:>16} {h['formula']:>12}  <- {h['corpus_id']}")
    if len(hits) > 20:
        print(f"  ... and {len(hits) - 20} more")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "n_test": len(test),
                    "n_leaked": len(hits),
                    "fraction": frac,
                    "matcher": {"stol": 0.5, "angle_tol": 10, "ltol": 0.3},
                    "hits": hits,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
