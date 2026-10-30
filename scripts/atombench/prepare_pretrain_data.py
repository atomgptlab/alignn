#!/usr/bin/env python3
"""Build a large pretraining corpus of crystals from a JARVIS dataset.

The AtomBench superconductor splits hold only ~850 training crystals, which a
diffusion denoiser memorises long before it generalises.  MatterGen sidesteps
this in the original study by fine-tuning from an Alexandria-pretrained
checkpoint; CDVAE and FlowMM do not, and train from scratch.  This script
builds the equivalent corpus so ALIGNN-CSP can be pretrained the same way.

**Leakage control.** The benchmark test set is drawn from the same JARVIS
database, so every material id in the benchmark's validation and test splits
is excluded here, and the ids are reported so the exclusion is auditable.
Pass ``--exclude-splits`` pointing at the prepared benchmark data directory.

Only jarvis-tools is needed (no pymatgen): pretraining never needs the target
POSCAR string, and the property column is a placeholder that training drops.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from tqdm import tqdm


def load_excluded_ids(data_dir: Path, splits=("val", "test")) -> set:
    """Material ids that must not appear in the pretraining corpus."""
    excluded = set()
    for name in splits:
        path = data_dir / f"{name}.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            excluded.add(str(r["material_id"]).strip())
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dft_3d")
    ap.add_argument("--id-key", default="jid")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--exclude-splits",
        default=None,
        help="prepared benchmark data dir whose val/test ids "
        "must be held out",
    )
    ap.add_argument(
        "--max-atoms",
        type=int,
        default=20,
        help="skip cells larger than this (the benchmark's "
        "largest cell is 18 atoms)",
    )
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = set()
    if args.exclude_splits:
        excluded = load_excluded_ids(Path(args.exclude_splits))
    print(f"holding out {len(excluded)} benchmark val/test ids")

    rows, n_excluded, n_toobig, n_bad = [], 0, 0, 0
    for item in tqdm(jarvis_data(args.dataset), desc=f"scan {args.dataset}"):
        mid = str(item[args.id_key]).strip()
        if mid in excluded:
            n_excluded += 1
            continue
        try:
            a = Atoms.from_dict(item["atoms"])
            n = len(a.elements)
            if n < 1 or n > args.max_atoms:
                n_toobig += 1
                continue
            lat = np.asarray(a.lattice_mat, dtype=float)
            if not np.isfinite(lat).all() or abs(np.linalg.det(lat)) < 1e-3:
                n_bad += 1
                continue
            rows.append(
                {
                    "material_id": mid,
                    "formula": a.composition.reduced_formula,
                    "spacegroup": -1,
                    # Placeholder: pretraining conditions on composition only,
                    # and the property modality is masked off.
                    "target": 0.0,
                    "lattice_mat": lat.tolist(),
                    "frac_coords": np.asarray(
                        a.frac_coords, dtype=float
                    ).tolist(),
                    "atomic_numbers": [int(z) for z in a.atomic_numbers],
                    "elements": list(a.elements),
                    "target_poscar": "",
                }
            )
        except Exception:
            n_bad += 1
            continue
        if args.max_size is not None and len(rows) >= args.max_size:
            break

    print(
        f"kept {len(rows)}  (excluded {n_excluded} held-out, "
        f"{n_toobig} too large, {n_bad} unusable)"
    )
    random.seed(args.seed)
    random.shuffle(rows)
    n_val = max(1, int(args.val_ratio * len(rows)))
    val, train = rows[:n_val], rows[n_val:]

    (out_dir / "train.json").write_text(json.dumps(train))
    (out_dir / "val.json").write_text(json.dumps(val))
    (out_dir / "split_meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_train": len(train),
                "n_val": len(val),
                "max_atoms": args.max_atoms,
                "n_excluded_heldout": n_excluded,
                "excluded_ids": sorted(excluded),
                "seed": args.seed,
            },
            indent=2,
        )
    )
    natoms = [len(r["atomic_numbers"]) for r in train]
    print(
        f"wrote train={len(train)} val={len(val)}  "
        f"natoms mean={np.mean(natoms):.1f} max={max(natoms)}"
    )


if __name__ == "__main__":
    main()
