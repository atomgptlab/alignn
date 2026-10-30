#!/usr/bin/env python3
"""Reproduce AtomBench's Alexandria DS-A/DS-B split for ALIGNN-CSP.

Mirrors ``atombench/alexandria/scripts/alexandria_preprocess.py`` so the
train/val/test partition matches the one AtomGPT, CDVAE, FlowMM and MatterGen
were evaluated on:

  * concatenate the DS-A and DS-B tables in that order,
  * iterate rows in order, skipping any with a missing/NaN ``Tc``, or a
    ``structure`` cell that will not parse,
  * stop at ``--max-size`` records (8253 for DS-A/B),
  * ``random.seed(seed); random.shuffle(range(n))`` then 80/10/10.

Unlike the JARVIS Supercon-3D pipeline there are no duplicate/leakage drops.

Must be run in an environment with pymatgen.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import json
import pickle
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.atoms import pmg_to_atoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def hash10(values):
    h = sha256()
    for v in values:
        h.update(str(v).encode())
        h.update(b",")
    return h.hexdigest()[:10]


def canonicalise(pmg, symprec: float = 0.1):
    """Return (cif_raw, cif_conv, spg_raw, spg_conv). Always succeeds."""
    try:
        cif_raw = pmg.to(fmt="cif")
    except Exception:
        cif_raw = ""
    try:
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        spg_raw = sga.get_space_group_number()
        conv = sga.get_conventional_standard_structure()
        cif_conv = conv.to(fmt="cif")
        spg_conv = SpacegroupAnalyzer(
            Structure.from_str(cif_conv, fmt="cif"), symprec=symprec
        ).get_space_group_number()
    except Exception:
        cif_conv, spg_raw, spg_conv = "", -1, -1
    return cif_raw, cif_conv, spg_raw, spg_conv


def _shim_legacy_pandas():
    """Let modern pandas unpickle the DS-*.pk files.

    The published pickles were written with pandas 1.x, whose
    ``pandas.core.indexes.numeric`` module no longer exists; the reference
    pipeline only works because it runs on the old version. Aliasing the
    removed index classes to ``pd.Index`` reads them without a downgrade.
    """
    import sys
    import types

    name = "pandas.core.indexes.numeric"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for cls in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(mod, cls, pd.Index)
    sys.modules[name] = mod


def load_tables(paths):
    """Read the DS-*.pk.bz2 pickles (or CSVs) into one concatenated frame."""
    _shim_legacy_pandas()
    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".csv":
            frames.append(pd.read_csv(p))
        else:
            with bz2.open(p, "rb") as fh:
                frames.append(pickle.load(fh))
    return pd.concat(frames, ignore_index=True)


def as_structure(value):
    """The `structure` column holds either a dict, or its repr as a string."""
    if isinstance(value, Structure):
        return value
    if isinstance(value, dict):
        return Structure.from_dict(value)
    return Structure.from_dict(ast.literal_eval(value))


def deterministic_split(n, val_ratio, test_ratio, seed):
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="DS-A.pk.bz2 DS-B.pk.bz2 (in that order)",
    )
    ap.add_argument("--id-key", default="mat_id")
    ap.add_argument("--target", dest="target_key", default="Tc")
    ap.add_argument("--max-size", type=int, default=8253)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_tables(args.inputs)
    print(f"loaded {len(df_all)} rows; columns: {list(df_all.columns)[:12]}")

    records = []
    for row in tqdm(
        df_all.itertuples(index=False), total=len(df_all), desc="parsing"
    ):
        if len(records) >= args.max_size:
            break
        tgt = getattr(row, args.target_key, "na")
        if tgt in (None, "na") or (isinstance(tgt, float) and np.isnan(tgt)):
            continue
        try:
            pmg = as_structure(getattr(row, "structure"))
            atoms = pmg_to_atoms(pmg)
        except Exception:
            continue
        cif_raw, _cif_conv, spg_raw, _spg_conv = canonicalise(pmg)
        records.append(
            {
                "material_id": getattr(row, args.id_key),
                "formula": pmg.composition.reduced_formula,
                "spacegroup": int(spg_raw),
                "target": float(tgt),
                "lattice_mat": np.asarray(
                    atoms.lattice_mat, dtype=float
                ).tolist(),
                "frac_coords": np.asarray(
                    atoms.frac_coords, dtype=float
                ).tolist(),
                "atomic_numbers": [int(z) for z in atoms.atomic_numbers],
                "elements": list(atoms.elements),
                # Target column of the benchmark CSV, via the same cif
                # round-trip the reference write_benchmark.py applies.
                "target_poscar": (
                    Poscar(
                        pmg_to_atoms(Structure.from_str(cif_raw, fmt="cif"))
                    ).to_string()
                    if cif_raw
                    else Poscar(atoms).to_string()
                ),
            }
        )

    print(f"collected {len(records)} usable entries")
    id_train, id_val, id_test = deterministic_split(
        len(records), args.val_ratio, args.test_ratio, args.seed
    )
    print(f"split train:{len(id_train)} val:{len(id_val)} test:{len(id_test)}")
    id_all = id_train + id_val + id_test
    fingerprint = hash10([records[i]["material_id"] for i in id_all])
    print(f"hash10(ids)={fingerprint}")

    for name, ids in (
        ("train", id_train),
        ("val", id_val),
        ("test", id_test),
    ):
        rows = [records[i] for i in ids]
        (out_dir / f"{name}.json").write_text(json.dumps(rows))
        n_at = [len(r["atomic_numbers"]) for r in rows]
        print(
            f"wrote {name}.json n={len(rows)} natoms "
            f"min/mean/max={min(n_at)}/{sum(n_at) / len(n_at):.1f}/{max(n_at)}"
        )

    (out_dir / "split_meta.json").write_text(
        json.dumps(
            {
                "dataset": "alexandria_DS-A_DS-B",
                "target_key": args.target_key,
                "seed": args.seed,
                "max_size": args.max_size,
                "n_train": len(id_train),
                "n_val": len(id_val),
                "n_test": len(id_test),
                "hash10_ids": fingerprint,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
