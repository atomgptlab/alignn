#!/usr/bin/env python3
"""Reproduce AtomBench's JARVIS Supercon-3D split for ALIGNN-CSP.

This mirrors ``atombench/tc_supercon/scripts/data_preprocess.py`` exactly so
that our train/val/test partition is identical to the one AtomGPT, CDVAE,
FlowMM and MatterGen were evaluated on:

  * iterate ``jarvis.db.figshare.data('dft_3d')`` in native order,
  * keep entries whose ``Tc_supercon`` is not 'na'/None,
  * drop entries where pymatgen's SpacegroupAnalyzer canonicalisation fails,
  * stop at ``--max-size`` records (1058 for Supercon-3D),
  * drop the known duplicate JVASP-19919 *before* shuffling,
  * ``random.seed(seed); random.shuffle(range(n))`` then 80/10/10,
  * drop the known leaked test IDs JVASP-20425 / JVASP-16080 *after* splitting.

Output is a single JSON per split holding everything the torch side needs
(lattice, fractional coords, atomic numbers, Tc) plus the exact POSCAR string
that must appear in the ``target`` column of the AtomBench benchmark CSV.

Must be run in an environment with pymatgen.
"""

from __future__ import annotations

import argparse
import json
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
from jarvis.core.atoms import Atoms, pmg_to_atoms
from jarvis.db.figshare import data as jarvis_data
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

# Hygiene constants copied verbatim from atombench's data_preprocess.py.
SUPERCON3D_DROP_BEFORE_SPLIT = "JVASP-19919"
SUPERCON3D_DROP_FROM_TEST_IDS = {"JVASP-20425", "JVASP-16080"}


def hash10(values):
    """Match atombench's split fingerprint so we can prove the split agrees."""
    h = sha256()
    for v in values:
        h.update(str(v).encode())
        h.update(b",")
    return h.hexdigest()[:10]


def canonicalise(pmg_struct, symprec: float = 0.1):
    """Return (cif_raw, cif_conv, spg_num, spg_conv); never raises."""
    try:
        sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec)
        spg_num = sga.get_space_group_number()
        cif_conv = sga.get_conventional_standard_structure().to(fmt="cif")
        spg_conv = SpacegroupAnalyzer(
            Structure.from_str(cif_conv, fmt="cif"), symprec=symprec
        ).get_space_group_number()
        return pmg_struct.to(fmt="cif"), cif_conv, spg_num, spg_conv
    except Exception:
        return "", "", -1, -1


def cif_to_poscar_string(cif_str: str) -> str:
    """Same round-trip the reference write_benchmark.py applies to targets."""
    pmg = Structure.from_str(cif_str, fmt="cif")
    return Poscar(pmg_to_atoms(pmg)).to_string()


def collect_records(dataset: str, target_key: str, id_key: str, max_size):
    records = []
    for item in tqdm(jarvis_data(dataset), desc=f"scan {dataset}"):
        tgt = item.get(target_key, "na")
        if tgt in ("na", None):
            continue
        atoms_j = Atoms.from_dict(item["atoms"])
        pmg = atoms_j.pymatgen_converter()
        try:
            cif_raw, _cif_conv, spg_raw, _spg_conv = canonicalise(pmg)
            if not cif_raw:
                continue
        except Exception:
            continue
        records.append(
            {
                "material_id": item[id_key],
                "atoms_j": atoms_j,
                "cif_raw": cif_raw,
                "spg_raw": spg_raw,
                "formula": pmg.composition.reduced_formula,
                target_key: float(tgt),
            }
        )
        if max_size is not None and len(records) == max_size:
            break
    return records


def deterministic_split(n_samples, val_ratio, test_ratio, seed):
    indices = list(range(n_samples))
    random.seed(seed)
    random.shuffle(indices)
    n_val = int(val_ratio * n_samples)
    n_test = int(test_ratio * n_samples)
    n_train = n_samples - n_val - n_test
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def serialise(rec, target_key):
    """Turn one record into a JSON-safe dict for the torch dataloader."""
    a = rec["atoms_j"]
    return {
        "material_id": rec["material_id"],
        "formula": rec["formula"],
        "spacegroup": int(rec["spg_raw"]),
        "target": rec[target_key],
        "lattice_mat": np.asarray(a.lattice_mat, dtype=float).tolist(),
        "frac_coords": np.asarray(a.frac_coords, dtype=float).tolist(),
        "atomic_numbers": [int(z) for z in a.atomic_numbers],
        "elements": list(a.elements),
        # POSCAR string for the benchmark CSV `target` column, produced by the
        # same cif round-trip the reference pipeline uses.
        "target_poscar": cif_to_poscar_string(rec["cif_raw"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dft_3d")
    ap.add_argument("--target", dest="target_key", default="Tc_supercon")
    ap.add_argument("--id-key", default="jid")
    ap.add_argument("--max-size", type=int, default=1058)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(
        args.dataset, args.target_key, args.id_key, args.max_size
    )
    print(f"collected {len(records)} usable entries")

    if args.dataset.strip().lower() == "dft_3d":
        before = len(records)
        records = [
            r
            for r in records
            if str(r["material_id"]).strip() != SUPERCON3D_DROP_BEFORE_SPLIT
        ]
        print(f"hygiene: dropped {before - len(records)} duplicate row(s)")

    id_train, id_val, id_test = deterministic_split(
        len(records), args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"pre-hygiene  train:{len(id_train)} val:{len(id_val)} "
        f"test:{len(id_test)}"
    )

    if args.dataset.strip().lower() == "dft_3d":
        id_test = [
            i
            for i in id_test
            if str(records[i]["material_id"]).strip()
            not in SUPERCON3D_DROP_FROM_TEST_IDS
        ]
    print(
        f"post-hygiene train:{len(id_train)} val:{len(id_val)} "
        f"test:{len(id_test)}"
    )

    # atombench's AtomGPT factory hashes ids in order train+val+test.
    id_all = id_train + id_val + id_test
    print(
        "hash10(ids)="
        + hash10([records[i]["material_id"] for i in id_all])
        + "  (compare against atombench's AtomGPT prep output)"
    )

    for name, ids in (
        ("train", id_train),
        ("val", id_val),
        ("test", id_test),
    ):
        rows = [serialise(records[i], args.target_key) for i in ids]
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(rows))
        n_at = [len(r["atomic_numbers"]) for r in rows]
        print(
            f"wrote {path}  n={len(rows)}  natoms min/mean/max="
            f"{min(n_at)}/{sum(n_at) / len(n_at):.1f}/{max(n_at)}"
        )

    (out_dir / "split_meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "target_key": args.target_key,
                "seed": args.seed,
                "max_size": args.max_size,
                "n_train": len(id_train),
                "n_val": len(id_val),
                "n_test": len(id_test),
                "hash10_ids": hash10(
                    [records[i]["material_id"] for i in id_all]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
