"""Build a per-atom (atomwise) training dataset: charges and magmoms.

Writes ``id_prop.json`` in the schema consumed by
``alignn.train_alignn --atomwise_key {charges,magmoms}``:

    {"jid": ..., "atoms": {...}, "target": <cell-level scalar>,
     "charges": [N floats], "magmoms": [N floats]}

Sources (pick one):

- ``--source json``: an existing JSON array with per-atom labels (e.g. the
  in-house dft_3d charges+magmoms set behind the DGL
  ``v2024.12.12_dft_3d_multi_prop`` model). Use ``--charges_key`` /
  ``--magmoms_key`` to map its field names.
- ``--source outcar``: a directory tree of VASP runs; site magmoms are read
  from each OUTCAR via ``jarvis.io.vasp.outputs.Outcar.magnetization`` and
  structures from the matching POSCAR/CONTCAR. (Charges require a separate
  Bader preprocessing step; entries without charges get ``charges: null``
  and are skipped when training with ``--atomwise_key charges``.)

Usage:
    python make_atomwise_dataset.py --source json --input raw.json \
        --output_dir dataset_atomwise
    python make_atomwise_dataset.py --source outcar --input /path/to/runs \
        --output_dir dataset_atomwise
"""

import argparse
import json
import os


def entry_ok(atoms, values):
    """Per-atom label list must match the atom count."""
    return (
        isinstance(values, (list, tuple))
        and len(values) == len(atoms["elements"])
        and all(v is not None for v in values)
    )


def from_json(args):
    raw = json.load(open(args.input))
    entries = []
    skipped = 0
    for i, item in enumerate(raw):
        atoms = item.get(args.atoms_key)
        charges = item.get(args.charges_key)
        magmoms = item.get(args.magmoms_key)
        jid = item.get(args.id_key, "entry-%d" % i)
        keep_charges = charges if atoms and entry_ok(atoms, charges) else None
        keep_magmoms = magmoms if atoms and entry_ok(atoms, magmoms) else None
        if atoms is None or (keep_charges is None and keep_magmoms is None):
            skipped += 1
            continue
        target = item.get(args.target_key)
        if target is None:
            target = (
                sum(keep_magmoms) if keep_magmoms is not None else 0.0
            )
        entries.append(
            {
                "jid": jid,
                "atoms": atoms,
                "target": target,
                "charges": keep_charges,
                "magmoms": keep_magmoms,
            }
        )
    print("kept", len(entries), "skipped", skipped)
    return entries


def from_outcar(args):
    from jarvis.core.atoms import Atoms
    from jarvis.io.vasp.outputs import Outcar

    entries = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(args.input):
        if "OUTCAR" not in filenames:
            continue
        struct_file = None
        for candidate in ("CONTCAR", "POSCAR"):
            if candidate in filenames:
                struct_file = os.path.join(dirpath, candidate)
                break
        if struct_file is None:
            skipped += 1
            continue
        try:
            atoms = Atoms.from_poscar(struct_file)
            outcar = Outcar(os.path.join(dirpath, "OUTCAR"))
            magmoms = [float(m) for m in outcar.magnetization]
        except Exception as exp:
            print("skip", dirpath, exp)
            skipped += 1
            continue
        atoms_dict = atoms.to_dict()
        if not entry_ok(atoms_dict, magmoms):
            skipped += 1
            continue
        entries.append(
            {
                "jid": os.path.relpath(dirpath, args.input),
                "atoms": atoms_dict,
                "target": sum(magmoms),
                "charges": None,
                "magmoms": magmoms,
            }
        )
    print("kept", len(entries), "skipped", skipped)
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Build per-atom charges/magmoms id_prop.json."
    )
    parser.add_argument("--source", choices=["json", "outcar"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="dataset_atomwise")
    parser.add_argument("--id_key", default="jid")
    parser.add_argument("--atoms_key", default="atoms")
    parser.add_argument("--charges_key", default="charges")
    parser.add_argument("--magmoms_key", default="magmoms")
    parser.add_argument(
        "--target_key",
        default="target",
        help="Optional cell-level scalar in the source (fallback: sum of "
        "magmoms; the graph-level loss is disabled during training anyway).",
    )
    args = parser.parse_args()

    entries = from_json(args) if args.source == "json" else from_outcar(args)
    if not entries:
        raise SystemExit("No usable entries.")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "id_prop.json")
    with open(out_path, "w") as f:
        json.dump(entries, f)
    n_charges = sum(1 for e in entries if e["charges"] is not None)
    n_magmoms = sum(1 for e in entries if e["magmoms"] is not None)
    print(
        "wrote {} ({} entries: {} with charges, {} with magmoms)".format(
            out_path, len(entries), n_charges, n_magmoms
        )
    )


if __name__ == "__main__":
    main()
