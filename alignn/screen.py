"""Bulk multi-model property screening (ALIGNN DB).

Applies many trained pure-torch property models to a database of
structures, building each batch's graph ONCE and evaluating every model on
it. Output is sharded (parquet or gzipped JSONL) with atomic writes;
re-running skips completed shards, which makes long cluster runs
resumable.

Examples:
    # smoke test: 100 on-hull Alexandria structures, 2 models, CPU
    python alignn/screen.py --dataset alex_pbe_hull --limit 100 \
        --models formation_energy_peratom,mbj_bandgap --device cpu

    # production (cluster): all manifest models over the full input
    python alignn/screen.py --input alexandria_dump.jsonl \
        --models all --batch_size 64 --device cuda \
        --output_dir alignn_db_out --format parquet --shard_size 10000

Input structures need an id and a jarvis Atoms dict:
``--dataset <name>`` pulls a jarvis figshare dataset (e.g. ``alex_pbe_hull``,
``alex_pbe_3d_all``, ``dft_3d``); ``--input`` reads a local ``.json`` array
or ``.jsonl`` file of ``{"id": ..., "atoms": {...}}`` records.
"""

import argparse
import gzip
import json
import os
import time

import torch

from alignn.models.model_zoo import ModelZoo
from alignn.torch_graph_builder import (
    batch_torch_graph_pairs,
    build_pure_torch_graph,
)

ID_TAGS = ("id", "jid", "material_id", "mat_id")


def iter_structures(args):
    """Yield (id, atoms_dict) up to --limit."""
    count = 0

    def take(records):
        nonlocal count
        for i, rec in enumerate(records):
            if args.limit and count >= args.limit:
                return
            atoms = rec.get("atoms")
            if atoms is None:
                continue
            sid = None
            for tag in (args.id_tag,) + ID_TAGS:
                if tag and rec.get(tag) is not None:
                    sid = rec[tag]
                    break
            if sid is None:
                sid = "entry-%d" % i
            count += 1
            yield str(sid), atoms

    if args.input:
        if args.input.endswith(".jsonl"):
            def records():
                with open(args.input) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)

            yield from take(records())
        else:
            yield from take(json.load(open(args.input)))
    else:
        from jarvis.db.figshare import data as jdata

        yield from take(jdata(args.dataset))


def resolve_model_dirs(manifest_path, models_arg):
    """Manifest -> {property: model_dir}, downloading figshare models."""
    manifest = json.load(open(manifest_path))
    props = manifest["properties"]
    if models_arg != "all":
        wanted = [m.strip() for m in models_arg.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in props]
        if unknown:
            raise SystemExit(
                "Not in manifest: %s (have: %s)"
                % (unknown, sorted(props))
            )
        props = {k: props[k] for k in wanted}
    model_dirs = {}
    for name, entry in props.items():
        if entry.get("dir"):
            model_dirs[name] = entry["dir"]
        elif entry.get("figshare"):
            from alignn.ff.ff import get_figshare_model_ff

            model_dirs[name] = get_figshare_model_ff(
                model_name=entry["figshare"]
            )
        else:
            print("WARNING: no model source for", name, "- skipped")
    if not model_dirs:
        raise SystemExit("No models resolved from manifest.")
    return model_dirs


def check_graph_compat(model_dirs, args):
    """Warn when a model was trained on a different graph than we build."""
    for name, model_dir in model_dirs.items():
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        for key, ours in (
            ("cutoff", args.cutoff),
            ("max_neighbors", args.max_neighbors),
            ("atom_features", "cgcnn"),
        ):
            theirs = cfg.get(key)
            if theirs is not None and theirs != ours:
                print(
                    "WARNING: %s trained with %s=%s, screening uses %s"
                    % (name, key, theirs, ours)
                )


def build_batch(structs, args, device):
    """[(id, atoms_dict)] -> (ids, natoms, formulas, batched model input)."""
    from jarvis.core.atoms import Atoms

    ids, natoms, formulas, pairs, lattices = [], [], [], [], []
    for sid, atoms_dict in structs:
        atoms = Atoms.from_dict(atoms_dict)
        g, lg = build_pure_torch_graph(
            atoms=atoms,
            two_body_cutoff=args.cutoff,
            three_body_cutoff=args.three_body_cutoff,
            max_neighbors=args.max_neighbors,
            atom_features="cgcnn",
            compute_line_graph=True,
        )
        pairs.append((g, lg))
        lattices.append(
            torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())
        )
        ids.append(sid)
        natoms.append(atoms.num_atoms)
        formulas.append(atoms.composition.reduced_formula)
    g_b, lg_b = batch_torch_graph_pairs(pairs)
    lat_b = torch.stack(lattices)
    return ids, natoms, formulas, [
        g_b.to(device) if hasattr(g_b, "to") else g_b,
        lg_b.to(device) if hasattr(lg_b, "to") else lg_b,
        lat_b.to(device),
    ]


def write_shard(rows, columns, path, fmt):
    """Atomic write (tmp + rename) of one output shard."""
    tmp_path = path + ".tmp"
    if fmt == "parquet":
        import pandas as pd

        pd.DataFrame(rows, columns=columns).to_parquet(tmp_path, index=False)
    else:
        with gzip.open(tmp_path, "wt") as f:
            for row in rows:
                f.write(json.dumps(dict(zip(columns, row))) + "\n")
    os.replace(tmp_path, path)


def shard_path(output_dir, idx, fmt):
    ext = "parquet" if fmt == "parquet" else "jsonl.gz"
    return os.path.join(output_dir, "shard_%05d.%s" % (idx, ext))


def screen(args):
    device = torch.device(args.device)
    model_dirs = resolve_model_dirs(args.manifest, args.models)
    check_graph_compat(model_dirs, args)
    zoo = ModelZoo.from_model_dirs(model_dirs, map_location=args.device)
    prop_names = list(model_dirs)
    print("Models:", prop_names)
    os.makedirs(args.output_dir, exist_ok=True)
    columns = ["id", "formula", "natoms"] + prop_names

    n_done = 0
    n_failed = 0
    shard_idx = 0
    shard_structs = []
    t0 = time.time()

    def process_shard(structs, idx):
        nonlocal n_done, n_failed
        out_path = shard_path(args.output_dir, idx, args.format)
        if os.path.exists(out_path):
            print("shard", idx, "exists - skipped (resume)")
            n_done += len(structs)
            return
        rows = []
        for start in range(0, len(structs), args.batch_size):
            chunk = structs[start:start + args.batch_size]
            try:
                ids, natoms, formulas, inputs = build_batch(
                    chunk, args, device
                )
                with torch.no_grad():
                    preds = {}
                    for prop in prop_names:
                        out = zoo.predict(inputs, prop)["value"]
                        preds[prop] = (
                            out.detach().cpu().numpy().flatten().tolist()
                        )
                        if args.sequential_models:
                            zoo.evict(prop)
                for i, sid in enumerate(ids):
                    rows.append(
                        [sid, formulas[i], natoms[i]]
                        + [float(preds[p][i]) for p in prop_names]
                    )
                n_done += len(ids)
            except Exception as exp:
                # fall back to one-by-one so a single bad structure
                # doesn't take down the whole batch
                if len(chunk) == 1:
                    print("FAILED", chunk[0][0], ":", exp)
                    n_failed += 1
                    continue
                for single in chunk:
                    process_single(single, rows)
        write_shard(rows, columns, out_path, args.format)
        rate = n_done / max(time.time() - t0, 1e-9)
        print(
            "shard %d: %d rows (total %d done, %d failed, %.1f struct/s)"
            % (idx, len(rows), n_done, n_failed, rate)
        )

    def process_single(single, rows):
        nonlocal n_done, n_failed
        try:
            ids, natoms, formulas, inputs = build_batch(
                [single], args, device
            )
            with torch.no_grad():
                row = [ids[0], formulas[0], natoms[0]]
                for prop in prop_names:
                    out = zoo.predict(inputs, prop)["value"]
                    row.append(float(out.detach().cpu().numpy().flatten()[0]))
                    if args.sequential_models:
                        zoo.evict(prop)
            rows.append(row)
            n_done += 1
        except Exception as exp:
            print("FAILED", single[0], ":", exp)
            n_failed += 1

    for struct in iter_structures(args):
        shard_structs.append(struct)
        if len(shard_structs) >= args.shard_size:
            process_shard(shard_structs, shard_idx)
            shard_idx += 1
            shard_structs = []
    if shard_structs:
        process_shard(shard_structs, shard_idx)
        shard_idx += 1

    elapsed = time.time() - t0
    run_info = {
        "dataset": args.dataset,
        "input": args.input,
        "models": {k: str(v) for k, v in model_dirs.items()},
        "graph": {
            "cutoff": args.cutoff,
            "max_neighbors": args.max_neighbors,
            "three_body_cutoff": args.three_body_cutoff,
        },
        "n_structures": n_done,
        "n_failed": n_failed,
        "n_shards": shard_idx,
        "elapsed_s": elapsed,
        "structures_per_s": n_done / max(elapsed, 1e-9),
        "device": args.device,
        "format": args.format,
    }
    with open(os.path.join(args.output_dir, "manifest_run.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print(json.dumps(run_info, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Bulk multi-model property screening (ALIGNN DB)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset",
        help="jarvis figshare dataset name (alex_pbe_hull, "
        "alex_pbe_3d_all, dft_3d, ...).",
    )
    src.add_argument(
        "--input",
        help="Local .json array or .jsonl of {id, atoms} records.",
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(
            os.path.dirname(__file__), "scripts", "alignn_db_manifest.json"
        ),
    )
    parser.add_argument(
        "--models",
        default="all",
        help='"all" or comma list of manifest property names.',
    )
    parser.add_argument("--id_tag", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="alignn_db_out")
    parser.add_argument(
        "--format", default="parquet", choices=["parquet", "jsonl"]
    )
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many structures (0 = all).",
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--three_body_cutoff", type=float, default=3.5)
    parser.add_argument(
        "--sequential_models",
        action="store_true",
        help="Evict each model after use (lower memory, slower).",
    )
    args = parser.parse_args()
    if args.format == "parquet":
        try:
            import pandas  # noqa: F401
        except ImportError:
            raise SystemExit(
                "parquet output needs pandas+pyarrow; "
                "use --format jsonl instead"
            )
    screen(args)


if __name__ == "__main__":
    main()
