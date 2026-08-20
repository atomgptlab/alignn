#!/usr/bin/env python3
"""Generate structures with ALIGNN-CSP and write an AtomBench benchmark CSV.

For each entry in the split, the model is conditioned on that entry's
composition and target property, ``--num-candidates`` structures are sampled,
and the best one is written as the prediction.  "Best" is decided by the
pipeline stages you enable:

    --relax cell   relax each candidate with ALIGNN-FF (positions + lattice)
    --rank energy  keep the lowest ALIGNN-FF energy per atom

Output columns are ``id,target,prediction`` with POSCAR strings whose newlines
are escaped as a literal backslash-n, matching the reference
``write_benchmark.py`` in the AtomBench repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from jarvis.io.vasp.inputs import Poscar

from alignn.inverse.data import compositions_from_split, make_generation_batch
from alignn.inverse.sample import load_model, sample, to_jarvis_atoms


def poscar_field(jatoms_or_str) -> str:
    """POSCAR string with newlines escaped, as the benchmark CSV expects."""
    if isinstance(jatoms_or_str, str):
        text = jatoms_or_str
    else:
        text = Poscar(jatoms_or_str).to_string()
    return text.replace("\n", r"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the number of denoising steps",
    )
    ap.add_argument("--n-corrector", type=int, default=1)
    ap.add_argument("--step-lr", type=float, default=1e-5)
    ap.add_argument(
        "--modalities",
        default=None,
        help="comma-separated subset to condition on "
        "(default: all the model was trained with)",
    )
    ap.add_argument(
        "--relax", choices=["none", "positions", "cell"], default="cell"
    )
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--relax-fmax", type=float, default=0.05)
    ap.add_argument("--rank", choices=["none", "energy"], default="energy")
    ap.add_argument(
        "--relax-workers",
        type=int,
        default=None,
        help="CPU processes for relaxation (default: ncores-2)",
    )
    ap.add_argument(
        "--prescreen-keep",
        type=int,
        default=None,
        help="rank all candidates by single-point energy first, "
        "then relax only this many per target",
    )
    ap.add_argument(
        "--max-batch-nodes",
        type=int,
        default=4096,
        help="cap on atoms per sampling batch",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N targets (for smoke tests)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument(
        "--save-candidates",
        default=None,
        help="optional JSON dump of every candidate + energy",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model, schedule, normalizer, cfg = load_model(
        args.checkpoint, device, use_ema=bool(args.use_ema)
    )
    if args.steps is not None:
        from alignn.inverse.diffusion import DiffusionSchedule

        schedule = DiffusionSchedule(
            num_steps=args.steps,
            sigma_min=cfg["sigma_min"],
            sigma_max=cfg["sigma_max"],
        ).to(device)
        model.denoiser.num_steps = args.steps
    active = (
        [m.strip() for m in args.modalities.split(",")]
        if args.modalities
        else None
    )
    print(f"model modalities={model.modalities} active={active or 'all'}")

    items = compositions_from_split(Path(args.data_dir) / f"{args.split}.json")
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} targets, {args.num_candidates} candidates each")

    # Replicate each target M times so all its candidates are sampled in the
    # same batch, then chunk by atom count to bound memory.
    expanded = []
    for idx, it in enumerate(items):
        for c in range(args.num_candidates):
            expanded.append((idx, c, it))

    chunks, cur, cur_nodes = [], [], 0
    for rec in expanded:
        n = len(rec[2]["atomic_numbers"])
        if cur and cur_nodes + n > args.max_batch_nodes:
            chunks.append(cur)
            cur, cur_nodes = [], 0
        cur.append(rec)
        cur_nodes += n
    if cur:
        chunks.append(cur)

    candidates = [[] for _ in items]
    t0 = time.time()
    for ci, chunk in enumerate(chunks, 1):
        batch = make_generation_batch([r[2] for r in chunk], device)
        out = sample(
            model,
            schedule,
            normalizer,
            batch,
            guidance=args.guidance,
            active_modalities=active,
            n_corrector=args.n_corrector,
            step_lr=args.step_lr,
        )
        atoms = to_jarvis_atoms(
            out["frac"],
            out["lattice"],
            batch["atomic_numbers"],
            batch["natoms"],
        )
        for (idx, _c, _it), a in zip(chunk, atoms):
            candidates[idx].append(a)
        print(
            f"  sampled chunk {ci}/{len(chunks)} "
            f"({len(chunk)} structures, {time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── relax / rank ─────────────────────────────────────────────────────
    predictions, records = [], []
    if args.relax == "none" and args.rank == "none":
        predictions = [(it, cands[0]) for it, cands in zip(items, candidates)]
    else:
        from alignn.inverse.relax_rank import parallel_rank

        t1 = time.time()
        pool = candidates
        keep = args.prescreen_keep
        if keep and args.relax != "none" and args.num_candidates > keep:
            # Two-stage screen. A single-point energy costs ~0.3 s while a
            # relaxation costs ~100x that, so score every candidate cheaply
            # first and spend the relaxation budget only on the survivors.
            # This is what makes a large candidate pool affordable, and the
            # pool size is the strongest lever on match rate.
            screened = parallel_rank(
                candidates,
                relax=False,
                n_workers=args.relax_workers,
                progress_every=200,
            )
            pool = [[r.atoms for r in g[:keep]] for g in screened]
            print(
                f"  prescreened {args.num_candidates} -> {keep} per target "
                f"in {time.time() - t1:.0f}s",
                flush=True,
            )
        ranked_groups = parallel_rank(
            pool,
            relax=(args.relax != "none"),
            relax_cell=(args.relax == "cell"),
            fmax=args.relax_fmax,
            steps=args.relax_steps,
            n_workers=args.relax_workers,
        )
        print(f"  relax/rank done in {time.time() - t1:.0f}s", flush=True)
        for it, group in zip(items, ranked_groups):
            predictions.append((it, group[0].atoms))
            records.append(
                {
                    "id": it["material_id"],
                    "energies": [r.energy_per_atom for r in group],
                    "converged": [r.converged for r in group],
                    "steps": [r.steps for r in group],
                    "errors": [r.error for r in group],
                }
            )
        n_conv = sum(bool(g[0].converged) for g in ranked_groups)
        print(f"  best candidate converged for {n_conv}/{len(items)} targets")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "target", "prediction"])
        for it, atoms in predictions:
            w.writerow(
                [
                    it["material_id"],
                    poscar_field(it["target_poscar"]),
                    poscar_field(atoms),
                ]
            )
    print(f"wrote {out_path} ({len(predictions)} rows)")

    if args.save_candidates and records:
        Path(args.save_candidates).write_text(json.dumps(records, indent=2))

    (out_path.parent / "generation_config.json").write_text(
        json.dumps(vars(args), indent=2)
    )


if __name__ == "__main__":
    main()
