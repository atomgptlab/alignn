#!/usr/bin/env python3
"""Run one of the manuscript's inverse-design tasks.

    python task_runners/run_task.py tasks                # what is available
    python task_runners/run_task.py doctor               # is this box ready
    python task_runners/run_task.py verify               # paper -> task map
    python task_runners/run_task.py bench-jarvis --list  # the units
    python task_runners/run_task.py bench-jarvis --unit 0
    python task_runners/run_task.py bench-jarvis         # all units, in order
    python task_runners/run_task.py bench-jarvis --aggregate

Three cost settings.  Full is the default.  ``--quick`` trains for 300 epochs
with 8 candidates on the whole test split, which is a real arm-vs-arm
comparison at roughly a tenth of the cost; ``--smoke`` is 2 epochs on 4
targets and proves only that the plumbing works.  Both write into their own
run tree, so a cheap run can never overwrite an expensive checkpoint.

Every stage records the exact command it ran in ``<rundir>/.stages``, and is
skipped on a re-run if that command has not changed.  Change a hyperparameter
and the affected stages re-run; change nothing and the task resumes where it
stopped.  ``--force`` ignores the markers.

Under SLURM, ``--unit $SLURM_ARRAY_TASK_ID`` makes each array element one
unit; ``task_runners/submit.sh`` sizes the array from ``--count``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_runners import tasks as T  # noqa: E402

REPO = T.REPO


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def marker_path(unit: T.Unit, stage: T.Stage) -> Path:
    return unit.rundir / ".stages" / f"{stage.name}.json"


def already_done(unit: T.Unit, stage: T.Stage) -> bool:
    """True if this exact command has completed here before."""
    path = marker_path(unit, stage)
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return rec.get("argv") == stage.argv and rec.get("env") == stage.env


def check_requires(stage: T.Stage) -> List[str]:
    """Inputs the stage needs that are not there yet."""
    return [
        f"{path} (produced by {producer})"
        for path, producer in stage.requires
        if not Path(path).exists()
    ]


def run_stage(
    unit: T.Unit, stage: T.Stage, *, dry_run: bool, force: bool
) -> str:
    """Run one stage.  Returns 'skip', 'ok', 'blocked' or 'fail'."""
    label = f"[{unit.name}/{stage.name}]"
    if already_done(unit, stage) and not force:
        print(f"{label} skip (already done)")
        return "skip"

    missing = check_requires(stage)
    if missing and not dry_run:
        print(f"{label} BLOCKED, missing input(s):")
        for m in missing:
            print(f"    {m}")
        return "blocked"

    env = dict(os.environ, PYTHONUNBUFFERED="1", **stage.env)
    printable = " ".join(stage.argv)
    if dry_run:
        prefix = " ".join(f"{k}={v}" for k, v in stage.env.items())
        print(f"{label} {prefix + ' ' if prefix else ''}{printable}")
        return "ok"

    print(f"{label} $ {printable}", flush=True)
    unit.rundir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(stage.argv, cwd=REPO, env=env)
    elapsed = time.time() - started
    if proc.returncode != 0:
        print(f"{label} FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
        return "fail"

    path = marker_path(unit, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "argv": stage.argv,
                "env": stage.env,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": round(elapsed, 1),
                "host": platform.node(),
                "git": git_rev(),
            },
            indent=2,
        )
    )
    print(f"{label} done in {elapsed:.0f}s", flush=True)
    return "ok"


def git_rev() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def select_stages(
    unit: T.Unit, only: Sequence[str], skip: Sequence[str]
) -> List[T.Stage]:
    """Filter a unit's stages, keeping their order."""
    stages = unit.stages
    if only:
        stages = [s for s in stages if s.name in only]
    if skip:
        stages = [s for s in stages if s.name not in skip]
    return stages


def run_unit(
    unit: T.Unit,
    *,
    dry_run: bool,
    force: bool,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> bool:
    """Run a unit's stages in order.  Stops at the first failure."""
    print(f"\n=== unit {unit.name}  ->  {unit.rundir}")
    stages = select_stages(unit, only, skip)
    if not stages:
        # Silently doing nothing and reporting success is the worst outcome
        # here: --loss-only on a task with no training stage would look like
        # a completed run.
        have = ", ".join(s.name for s in unit.stages)
        print(f"    nothing to run: the filter left no stages of [{have}]")
        return True
    for stage in stages:
        status = run_stage(unit, stage, dry_run=dry_run, force=force)
        if status in ("fail", "blocked"):
            return False
    return True


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def build_ctx(args, task: T.Task) -> T.Ctx:
    seeds = (
        tuple(int(s) for s in args.seeds.split(","))
        if args.seeds
        else tuple(task.default_seeds)
    )
    if args.smoke:
        seeds = seeds[:1]
    elif args.quick and not args.seeds:
        seeds = seeds[: T.QUICK["seeds"]]
    return T.Ctx(
        runs=Path(args.runs_root).resolve(),
        seeds=seeds,
        epochs=args.epochs,
        device=args.device,
        num_candidates=args.num_candidates,
        guidance=args.guidance,
        symprec=args.symprec,
        relax_workers=args.relax_workers,
        relax_steps=args.relax_steps,
        limit=args.limit,
        checkpoint=args.checkpoint,
        alex_inputs=tuple(args.alex_inputs or ()),
        smoke=args.smoke,
        quick=args.quick,
        from_scratch=args.from_scratch,
    )


# ---------------------------------------------------------------------------
# Sub-commands that are not tasks
# ---------------------------------------------------------------------------


def cmd_tasks() -> int:
    width = max(len(n) for n in T.TASKS)
    print("\nInverse-design tasks (task_runners/tasks.py)\n")
    for name, task in T.TASKS.items():
        print(f"  {name.ljust(width)}  {task.summary}")
        print(f"  {' ' * width}  reproduces: {task.reproduces}")
        if task.needs:
            print(f"  {' ' * width}  needs: {', '.join(task.needs)}")
        print()
    return 0


def cmd_verify(args) -> int:
    """Map every published number to a task, and to what was measured."""
    from task_runners import aggregate, claims

    print("\nManuscript coverage -- inverse design\n")
    if args.quick or args.smoke:
        mode = "smoke" if args.smoke else "quick"
        print(
            f"  NOTE: --{mode} reads a separate run tree whose arms are "
            "comparable to\n        each other but not to the published "
            "numbers. Expect disagreement.\n"
        )
    orphans = [c for c in claims.CLAIMS if c.task not in T.TASKS]
    if orphans:
        for claim in orphans:
            print(f"  NO TASK for {claim.source}: {claim.statement}")
        print(f"\n{len(orphans)} claim(s) have no runner. This is a bug.")
        return 2

    # Build each task once; a task's units are the same for every claim on it.
    built = {}
    for name in claims.tasks_needed():
        task = T.get(name)
        built[name] = (task, task.build(build_ctx(args, task)))

    header = (
        f"  {'source':<16}{'claim':<52}{'published':>10}"
        f"{'measured':>20}{'':>3}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    counts = {"ok": 0, "within sd": 0, "differs": 0, "not run": 0}
    todo = []
    for claim in claims.CLAIMS:
        task, units = built[claim.task]
        result = aggregate.check(task, units, claim)
        counts[result["status"]] += 1
        if result["status"] == "not run" and claim.task not in todo:
            todo.append(claim.task)
        measured = result["measured"]
        if measured is None:
            shown = "-"
        elif result["sd"]:
            shown = f"{measured:.4g} +-{result['sd']:.2g}"
        else:
            shown = f"{measured:.4g}"
        mark = {
            "ok": "ok",
            "within sd": "~",
            "differs": "!!",
            "not run": "",
        }[result["status"]]
        print(
            f"  {claim.source:<16}{claim.statement[:50]:<52}"
            f"{claim.published:>10.4g}{shown:>20}{mark:>3}"
        )

    print(
        "\n  ok = within the claim's tolerance   ~ = within one measured "
        "standard deviation\n  !! = outside both   blank = not run yet"
    )
    print(f"\n  {len(claims.CLAIMS)} published numbers, all mapped to a task.")
    print(
        f"  ok {counts['ok']}   within sd {counts['within sd']}   "
        f"differs {counts['differs']}   not run {counts['not run']}"
    )
    if todo:
        print(
            "\n  to fill the gaps, in dependency order (a prerequisite that "
            "is\n  already finished costs nothing to re-run -- its stages "
            "are skipped):"
        )
        for name in _ordered(todo):
            print(f"    python task_runners/run_task.py {name}")
    print()
    return 0


def _ordered(names: Sequence[str]) -> List[str]:
    """Topologically sort task names by their declared prerequisites.

    Ties are broken by registration order in ``TASKS``, which is the order a
    person would run them in, so the list reads as a plan rather than as
    whichever claim happened to be listed first.
    """
    order = list(T.TASKS)
    out: List[str] = []

    def visit(name: str) -> None:
        if name in out or name not in T.TASKS:
            return
        for need in T.TASKS[name].needs:
            visit(need)
        if name not in out:
            out.append(name)

    for name in sorted(names, key=lambda n: order.index(n)):
        visit(name)
    return out


def _probe(label: str, fn) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a doctor reports, never raises
        print(f"  [ ] {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"  [x] {label}: {detail}")
    return True


def cmd_doctor(runs_root: Path) -> int:
    """Check the things that make a task fail an hour in, not a minute in."""
    print("\nEnvironment")
    ok = True

    def _torch():
        import torch

        cuda = (
            f"cuda {torch.version.cuda}, "
            f"{torch.cuda.device_count()} device(s)"
            if torch.cuda.is_available()
            else "no CUDA (use --device cpu)"
        )
        # Importing torch is not the same as being able to train with it: a
        # mismatched install can import cleanly and then fail inside the
        # optimiser, an hour into a queued job. Take one step here instead.
        net = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        net(torch.zeros(1, 4)).sum().backward()
        opt.step()
        return f"{torch.__version__}, {cuda}, one optimiser step ok"

    ok &= _probe("torch", _torch)
    ok &= _probe(
        "alignn.inverse",
        lambda: __import__(
            "alignn.inverse.train_csp", fromlist=["main"]
        ).__name__,
    )
    ok &= _probe(
        "jarvis-tools",
        lambda: __import__("jarvis").__version__,
    )
    ok &= _probe("pymatgen", lambda: __import__("pymatgen.core").core.__name__)
    ok &= _probe(
        "average-minimum-distance (ccRMSD)",
        lambda: __import__("amd").__version__,
    )

    print("\nScoring")
    compute = find_compute_metrics()
    if compute:
        print(f"  [x] AtomBench compute_metrics.py: {compute}")
    else:
        ok = False
        print(
            "  [ ] AtomBench compute_metrics.py not found. Clone\n"
            "      https://github.com/atomgptlab/atombench and set\n"
            "      ATOMBENCH_REPO to it (score.sh also looks in ~/atombench)."
        )

    print(f"\nRun root: {runs_root}")
    for name in ("jarvis", "alex", "pretrain"):
        path = runs_root / "data" / name
        mark = "x" if (path / "train.json").exists() else " "
        print(f"  [{mark}] data/{name}")

    print()
    return 0 if ok else 1


def _names(spec: Optional[str]) -> tuple:
    if not spec:
        return ()
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def find_compute_metrics() -> Optional[Path]:
    """Mirror score.sh's search for AtomBench's metric script."""
    candidates = []
    if os.environ.get("ATOMBENCH_REPO"):
        candidates.append(Path(os.environ["ATOMBENCH_REPO"]))
    candidates += [Path.home() / "atombench", REPO.parent / "atombench"]
    for repo in candidates:
        path = repo / "scripts" / "scripts_consolidated" / "compute_metrics.py"
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "task",
        help="task name, or one of: tasks (list them), doctor (check this "
        "machine), verify (map the paper's numbers to tasks and to what "
        "has been measured)",
    )
    ap.add_argument(
        "--unit",
        type=int,
        default=None,
        help="run only this unit index (SLURM array element)",
    )
    ap.add_argument("--list", action="store_true", help="list units and exit")
    ap.add_argument(
        "--count",
        action="store_true",
        help="print the number of units and exit (sizes an sbatch array)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="re-run completed stages"
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="summarise this task's results instead of running it",
    )
    ap.add_argument("--latex", action="store_true", help="with --aggregate")
    ap.add_argument(
        "--variant",
        default=None,
        help="with --aggregate: which scored CSV to read, 'sym' or 'nosym' "
        "(default: whichever the task's published numbers are quoted on)",
    )

    ap.add_argument(
        "--runs-root",
        default=os.environ.get("ALIGNN_RUNS", str(REPO / "runs")),
        help="where data, checkpoints and results live "
        "(env ALIGNN_RUNS; default <repo>/runs)",
    )
    ap.add_argument(
        "--seeds", default=None, help="comma-separated, e.g. 0,1,2"
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--device", default=os.environ.get("ALIGNN_DEVICE", "cuda")
    )
    ap.add_argument("--num-candidates", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="symmetrisation tolerance; choose it with symprec-sweep",
    )
    ap.add_argument("--relax-steps", type=int, default=None)
    ap.add_argument("--relax-workers", type=int, default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N targets only"
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint for pipeline-ablation / symprec-sweep "
        "(default: this task's first seed)",
    )
    ap.add_argument(
        "--alex-inputs",
        nargs="+",
        default=None,
        help="DS-A.pk.bz2 DS-B.pk.bz2, in that order",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs, 2 candidates, 4 targets, one seed: plumbing only",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help=f"{T.QUICK['epochs']} epochs, {T.QUICK['num_candidates']} "
        f"candidates, {T.QUICK['seeds']} seeds, whole test split: a real "
        "arm-vs-arm comparison at a fraction of the cost",
    )
    ap.add_argument(
        "--loss-only",
        action="store_true",
        help="train and stop. The denoising loss is the cheapest and most "
        "reproducible arm-vs-arm signal, and needs no scoring environment",
    )
    ap.add_argument(
        "--only-stages",
        default=None,
        help="comma-separated stage names to run (train, generate, "
        "symmetrize, score-nosym, score-sym)",
    )
    ap.add_argument(
        "--skip-stages", default=None, help="comma-separated stages to skip"
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="bench-alex: train on Alexandria alone, no pretrained init",
    )
    args = ap.parse_args()

    if args.task == "tasks":
        return cmd_tasks()
    if args.task == "doctor":
        return cmd_doctor(Path(args.runs_root).resolve())
    if args.task == "verify":
        return cmd_verify(args)

    try:
        task = T.get(args.task)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    ctx = build_ctx(args, task)
    units = task.build(ctx)

    if args.count:
        print(len(units))
        return 0

    if args.aggregate:
        from task_runners import aggregate

        return aggregate.report(
            task,
            units,
            variant=args.variant or task.variant,
            latex=args.latex,
        )

    if args.list:
        print(f"\n{task.name}: {task.summary}")
        print(f"reproduces: {task.reproduces}")
        if task.needs:
            print(f"needs: {', '.join(task.needs)}")
        print(f"\n{len(units)} unit(s):")
        for i, unit in enumerate(units):
            stages = ", ".join(s.name for s in unit.stages)
            print(f"  {i:3d}  {unit.name:<28} [{stages}]")
            print(f"       {unit.rundir}")
        return 0

    if args.smoke:
        print("smoke mode: 2 epochs, 2 candidates, 4 targets, one seed")
    elif args.quick:
        print(
            f"quick mode: {ctx.epochs_for('jarvis')} epochs, "
            f"{T.QUICK['num_candidates']} candidates, {len(ctx.seeds)} "
            "seed(s), whole test split -- arms comparable to each other, "
            "not to the published numbers"
        )

    only = _names(args.only_stages)
    skip = _names(args.skip_stages)
    if args.loss_only:
        only = ("train",)

    selected = units if args.unit is None else [units[args.unit]]
    failed = []
    for unit in selected:
        if not run_unit(
            unit,
            dry_run=args.dry_run,
            force=args.force,
            only=only,
            skip=skip,
        ):
            failed.append(unit.name)

    if failed:
        print(f"\n{len(failed)} unit(s) did not finish: {', '.join(failed)}")
        return 1
    if not args.dry_run:
        print(f"\n{len(selected)} unit(s) complete.")
        print(
            f"summarise with: python task_runners/run_task.py "
            f"{task.name} --aggregate"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
