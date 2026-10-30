"""Declarative definitions of the manuscript's inverse-design tasks.

Each entry in :data:`TASKS` reproduces one claim in the "Generative inverse
design" section.  A task is a list of *units* — independent pieces of work
that can run as SLURM array elements — and each unit is a list of *stages*,
which are ordinary shell commands against the scripts already in this
repository.  Nothing here reimplements the science; it pins the arguments.

    task                  reproduces
    ------------------    --------------------------------------------------
    data-jarvis           the JARVIS Supercon-3D split (847/105/103)
    data-alex             the Alexandria DS-A/B split (6603/825/825)
    data-pretrain         the 65k dft_3d pretraining corpus
    pretrain              the csp_pretrain_dft3d base model
    bench-jarvis          Table 4, JARVIS block (3 seeds)
    bench-alex            Table 4, Alexandria block (single run)
    pretrain-transfer     Alexandria from scratch vs fine-tuned
    ablation-linegraph    Table 3, line graph vs pair-graph depth
    angle-ablation        the A0-A6 angular-diffusion suite (this branch)
    pipeline-ablation     "Closing the loop with the force field"
    symprec-sweep         the symmetrisation tolerance, chosen on validation
    leakage               the 18.4% / 15.4% recall-not-generation caveat

Units are keyed by their run directory, so configurations shared between
tasks are trained once.  ``bench-jarvis`` and arm A of ``ablation-linegraph``
and ``A0`` of ``angle-ablation`` are the same six-layer baseline and land in
the same ``train/jarvis_A0/seed*`` directories; running all three costs one
set of trainings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from alignn.inverse.ablations import ABLATIONS, COMPARISONS, DESCRIPTIONS

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Hyperparameters.
#
# The JARVIS numbers are the ones the inverse-design README records as having
# produced the published results; the Alexandria and pretraining epoch counts
# are *not* pinned by the manuscript, so they are marked and can be recovered
# exactly from a released checkpoint with ``inspect_checkpoint.py``.
# ---------------------------------------------------------------------------

EPOCHS = {
    "jarvis": 3000,
    "alex": 1000,  # not pinned by the manuscript
    "pretrain": 200,  # not pinned by the manuscript
}

#: Generation settings for the benchmark tables.  32 candidates with a
#: single-point energy prescreen down to 4 relaxations is the configuration
#: the README credits with match 0.524 on JARVIS.
GEN = {
    "num_candidates": 32,
    "prescreen_keep": 4,
    "relax": "cell",
    "rank": "energy",
    "relax_steps": 200,
}

#: ``--quick``: a real measurement at a fraction of the cost, for deciding
#: whether an ablation is worth a full run.  Everything that changes the
#: *comparison* is untouched -- both arms still see the whole test split, the
#: same candidate pool policy and the same scoring -- so the arms stay
#: comparable to each other.  They are not comparable to the published
#: numbers, which is what ``verify`` will tell you if you try.
QUICK = {
    "epochs": 300,
    "num_candidates": 8,
    "prescreen_keep": 2,
    "sample_steps": 200,
    "relax_steps": 50,
    "seeds": 2,
}

#: Tolerances swept on the validation split by ``symprec-sweep``.
SYMPREC_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

#: The four stages of the generate -> select -> relax pipeline, isolated.
PIPELINE_VARIANTS = {
    "raw": {"num_candidates": 1, "relax": "none", "rank": "none"},
    "rank": {"num_candidates": 32, "relax": "none", "rank": "energy"},
    "relax": {"num_candidates": 1, "relax": "cell", "rank": "energy"},
    "full": {
        "num_candidates": 32,
        "relax": "cell",
        "rank": "energy",
        "prescreen_keep": 4,
    },
}


@dataclass(frozen=True)
class Stage:
    """One command, plus the marker that says it already succeeded."""

    name: str
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    #: A path that must exist before the stage can run, with the task that
    #: makes it.  Checked up front so a missing prerequisite is reported
    #: rather than crashed into an hour later.
    requires: Sequence[tuple] = ()


@dataclass(frozen=True)
class Unit:
    """One array element: a run directory and the stages that fill it."""

    name: str
    rundir: Path
    stages: List[Stage]
    #: Aggregation label.  Units sharing a group are averaged over seeds.
    group: str = ""
    seed: Optional[int] = None
    #: Where this unit's scored metrics.json files live, by variant.
    metrics: Dict[str, Path] = field(default_factory=dict)
    #: history.json, for the denoising validation loss.
    history: Optional[Path] = None


@dataclass(frozen=True)
class Task:
    """A named group of units, plus how to summarise them."""

    name: str
    summary: str
    reproduces: str
    build: Callable[["Ctx"], List[Unit]]
    needs: Sequence[str] = ()
    #: AtomBench baseline block to print alongside, if any.
    baselines: Optional[str] = None
    #: Pairs of groups whose difference is the point of the task.
    comparisons: Dict[str, tuple] = field(default_factory=dict)
    #: Short group label -> what it means, printed under the table.
    legend: Dict[str, str] = field(default_factory=dict)
    #: Which scored CSV --aggregate reads by default.  The lattice columns
    #: are measured after symmetrisation; the pipeline paragraph is quoted
    #: before it.
    variant: str = "sym"
    #: Whether units want a GPU (drives the sbatch header we ship).
    gpu: bool = True
    default_seeds: Sequence[int] = (0, 1, 2)


@dataclass
class Ctx:
    """Runtime knobs shared by every task."""

    runs: Path
    seeds: Sequence[int] = (0, 1, 2)
    epochs: Optional[int] = None
    device: str = "cuda"
    num_candidates: Optional[int] = None
    guidance: float = 2.0
    symprec: float = 0.1
    relax_workers: Optional[int] = None
    relax_steps: Optional[int] = None
    limit: Optional[int] = None
    checkpoint: Optional[str] = None
    alex_inputs: Sequence[str] = ()
    smoke: bool = False
    quick: bool = False
    from_scratch: bool = False

    # -- derived paths ------------------------------------------------------
    @property
    def data(self) -> Path:
        # Splits are identical in every mode, so they are shared: a quick run
        # should not rebuild the data, only the models.
        return self.runs / "data"

    @property
    def suffix(self) -> str:
        """Keeps reduced-cost runs out of the full runs' directories.

        Without this a ``--quick`` run would land on the same checkpoint path
        as the real one, and because the training command differs it would
        overwrite it -- days of GPU time destroyed by a 20-minute sanity
        check.
        """
        if self.smoke:
            return "_smoke"
        if self.quick:
            return "_quick"
        return ""

    def out(self, *parts: str) -> Path:
        """A top-level output directory for this mode."""
        head, *rest = parts
        return self.runs.joinpath(f"{head}{self.suffix}", *rest)

    def train_dir(self, config: str, seed: int) -> Path:
        return self.out("train", config, f"seed{seed}")

    def epochs_for(self, key: str) -> int:
        """--epochs wins, then --smoke, then --quick, then the table."""
        if self.epochs is not None:
            return self.epochs
        if self.smoke:
            return 2
        if self.quick:
            return QUICK["epochs"]
        return EPOCHS[key]

    def candidates(self, default: int) -> int:
        if self.num_candidates:
            return self.num_candidates
        if self.smoke:
            return 2
        if self.quick:
            # Never sample *more* than the variant asks for: the pipeline
            # ablation's one-sample arms must stay at one sample.
            return min(default, QUICK["num_candidates"])
        return default


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------


def _py(script: str, *args: str) -> List[str]:
    return [
        "python",
        "-u",
        str(REPO / "scripts" / "atombench" / script),
        *args,
    ]


def train_stage(
    ctx: Ctx,
    rundir: Path,
    data_dir: Path,
    *,
    seed: int,
    epochs_key: str,
    alignn_layers: int = 3,
    gcn_layers: int = 3,
    ablation: str = "A0",
    augment: int = 0,
    init_from: Optional[Path] = None,
) -> Stage:
    """A single ``alignn.inverse.train_csp`` run.

    Everything not under test is fixed here: hidden size, kNN, the diffusion
    schedule, batch size and learning rate.  A comparison is only meaningful
    if the arms differ in the switch being studied and nothing else.
    """
    argv = [
        "python",
        "-u",
        "-m",
        "alignn.inverse.train_csp",
        "--data-dir",
        str(data_dir),
        "--output",
        str(rundir),
        "--epochs",
        str(ctx.epochs_for(epochs_key)),
        "--seed",
        str(seed),
        "--ablation",
        ablation,
        "--alignn-layers",
        str(alignn_layers),
        "--gcn-layers",
        str(gcn_layers),
        "--hidden-features",
        "256",
        "--knn",
        "12",
        "--num-steps",
        "1000",
        "--batch-size",
        "64",
        "--lr",
        "1e-3",
        "--augment",
        str(augment),
        "--device",
        ctx.device,
        "--log-every",
        "25",
    ]
    if init_from is not None:
        argv += ["--init-from", str(init_from)]
    return Stage(
        "train",
        argv,
        requires=(((data_dir / "train.json"), f"data ({data_dir.name})"),)
        + (((init_from, "pretrain"),) if init_from is not None else ()),
    )


def generate_stage(
    ctx: Ctx,
    checkpoint: Path,
    data_dir: Path,
    out_csv: Path,
    *,
    seed: int = 0,
    split: str = "test",
    overrides: Optional[Dict] = None,
) -> Stage:
    """Sample, optionally rank and relax, and write an AtomBench CSV."""
    cfg = dict(GEN)
    cfg.update(overrides or {})
    cfg["num_candidates"] = ctx.candidates(cfg["num_candidates"])
    if ctx.relax_steps is not None:
        cfg["relax_steps"] = ctx.relax_steps
    if ctx.smoke:
        # The models are trained at 1000 denoising steps and sampled at 1000;
        # a smoke run only needs the code path, so it takes the cheap end of
        # that trade and says so rather than passing off the result.
        cfg["relax_steps"] = 20
    elif ctx.quick and ctx.relax_steps is None:
        cfg["relax_steps"] = QUICK["relax_steps"]
        if cfg.get("prescreen_keep"):
            cfg["prescreen_keep"] = QUICK["prescreen_keep"]
    argv = _py(
        "generate_benchmark.py",
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--split",
        split,
        "--output-csv",
        str(out_csv),
        "--num-candidates",
        str(cfg["num_candidates"]),
        "--guidance",
        str(ctx.guidance),
        "--relax",
        cfg["relax"],
        "--rank",
        cfg["rank"],
        "--relax-steps",
        str(cfg["relax_steps"]),
        "--seed",
        str(seed),
        "--device",
        ctx.device,
        "--save-candidates",
        str(out_csv.with_name("candidates.json")),
    )
    keep = cfg.get("prescreen_keep")
    if keep and cfg["relax"] != "none":
        # Never prescreen to more candidates than were sampled.
        argv += ["--prescreen-keep", str(min(keep, cfg["num_candidates"]))]
    if ctx.smoke:
        argv += ["--steps", "50"]
    elif ctx.quick:
        argv += ["--steps", str(QUICK["sample_steps"])]
    if ctx.relax_workers is not None:
        argv += ["--relax-workers", str(ctx.relax_workers)]
    limit = 4 if ctx.smoke else ctx.limit
    if limit is not None:
        argv += ["--limit", str(limit)]
    return Stage(
        "generate",
        argv,
        # Relaxation forks CPU workers; leaving BLAS threaded oversubscribes
        # the node badly, which is what run_ablation.sh guards against too.
        env={"OMP_NUM_THREADS": "1"},
        requires=((checkpoint, "the training stage"),),
    )


def symmetrize_stage(csv_in: Path, csv_out: Path, symprec: float) -> Stage:
    """Idealise each predicted cell to its detected space group."""
    return Stage(
        "symmetrize",
        _py(
            "symmetrize_predictions.py",
            "--csv",
            str(csv_in),
            "--out",
            str(csv_out),
            "--symprec",
            str(symprec),
        ),
        requires=((csv_in, "the generate stage"),),
    )


def score_stage(csv: Path, name: str = "score") -> Stage:
    """Run AtomBench's own metric code; writes metrics.json beside the CSV."""
    return Stage(
        name,
        ["bash", str(REPO / "scripts" / "atombench" / "score.sh"), str(csv)],
        requires=((csv, "the stage that writes this CSV"),),
    )


def eval_stages(
    ctx: Ctx,
    rundir: Path,
    checkpoint: Path,
    data_dir: Path,
    *,
    seed: int = 0,
    split: str = "test",
    overrides: Optional[Dict] = None,
) -> tuple:
    """Generate, symmetrise and score, unsymmetrised and symmetrised.

    Both are kept because they answer different questions: the pipeline
    ablation is about what sampling and the force field contribute, which is
    visible before symmetrisation, while the manuscript's lattice-angle and
    KLD columns are measured after it.
    """
    nosym = rundir / "bench" / "nosym" / "pred.csv"
    sym = rundir / "bench" / "sym" / "pred.csv"
    stages = [
        generate_stage(
            ctx,
            checkpoint,
            data_dir,
            nosym,
            seed=seed,
            split=split,
            overrides=overrides,
        ),
        symmetrize_stage(nosym, sym, ctx.symprec),
        score_stage(nosym, "score-nosym"),
        score_stage(sym, "score-sym"),
    ]
    metrics = {
        "nosym": nosym.with_name("metrics.json"),
        "sym": sym.with_name("metrics.json"),
    }
    return stages, metrics


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _data_jarvis(ctx: Ctx) -> List[Unit]:
    out = ctx.data / "jarvis"
    return [
        Unit(
            "jarvis",
            out,
            [Stage("prepare", _py("prepare_data.py", "--output", str(out)))],
            group="data",
        )
    ]


def _data_alex(ctx: Ctx) -> List[Unit]:
    out = ctx.data / "alex"
    inputs = list(ctx.alex_inputs) or [
        str(ctx.data / "alexandria" / "DS-A.pk.bz2"),
        str(ctx.data / "alexandria" / "DS-B.pk.bz2"),
    ]
    return [
        Unit(
            "alex",
            out,
            [
                Stage(
                    "prepare",
                    _py(
                        "prepare_alex_data.py",
                        "--inputs",
                        *inputs,
                        "--output",
                        str(out),
                    ),
                    requires=tuple(
                        (
                            Path(p),
                            "the Alexandria DS-A/DS-B pickles "
                            "(figshare 10.6084/m9.figshare.31045597); "
                            "pass --alex-inputs to point elsewhere",
                        )
                        for p in inputs
                    ),
                )
            ],
            group="data",
        )
    ]


def _data_pretrain(ctx: Ctx) -> List[Unit]:
    out = ctx.data / "pretrain"
    return [
        Unit(
            "pretrain",
            out,
            [
                Stage(
                    "prepare",
                    _py(
                        "prepare_pretrain_data.py",
                        "--output",
                        str(out),
                        "--exclude-splits",
                        str(ctx.data / "jarvis"),
                    ),
                    requires=(
                        (ctx.data / "jarvis" / "test.json", "data-jarvis"),
                    ),
                )
            ],
            group="data",
        )
    ]


def _pretrain(ctx: Ctx) -> List[Unit]:
    rundir = ctx.train_dir("pretrain_dft3d", 0)
    return [
        Unit(
            "pretrain_dft3d",
            rundir,
            [
                train_stage(
                    ctx,
                    rundir,
                    ctx.data / "pretrain",
                    seed=0,
                    epochs_key="pretrain",
                    augment=1,
                )
            ],
            group="pretrain",
            seed=0,
            history=rundir / "history.json",
        )
    ]


def _jarvis_units(
    ctx: Ctx,
    config: str,
    group: str,
    *,
    alignn_layers: int = 3,
    gcn_layers: int = 3,
    ablation: str = "A0",
) -> List[Unit]:
    """Train-and-evaluate units on the JARVIS split, one per seed."""
    data = ctx.data / "jarvis"
    units = []
    for seed in ctx.seeds:
        rundir = ctx.train_dir(config, seed)
        stages = [
            train_stage(
                ctx,
                rundir,
                data,
                seed=seed,
                epochs_key="jarvis",
                alignn_layers=alignn_layers,
                gcn_layers=gcn_layers,
                ablation=ablation,
                # 48 basis relabellings of 847 crystals cost accuracy on a
                # split this small; the README records augmentation off as
                # the best small-data setting.
                augment=0,
            )
        ]
        ev, metrics = eval_stages(
            ctx, rundir, rundir / "best_model.pt", data, seed=seed
        )
        units.append(
            Unit(
                f"{config}-seed{seed}",
                rundir,
                stages + ev,
                group=group,
                seed=seed,
                metrics=metrics,
                history=rundir / "history.json",
            )
        )
    return units


def _bench_jarvis(ctx: Ctx) -> List[Unit]:
    return _jarvis_units(ctx, "jarvis_A0", "ALIGNN-CSP")


def _alex_units(
    ctx: Ctx, config: str, group: str, *, pretrained: bool
) -> List[Unit]:
    """Train-and-evaluate units on Alexandria DS-A/B, one per seed."""
    data = ctx.data / "alex"
    init = (
        ctx.train_dir("pretrain_dft3d", 0) / "best_model.pt"
        if pretrained
        else None
    )
    units = []
    for seed in ctx.seeds:
        rundir = ctx.train_dir(config, seed)
        stages = [
            train_stage(
                ctx,
                rundir,
                data,
                seed=seed,
                epochs_key="alex",
                # 6603 training crystals, so the 48 basis relabellings are
                # affordable here in a way they are not on the 847-crystal
                # JARVIS split.
                augment=1,
                init_from=init,
            )
        ]
        ev, metrics = eval_stages(
            ctx, rundir, rundir / "best_model.pt", data, seed=seed
        )
        units.append(
            Unit(
                f"{config}-seed{seed}",
                rundir,
                stages + ev,
                group=group,
                seed=seed,
                metrics=metrics,
                history=rundir / "history.json",
            )
        )
    return units


def _bench_alex(ctx: Ctx) -> List[Unit]:
    # The released csp_supercon_alex, whose 0.485 match rate is the Table 4
    # Alexandria row, was fine-tuned from csp_pretrain_dft3d -- so that is the
    # default here.  The leakage paragraph says the quoted ALIGNN-CSP results
    # use no pretraining, which does not fit that row; --from-scratch trains
    # the arm that sentence describes, and pretrain-transfer runs both.
    if ctx.from_scratch:
        return _alex_units(ctx, "alex_scratch", "ALIGNN-CSP", pretrained=False)
    return _alex_units(ctx, "alex", "ALIGNN-CSP", pretrained=True)


def _pretrain_transfer(ctx: Ctx) -> List[Unit]:
    """Both Alexandria arms, so the pretraining claim is a measurement."""
    return _alex_units(
        ctx, "alex_scratch", "from scratch", pretrained=False
    ) + _alex_units(ctx, "alex", "pretrained", pretrained=True)


def _ablation_linegraph(ctx: Ctx) -> List[Unit]:
    # Arm B spends the deleted angular budget on pair-graph depth: nine
    # convolution blocks in both arms, parameters matched to within 1%.
    return _jarvis_units(ctx, "jarvis_A0", "A: line graph") + _jarvis_units(
        ctx,
        "jarvis_nolg",
        "B: no line graph",
        alignn_layers=0,
        gcn_layers=9,
    )


def _angle_ablation(ctx: Ctx) -> List[Unit]:
    units: List[Unit] = []
    for name in sorted(ABLATIONS):
        units += _jarvis_units(ctx, f"jarvis_{name}", name, ablation=name)
    return units


def _default_checkpoint(ctx: Ctx) -> Path:
    if ctx.checkpoint:
        return Path(ctx.checkpoint)
    return ctx.train_dir("jarvis_A0", ctx.seeds[0]) / "best_model.pt"


def _pipeline_ablation(ctx: Ctx) -> List[Unit]:
    ckpt = _default_checkpoint(ctx)
    data = ctx.data / "jarvis"
    units = []
    for variant, overrides in PIPELINE_VARIANTS.items():
        rundir = ctx.out("pipeline", variant)
        ev, metrics = eval_stages(
            ctx, rundir, ckpt, data, seed=ctx.seeds[0], overrides=overrides
        )
        units.append(
            Unit(
                variant,
                rundir,
                ev,
                group=variant,
                metrics=metrics,
            )
        )
    return units


def _symprec_sweep(ctx: Ctx) -> List[Unit]:
    """Choose the symmetrisation tolerance on validation, never on test."""
    ckpt = _default_checkpoint(ctx)
    data = ctx.data / "jarvis"
    root = ctx.out("symprec")
    val_csv = root / "val" / "pred.csv"
    stages = [
        generate_stage(
            ctx, ckpt, data, val_csv, seed=ctx.seeds[0], split="val"
        ),
        Stage(
            "sweep",
            _py(
                "symmetrize_predictions.py",
                "--csv",
                str(val_csv),
                "--out",
                str(root),
                "--sweep",
                ",".join(f"{s:g}" for s in SYMPREC_GRID),
            ),
            requires=((val_csv, "the generate stage"),),
        ),
        score_stage(val_csv, "score-nosym"),
    ]
    metrics = {"none": val_csv.with_name("metrics.json")}
    for sp in SYMPREC_GRID:
        tag = f"symprec{sp:g}".replace(".", "p")
        csv = root / tag / f"pred_{tag}.csv"
        stages.append(score_stage(csv, f"score-{tag}"))
        metrics[f"{sp:g}"] = csv.with_name("metrics.json")
    return [Unit("sweep", root, stages, group="symprec", metrics=metrics)]


def _leakage(ctx: Ctx) -> List[Unit]:
    """Quantify how many test targets a pretrained model could recall."""
    units = []
    sources = {
        "jarvis": (
            ctx.data / "jarvis",
            ctx.train_dir("jarvis_A0", ctx.seeds[0]),
        ),
        "alex": (ctx.data / "alex", ctx.train_dir("alex", ctx.seeds[0])),
    }
    for name, (data, train) in sources.items():
        root = ctx.out("leakage", name)
        report = root / "leakage.json"
        pred = train / "bench" / "sym" / "pred.csv"
        filtered = root / "filtered" / "pred.csv"
        stages = [
            Stage(
                "check",
                _py(
                    "check_pretrain_leakage.py",
                    "--pretrain-dir",
                    str(ctx.data / "pretrain"),
                    "--test-json",
                    str(data / "test.json"),
                    "--output",
                    str(report),
                ),
                requires=(
                    (ctx.data / "pretrain" / "train.json", "data-pretrain"),
                    (data / "test.json", f"data-{name}"),
                ),
            ),
            Stage(
                "filter",
                _py(
                    "filter_leaked.py",
                    "--csv",
                    str(pred),
                    "--leakage-json",
                    str(report),
                    "--out",
                    str(filtered),
                ),
                requires=((pred, f"bench-{name}"),),
            ),
            score_stage(filtered, "score-filtered"),
        ]
        units.append(
            Unit(
                name,
                root,
                stages,
                group=name,
                metrics={
                    "all": pred.with_name("metrics.json"),
                    "not-leaked": filtered.with_name("metrics.json"),
                },
            )
        )
    return units


TASKS: Dict[str, Task] = {
    "data-jarvis": Task(
        "data-jarvis",
        "Build the AtomBench JARVIS Supercon-3D split (847/105/103).",
        "the split every JARVIS number in the paper is measured on",
        _data_jarvis,
        gpu=False,
        default_seeds=(0,),
    ),
    "data-alex": Task(
        "data-alex",
        "Build the AtomBench Alexandria DS-A/B split (6603/825/825).",
        "the split the Alexandria block of Table 4 is measured on",
        _data_alex,
        gpu=False,
        default_seeds=(0,),
    ),
    "data-pretrain": Task(
        "data-pretrain",
        "Build the 65k dft_3d pretraining corpus, benchmark ids held out.",
        "the corpus behind csp_pretrain_dft3d",
        _data_pretrain,
        needs=("data-jarvis",),
        gpu=False,
        default_seeds=(0,),
    ),
    "pretrain": Task(
        "pretrain",
        "Train the composition-only base model on 65k dft_3d crystals.",
        "csp_pretrain_dft3d, the checkpoint bench-alex fine-tunes from",
        _pretrain,
        needs=("data-pretrain",),
        default_seeds=(0,),
    ),
    "bench-jarvis": Task(
        "bench-jarvis",
        "Train and benchmark ALIGNN-CSP on JARVIS Supercon-3D, three seeds.",
        "Table 4 (tab:inverse_bench), JARVIS Supercon-3D block",
        _bench_jarvis,
        needs=("data-jarvis",),
        baselines="jarvis",
    ),
    "bench-alex": Task(
        "bench-alex",
        "Fine-tune from the base model and benchmark on Alexandria DS-A/B.",
        "Table 4 (tab:inverse_bench), Alexandria DS-A/B block",
        _bench_alex,
        needs=("data-alex", "pretrain"),
        baselines="alex",
        default_seeds=(0,),
    ),
    "pretrain-transfer": Task(
        "pretrain-transfer",
        "Alexandria from scratch vs fine-tuned from the dft_3d base model.",
        "the pretraining claim behind the Table 4 Alexandria row",
        _pretrain_transfer,
        needs=("data-alex", "pretrain"),
        comparisons={
            "does pretraining help on the larger split": (
                "from scratch",
                "pretrained",
            )
        },
        legend={
            "from scratch": "trained on Alexandria alone",
            "pretrained": "fine-tuned from csp_pretrain_dft3d (65k dft_3d)",
        },
        default_seeds=(0,),
    ),
    "ablation-linegraph": Task(
        "ablation-linegraph",
        "Line graph vs the same budget spent on pair-graph depth.",
        "Table 3 (tab:inverse_ablation)",
        _ablation_linegraph,
        needs=("data-jarvis",),
        comparisons={
            "does the line graph transfer to generation": (
                "B: no line graph",
                "A: line graph",
            )
        },
        legend={
            "A: line graph": "three ALIGNN layers, three pair-graph "
            "convolutions (3.79 M parameters)",
            "B: no line graph": "no angular channel, nine pair-graph "
            "convolutions (3.75 M parameters)",
        },
    ),
    "angle-ablation": Task(
        "angle-ablation",
        "The A0-A6 angular-diffusion suite from alignn.inverse.ablations.",
        "the explicit bond-angle denoising extension (this branch)",
        _angle_ablation,
        needs=("data-jarvis",),
        comparisons=dict(COMPARISONS),
        legend=dict(DESCRIPTIONS),
    ),
    "pipeline-ablation": Task(
        "pipeline-ablation",
        "raw / rank / relax / full: what sampling and the force field buy.",
        "'Closing the loop with the force field'",
        _pipeline_ablation,
        needs=("bench-jarvis",),
        comparisons={
            "what one sample plus the full pipeline is worth": (
                "raw",
                "full",
            ),
            "selection or refinement": ("relax", "rank"),
        },
        variant="nosym",
        legend={
            "raw": "one sample per target, straight from the diffusion model",
            "rank": "32 samples, lowest ALIGNN-FF energy, no relaxation",
            "relax": "one sample, relaxed with ALIGNN-FF",
            "full": "32 samples, energy prescreen, top 4 relaxed",
        },
        default_seeds=(0,),
    ),
    "symprec-sweep": Task(
        "symprec-sweep",
        "Pick the symmetrisation tolerance on the validation split.",
        "the symmetrisation step used before scoring the test split",
        _symprec_sweep,
        needs=("bench-jarvis",),
        default_seeds=(0,),
    ),
    "leakage": Task(
        "leakage",
        "Test targets recoverable from JARVIS-DFT by recall, and the score "
        "on the complement.",
        "the 18.4% / 15.4% leakage caveat",
        _leakage,
        needs=("data-pretrain", "bench-jarvis"),
        gpu=False,
        default_seeds=(0,),
    ),
}


def get(name: str) -> Task:
    """Look a task up, with the available names in the error."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {', '.join(TASKS)}")
    return TASKS[name]
