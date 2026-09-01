# Task runners — generative inverse design

One command per result in the manuscript's "Generative inverse design"
section. Each task pins the arguments to the scripts that already live in
`scripts/atombench/` and `alignn/inverse/`, runs them in order, resumes where
it stopped, and prints the table with error bars over seeds.

**Step-by-step procedure: `INSTRUCTIONS.md`.** This file is the reference.

```bash
python task_runners/run_task.py tasks          # what is available
python task_runners/run_task.py doctor         # is this machine ready
python task_runners/run_task.py verify         # paper number -> task -> measured
python task_runners/run_task.py data-jarvis    # build the split
python task_runners/run_task.py bench-jarvis --smoke --device cpu  # minutes
python task_runners/run_task.py bench-jarvis           # the real thing
python task_runners/run_task.py bench-jarvis --aggregate --latex
```

On a cluster, the same tasks go through SLURM as a job array over their units,
plus a dependent job that prints the table when the array succeeds:

```bash
$EDITOR task_runners/cluster.env        # account, partition, conda env, scratch
bash task_runners/submit.sh bench-jarvis --dry-run
bash task_runners/submit.sh bench-jarvis
```

## The tasks

| task | reproduces | units | needs |
|---|---|---|---|
| `data-jarvis` | the JARVIS Supercon-3D split, 847/105/103 | 1 | — |
| `data-alex` | the Alexandria DS-A/B split, 6603/825/825 | 1 | the DS-A/B pickles |
| `data-pretrain` | the 65k dft_3d corpus, benchmark ids held out | 1 | `data-jarvis` |
| `pretrain` | `csp_pretrain_dft3d`, the base model | 1 | `data-pretrain` |
| `bench-jarvis` | Table `tab:inverse_bench`, JARVIS block | 3 seeds | `data-jarvis` |
| `bench-alex` | Table `tab:inverse_bench`, Alexandria block | 1 | `data-alex`, `pretrain` |
| `pretrain-transfer` | Alexandria from scratch vs fine-tuned | 2 | `data-alex`, `pretrain` |
| `ablation-linegraph` | Table `tab:inverse_ablation` | 2 arms × 3 seeds | `data-jarvis` |
| `angle-ablation` | the A0–A6 angular-diffusion suite | 6 arms × 3 seeds | `data-jarvis` |
| `pipeline-ablation` | "Closing the loop with the force field" | 4 | `bench-jarvis` |
| `symprec-sweep` | the symmetrisation tolerance, chosen on val | 1 | `bench-jarvis` |
| `leakage` | the 18.4% / 15.4% recall caveat | 2 | `data-pretrain`, `bench-jarvis` |

Configurations shared between tasks are keyed by their run directory and
therefore trained **once**. `bench-jarvis`, arm A of `ablation-linegraph` and
`A0` of `angle-ablation` are the same model; running all three costs one set
of trainings, and whichever runs second finds the first one's work and skips
straight to what is missing.

## How a task is put together

A task is a list of **units** — independent pieces of work, one per SLURM
array element — and a unit is a list of **stages**:

```
train  ->  generate  ->  symmetrize  ->  score-nosym  ->  score-sym
```

Every finished stage writes `<rundir>/.stages/<stage>.json` recording the
exact command it ran. On a re-run a stage is skipped if that command is
unchanged, and re-runs if it is not — so bumping `--epochs` retrains and
rescores, while re-submitting after a walltime kill picks up where it left
off. `--force` ignores the markers.

Stages also declare their inputs, so a missing prerequisite is reported
immediately:

```
[alex-seed0/train] BLOCKED, missing input(s):
    .../runs/train/pretrain_dft3d/seed0/best_model.pt (produced by pretrain)
```

Everything lands under one root, `--runs-root` (env `ALIGNN_RUNS`, default
`<repo>/runs`, `CSP_RUNS` in `cluster.env`):

```
runs/
├── data/{jarvis,alex,pretrain}/          train.json val.json test.json
├── train/<config>/seed<N>/               best_model.pt history.json config.json
│   └── bench/{nosym,sym}/                pred.csv metrics.json candidates.json
├── pipeline/{raw,rank,relax,full}/
├── symprec/
└── leakage/{jarvis,alex}/
```

Both the unsymmetrised and the symmetrised predictions are scored, because
they answer different questions. Symmetrisation snaps each predicted cell onto
its detected space group, which matters a great deal for the two metrics
measured after Niggli reduction — angle MAE 15.9 → 8.4, KLD 0.030 → 0.018 —
and not at all for match rate. The manuscript's lattice columns are the
symmetrised ones (`--variant sym`, the default); the pipeline ablation is
easier to read before symmetrisation (`--variant nosym`).

## Reading the results

```bash
python task_runners/run_task.py ablation-linegraph --aggregate --latex
```

prints mean ± one standard deviation per arm, every individual run, the
declared comparison with a Welch p-value where scipy is installed, and a LaTeX
tabular in the shape of the manuscript table.

Individual runs are always printed, and the reason is in the manuscript:
across independently trained models the match rate on 103 JARVIS targets
spanned 0.437–0.524 — nine models in the manuscript, fifteen by the time the
inverse-design README was written. A difference of a few percent in match rate
between two single runs is not a result; a difference in coordinate RMSD of
the size reported there is. The comparison output marks a change smaller than
the arms' own spread as *within noise* rather than letting a sign carry the
argument.

## On a cluster

`cluster.env` is the only site-specific file. Fill in what your site needs and
leave the rest empty:

```bash
CSP_ACCOUNT="..."          # sbatch --account
CSP_PARTITION="gpu"        # sbatch --partition
CSP_GPU_GRES="gpu:a100:1"  # sbatch --gres, GPU tasks only
CSP_MAX_CONCURRENT="4"     # array throttle
CSP_ENV="alignn2"          # conda env with torch + this repo
CSP_SCORE_ENV="atombench"  # conda env with pymatgen + average-minimum-distance
CSP_ATOMBENCH_REPO="$HOME/atombench"
CSP_RUNS="/scratch/$USER/alignn_csp"
CSP_MODULES="cuda/12.1"    # module load ...
```

`submit.sh` reads it, sizes the array from the arguments you actually pass
(`--seeds 0,1,2,3,4` submits five elements, not three), submits, and queues
the aggregation job with `--dependency=afterok`. The `#SBATCH` directives
inside `sbatch/<task>.sbatch` are defaults for the default seeds; anything on
the sbatch command line overrides them, which is how `submit.sh` applies
`cluster.env`.

**The walltimes in the sbatch headers are placeholders.** They have not been
measured on any particular machine. Check the first array element and adjust.
The one measured cost figure is relative: the angular channel costs 2.4× per
training step, so `angle-ablation`'s A1–A6 arms need more walltime than A0.

Submit from the repository root — the log paths (`task_runners/logs/`) and
`source task_runners/common.sh` are relative to it. `submit.sh` handles that;
`sbatch` by hand does not.

## Coverage

`claims.py` registers all 48 numbers the inverse-design section prints —
both tables, the force-field paragraph, the leakage fractions, the split
sizes, the parameter match and the 2.4× step cost — against the task that
regenerates each one.

```bash
python task_runners/run_task.py verify
```

Before you have run anything it is a to-do list in dependency order. After
you have, it is a measured-vs-published table: `ok` inside the claim's
tolerance, `~` inside one measured standard deviation, `!!` outside both,
blank for not yet run. It reads only what is on disk and writes nothing. If a
claim ever appears with no task behind it, `verify` exits non-zero and calls
it a bug — that is the check that this directory stays complete.

## Three cost settings

| | epochs | candidates | seeds | targets | run tree |
|---|---|---|---|---|---|
| full (default) | 3000 | 32 | 3 | all | `runs/train/` |
| `--quick` | 300 | 8 | 2 | all | `runs/train_quick/` |
| `--smoke` | 2 | 2 | 1 | 4 | `runs/train_smoke/` |

The separate trees matter: a `--quick` run has a different training command
from a full one, so without them it would be detected as stale work and
overwrite a checkpoint that cost days.

`--quick` keeps everything that makes two arms comparable to each other — the
whole test split, the same pipeline, the same scoring code — and cuts only
the things that scale cost. Its numbers are not comparable to the published
ones, and `verify --quick` says so rather than letting you read across.

Cheaper still, and the right first question for an ablation:

```bash
python task_runners/run_task.py angle-ablation --quick --loss-only
python task_runners/run_task.py angle-ablation --aggregate
```

`--loss-only` runs the training stage and stops. It needs no scoring
environment and no AtomBench clone, and the denoising validation loss is the
most reproducible arm-vs-arm signal there is — the line-graph loss gap
repeated to three decimals on a second machine while the match rate did not
move at all. It also will not tell you whether the right structure is found
more often, which is why it is a filter and not a verdict.

`--only-stages` and `--skip-stages` take the stage names for anything in
between.

## Smoke test

`--smoke` runs every stage of a task at a size that finishes in minutes: two
epochs, two candidates, four targets, one seed. It proves the plumbing —
data layout, checkpoint format, CSV columns, the scoring environment — and
proves nothing at all about the science.

```bash
python task_runners/run_task.py bench-jarvis --smoke --device cpu
```

Smoke runs write into the same tree, so use a throwaway root:
`--runs-root /tmp/csp_smoke`.

## Two epoch counts are not pinned

`EPOCHS["jarvis"] = 3000` is the value the inverse-design README records for
the published JARVIS runs. `EPOCHS["alex"]` and `EPOCHS["pretrain"]` are
plausible choices, not the manuscript's. To replace them with what the
released checkpoints actually used:

```bash
python task_runners/inspect_checkpoint.py csp_supercon_alex
python task_runners/inspect_checkpoint.py csp_pretrain_dft3d --command
```

That reads the argument namespace `train_csp.py` stores in every checkpoint,
so it also works on your own runs (`runs/train/jarvis_A0/seed0`).

## Requirements

Training and generation need torch, jarvis-tools and this repository
installed (`pip install -e .` — re-run it if your editable install predates
`alignn/inverse`). Scoring runs AtomBench's own metric code and additionally
needs pymatgen and `average-minimum-distance`, plus a clone of
[atombench](https://github.com/atomgptlab/atombench) pointed at by
`ATOMBENCH_REPO`. If those live in a separate environment, set
`CSP_SCORE_ENV`; `score.sh` switches into it by itself.

`python task_runners/run_task.py doctor` checks all of it before you queue
anything.

The Alexandria pickles (`DS-A.pk.bz2`, `DS-B.pk.bz2`) are not downloadable
from here — fetch them from figshare DOI `10.6084/m9.figshare.31045597` and
either drop them in `runs/data/alexandria/` or pass `--alex-inputs`.
