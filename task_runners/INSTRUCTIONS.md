# INSTRUCTIONS — reproducing the paper's inverse-design results

**Audience:** whoever (or whatever) is sitting at a terminal on the machine
that will do the work, with no memory of the conversation that produced this
directory. Everything needed is below.

`README.md` next to this file is the reference — what each task is, how the
runner works, what the output tree looks like. This file is the procedure.

---

## 0. What you are reproducing

The "Generative inverse design" section of the manuscript. Every number it
prints — both tables, the force-field paragraph, the leakage caveat, the split
sizes — is registered in `claims.py` and mapped to a task. To see that map,
and how much of it you have measured so far:

```bash
python task_runners/run_task.py verify
```

Run that first and last. First it tells you which tasks you still owe; last it
tells you whether what you measured agrees with what was published. It never
writes anything.

**Never edit a published number to match a measurement, and never report a
measurement you did not take.** If a claim will not reproduce, say so, say by
how much, and say what you think went wrong. `verify` prints the measured
spread beside the published value precisely so that a disagreement can be
argued about rather than papered over.

---

## 1. Set the machine up

```bash
cd <repo root>                      # the directory containing task_runners/
pip install -e .                    # re-run even if you installed before:
                                    # the editable finder does not see
                                    # alignn/inverse if it predates it
python task_runners/run_task.py doctor
```

`doctor` must show `[x]` on every line before you queue anything. What each
failure means:

| line | fix |
|---|---|
| `torch` | a broken or CPU-only install. The check takes a real optimiser step, so an import that "works" can still fail here. |
| `alignn.inverse` | `pip install -e .` from the repo root |
| `jarvis-tools`, `pymatgen` | `pip install jarvis-tools pymatgen` |
| `average-minimum-distance` | `pip install average-minimum-distance` — needed for ccRMSD only |
| `AtomBench compute_metrics.py` | `git clone https://github.com/atomgptlab/atombench` and `export ATOMBENCH_REPO=<path>` |

Scoring dependencies may live in a separate environment; set `CSP_SCORE_ENV`
in `cluster.env` and `score.sh` will switch into it by itself.

On a cluster, fill in `task_runners/cluster.env` — it is the only
site-specific file. Account, partition, QoS, `--gres`, conda environments,
modules, and where the run tree lives (`CSP_RUNS`, point it at scratch).

---

## 2. Prove the plumbing before spending a GPU-week

```bash
python task_runners/run_task.py bench-jarvis --smoke --device cpu \
    --runs-root /tmp/csp_smoke
```

Two epochs, two candidates, four targets. It exercises every stage — training,
sampling, relaxation, symmetrisation, scoring — and so catches a missing
dependency, a broken checkpoint format or an unreachable AtomBench clone in
minutes rather than after a queue wait. The numbers it produces are
meaningless and it writes into its own `train_smoke/` tree, so it cannot
touch a real run.

---

## 3. Run the tasks, in this order

Each line is safe to re-run: finished stages are skipped, and a stage whose
command changed re-runs. Everything after `data-jarvis` can also be submitted
with `bash task_runners/submit.sh <task>`.

```bash
# 1. data (CPU. Minutes, plus a one-off ~40 MB JARVIS download)
python task_runners/run_task.py data-jarvis
python task_runners/run_task.py data-alex          # needs the DS-A/B pickles
python task_runners/run_task.py data-pretrain

# 2. the base model (GPU, long)
python task_runners/run_task.py pretrain

# 3. the two benchmark tables
python task_runners/run_task.py bench-jarvis       # 3 seeds -> Table 4 JARVIS
python task_runners/run_task.py bench-alex         # -> Table 4 Alexandria

# 4. the line-graph ablation (Table 3). Arm A is bench-jarvis, already done
python task_runners/run_task.py ablation-linegraph

# 5. the paragraphs
python task_runners/run_task.py pipeline-ablation  # force-field loop
python task_runners/run_task.py leakage            # the 18.4% / 15.4% caveat

# 6. optional: the tolerance used before scoring, chosen on validation
python task_runners/run_task.py symprec-sweep

# 7. finally
python task_runners/run_task.py verify
```

`data-alex` needs `DS-A.pk.bz2` and `DS-B.pk.bz2` from figshare DOI
`10.6084/m9.figshare.31045597`. Put them in `<runs>/data/alexandria/` or pass
`--alex-inputs A.pk.bz2 B.pk.bz2`. Nothing else needs a manual download.

### Reading each table

```bash
python task_runners/run_task.py ablation-linegraph --aggregate --latex
python task_runners/run_task.py bench-jarvis       --aggregate --latex
python task_runners/run_task.py pipeline-ablation  --aggregate --variant nosym
```

`--aggregate` prints mean ± one standard deviation per arm, every individual
run, parameter counts and training wall time, the declared comparison with a
Welch p-value, and the published baselines where they apply. `--latex` adds a
tabular in the shape of the manuscript's.

---

## 4. Benchmarking an ablation quickly

This is the common case: you have changed something in the denoiser and want
to know, today, whether it is worth a full run.

**Fastest useful signal — denoising loss only.** No sampling, no relaxation,
no scoring environment, no AtomBench clone:

```bash
python task_runners/run_task.py angle-ablation --quick --loss-only
python task_runners/run_task.py angle-ablation --aggregate
```

Twelve 300-epoch trainings (six arms × two seeds). The comparison block prints
each pair from `alignn.inverse.ablations.COMPARISONS` with the change, a
p-value and a *within noise* marker, plus the training cost of each arm.

The loss is the right first metric here, and the reason is in the
inverse-design README: the line-graph loss gap reproduced to three decimals
across two machines while the downstream match rate did not move at all. Loss
tells you how precisely atoms are placed; it does not tell you how often the
right structure is found. So:

**Then the full pipeline, still quick**, when the loss says something moved:

```bash
python task_runners/run_task.py angle-ablation --quick
python task_runners/run_task.py angle-ablation --aggregate --latex
```

300 epochs, 8 candidates, 2 seeds, but the **whole** test split and the same
scoring code, so the arms are comparable to each other. They are not
comparable to the published numbers — `verify --quick` will say so.

**Only then the real thing**, on the two arms that survived:

```bash
python task_runners/run_task.py angle-ablation --seeds 0,1,2
```

Some knobs, when you want a variant that is not in the table:

```bash
--only-stages train,generate      # or --skip-stages score-nosym
--seeds 0,1,2,3,4                 # more seeds; a few percent needs them
--epochs 800                      # anything explicit overrides --quick
--num-candidates 16
```

A one-off configuration does not need a new entry in `ablations.py`: pass the
denoiser switches straight through, e.g.
`--only-stages train --epochs 300` with a new `--ablation` name added to
`alignn/inverse/ablations.py` if you want it to be a named arm.

### Cost, roughly

Relative, since absolute times depend on the machine. One JARVIS training run
at 3000 epochs is the unit.

| | cost |
|---|---|
| `--smoke` | negligible, minutes |
| `--quick --loss-only`, one arm one seed | ~0.1 |
| `--quick`, one arm one seed | ~0.1 + generation |
| full, one arm one seed | 1 |
| an arm with the angular channel on | ×2.4 per step |
| generation, 32 candidates on 103 targets | dominated by relaxation; use `--relax-workers` |

`angle-ablation` at full size is 18 runs. Decide with `--quick` first.

---

## 5. When something disagrees

Work through these before concluding the model changed:

1. **Is it the seed?** Across independently trained models the JARVIS match
   rate spanned 0.437–0.524 — nine structures out of 103. `--aggregate` prints every
   individual run; look at the spread before believing a mean.
2. **Is it symmetrisation?** The lattice-angle and KLD columns are measured
   after Niggli reduction and move a lot with the tolerance (angle MAE
   15.9 → 8.4 at `--symprec 0.1`). Compare `--variant sym` against
   `--variant nosym`, and pick the tolerance with `symprec-sweep` on
   *validation*, never on test.
3. **Is it the pipeline, not the model?** `pipeline-ablation` separates what
   the generator contributes from what candidate selection and the force
   field contribute. One sample scores 0.22; the full pipeline scores 0.52.
4. **Is it the epochs?** `EPOCHS["alex"]` and `EPOCHS["pretrain"]` in
   `tasks.py` are *not* pinned by the manuscript. Recover the published
   values with
   `python task_runners/inspect_checkpoint.py csp_supercon_alex` and set them
   before blaming anything else.
5. **Is it leakage?** If you fine-tuned from `csp_pretrain_dft3d`, some test
   targets are reachable by recall. Run `leakage` and compare the filtered
   score.

Two known inconsistencies in the source material, so you are not surprised by
them:

- The Table 4 "best single run" row pairs match 0.524 with RMSD 0.023. In the
  released model registry those belong to two different checkpoints
  (`csp_supercon_jarvis`: 0.524 / 0.056, `csp_supercon_jarvis_pt`: 0.515 /
  0.023). `--aggregate` names which run it is quoting, so you will see this.
- The leakage paragraph says the quoted ALIGNN-CSP results use no
  pretraining, but the released `csp_supercon_alex` behind the 0.485
  Alexandria row was fine-tuned from `csp_pretrain_dft3d`. `bench-alex`
  defaults to the pretrained arm; `--from-scratch` trains the other one, and
  `pretrain-transfer` runs both.

---

## 6. What to hand back

- the `verify` table, unedited;
- `--aggregate --latex` output for each table you regenerated;
- a note of any claim marked `!!`, with your reading of why;
- the run tree, or at least each run's `config.json`, `history.json`,
  `.stages/*.json` (which record the exact command, host and git revision) and
  `bench/*/metrics.json`.
