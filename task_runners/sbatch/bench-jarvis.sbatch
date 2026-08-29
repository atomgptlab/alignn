#!/usr/bin/env bash
#SBATCH --job-name=csp-bench-jarvis
#SBATCH --output=task_runners/logs/%x-%A_%a.out
#SBATCH --error=task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Train and benchmark ALIGNN-CSP on JARVIS Supercon-3D, three seeds.
# reproduces: Table 4 (tab:inverse_bench), JARVIS Supercon-3D block
#
# 3 array element(s) at the default seeds.
# Needs, in order: data-jarvis.
#
# Account, partition, QoS, constraint and the exact --gres come from
# task_runners/cluster.env, applied by submit.sh, which also resizes the array
# to match the seeds actually requested:
#
#     bash task_runners/submit.sh bench-jarvis --seeds 0,1,2,3,4
#
# Plain `sbatch task_runners/sbatch/bench-jarvis.sbatch` works too, from the repo
# root; flags given on the sbatch command line override the directives above.
#
# The walltime above is a placeholder: it has not been measured on
# any particular machine. Check the first element and adjust.
#

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source task_runners/common.sh

# shellcheck disable=SC2086  # CSP_RUN_ARGS is a deliberate word-split
python task_runners/run_task.py bench-jarvis \
    --unit "${SLURM_ARRAY_TASK_ID:-0}" \
    ${CSP_RUN_ARGS:-}
