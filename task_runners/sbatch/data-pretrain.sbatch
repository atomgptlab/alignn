#!/usr/bin/env bash
#SBATCH --job-name=csp-data-pretrain
#SBATCH --output=task_runners/logs/%x-%A_%a.out
#SBATCH --error=task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Build the 65k dft_3d pretraining corpus, benchmark ids held out.
# reproduces: the corpus behind csp_pretrain_dft3d
#
# 1 array element(s) at the default seeds.
# Needs, in order: data-jarvis.
#
# Account, partition, QoS, constraint and the exact --gres come from
# task_runners/cluster.env, applied by submit.sh, which also resizes the array
# to match the seeds actually requested:
#
#     bash task_runners/submit.sh data-pretrain
#
# Plain `sbatch task_runners/sbatch/data-pretrain.sbatch` works too, from the repo
# root; flags given on the sbatch command line override the directives above.
#
# The walltime above is a placeholder: it has not been measured on
# any particular machine. Check the first element and adjust.
#

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source task_runners/common.sh

# shellcheck disable=SC2086  # CSP_RUN_ARGS is a deliberate word-split
python task_runners/run_task.py data-pretrain \
    --unit "${SLURM_ARRAY_TASK_ID:-0}" \
    ${CSP_RUN_ARGS:-}
