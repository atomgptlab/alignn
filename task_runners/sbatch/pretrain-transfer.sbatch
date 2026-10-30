#!/usr/bin/env bash
#SBATCH --job-name=csp-pretrain-transfer
#SBATCH --output=task_runners/logs/%x-%A_%a.out
#SBATCH --error=task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#
# Alexandria from scratch vs fine-tuned from the dft_3d base model.
# reproduces: the pretraining claim behind the Table 4 Alexandria row
#
# 2 array element(s) at the default seeds.
# Needs, in order: data-alex -> pretrain.
#
# Account, partition, QoS, constraint and the exact --gres come from
# task_runners/cluster.env, applied by submit.sh, which also resizes the array
# to match the seeds actually requested:
#
#     bash task_runners/submit.sh pretrain-transfer
#
# Plain `sbatch task_runners/sbatch/pretrain-transfer.sbatch` works too, from the repo
# root; flags given on the sbatch command line override the directives above.
#
# The walltime above is a placeholder: it has not been measured on
# any particular machine. Check the first element and adjust.
#

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source task_runners/common.sh

# shellcheck disable=SC2086  # CSP_RUN_ARGS is a deliberate word-split
python task_runners/run_task.py pretrain-transfer \
    --unit "${SLURM_ARRAY_TASK_ID:-0}" \
    ${CSP_RUN_ARGS:-}
