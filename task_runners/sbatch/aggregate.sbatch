#!/usr/bin/env bash
#SBATCH --job-name=csp-aggregate
#SBATCH --output=task_runners/logs/%x-%j.out
#SBATCH --error=task_runners/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Print a task's table once its array has finished. submit.sh queues this
# with --dependency=afterok on the array job, so it only runs if every unit
# succeeded; run it by hand at any time to see partial results.
#
#     CSP_TASK=bench-jarvis sbatch task_runners/sbatch/aggregate.sbatch
#
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source task_runners/common.sh

TASK="${CSP_TASK:?set CSP_TASK to the task to summarise}"

# shellcheck disable=SC2086
python task_runners/run_task.py "$TASK" --aggregate --latex ${CSP_RUN_ARGS:-}
