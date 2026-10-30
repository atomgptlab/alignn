#!/usr/bin/env bash
# Submit one task to SLURM: a job array over its units, plus a dependent
# aggregation job that prints the table once every element has succeeded.
#
#   bash task_runners/submit.sh bench-jarvis
#   bash task_runners/submit.sh angle-ablation --seeds 0,1,2,3,4
#   bash task_runners/submit.sh bench-jarvis --dry-run     # show, don't submit
#
# Arguments after the task name are forwarded to run_task.py, and are used
# both to size the array and to run each element, so `--seeds 0,1` really does
# submit two elements.  Scheduler metadata comes from task_runners/cluster.env.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$HERE/cluster.env" ] && source "$HERE/cluster.env"

TASK="${1:?usage: submit.sh <task> [run_task.py args...]}"
shift || true

SBATCH_FILE="$HERE/sbatch/$TASK.sbatch"
if [ ! -f "$SBATCH_FILE" ]; then
    echo "no sbatch script for task '$TASK'" >&2
    echo "available:" >&2
    ls "$HERE/sbatch" | sed 's/\.sbatch$//' | sed 's/^/  /' >&2
    exit 2
fi

DRY=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then DRY=1; else ARGS+=("$arg"); fi
done

cd "$REPO"
export ALIGNN_RUNS="${CSP_RUNS:-$REPO/runs}"

# Size the array from the same arguments the elements will run with, so a
# non-default --seeds cannot silently under- or over-submit.
N_UNITS="$(python "$HERE/run_task.py" "$TASK" --count "${ARGS[@]+"${ARGS[@]}"}")"
if [ "$N_UNITS" -lt 1 ]; then
    echo "task '$TASK' has no units" >&2
    exit 1
fi

OPTS=()
[ -n "${CSP_ACCOUNT:-}" ]     && OPTS+=(--account="$CSP_ACCOUNT")
[ -n "${CSP_PARTITION:-}" ]   && OPTS+=(--partition="$CSP_PARTITION")
[ -n "${CSP_QOS:-}" ]         && OPTS+=(--qos="$CSP_QOS")
[ -n "${CSP_CONSTRAINT:-}" ]  && OPTS+=(--constraint="$CSP_CONSTRAINT")
[ -n "${CSP_RESERVATION:-}" ] && OPTS+=(--reservation="$CSP_RESERVATION")
if [ -n "${CSP_MAIL_USER:-}" ]; then
    OPTS+=(--mail-user="$CSP_MAIL_USER" --mail-type=END,FAIL)
fi
# --gres only makes sense for the GPU tasks; the CPU sbatch files declare
# none, and adding one there would queue them behind GPU availability.
if [ -n "${CSP_GPU_GRES:-}" ] && grep -q '^#SBATCH --gres' "$SBATCH_FILE"; then
    OPTS+=(--gres="$CSP_GPU_GRES")
fi
# shellcheck disable=SC2206
[ -n "${CSP_SBATCH_EXTRA:-}" ] && OPTS+=(${CSP_SBATCH_EXTRA})

THROTTLE="${CSP_MAX_CONCURRENT:-4}"
ARRAY="0-$((N_UNITS - 1))%${THROTTLE}"

# Forwarded through the submitting environment rather than --export=K=V, which
# does not survive values containing spaces or commas.
CSP_RUN_ARGS="${ARGS[*]+${ARGS[*]}}"
export CSP_RUN_ARGS CSP_TASK="$TASK"

echo "task:     $TASK"
echo "units:    $N_UNITS   array $ARRAY"
echo "run args: ${CSP_RUN_ARGS:-<none>}"
echo "sbatch:   ${OPTS[*]+${OPTS[*]}}"

if [ "$DRY" = "1" ]; then
    echo
    echo "would submit:"
    echo "  sbatch --array=$ARRAY ${OPTS[*]+${OPTS[*]}} $SBATCH_FILE"
    echo "  sbatch --dependency=afterok:<jobid> ${OPTS[*]+${OPTS[*]}} \\"
    echo "      $HERE/sbatch/aggregate.sbatch"
    exit 0
fi

JOB="$(sbatch --parsable --export=ALL --array="$ARRAY" \
        "${OPTS[@]+"${OPTS[@]}"}" "$SBATCH_FILE")"
echo "submitted array job $JOB"

AGG_OPTS=()
[ -n "${CSP_ACCOUNT:-}" ]   && AGG_OPTS+=(--account="$CSP_ACCOUNT")
[ -n "${CSP_PARTITION:-}" ] && AGG_OPTS+=(--partition="$CSP_PARTITION")
[ -n "${CSP_QOS:-}" ]       && AGG_OPTS+=(--qos="$CSP_QOS")
AGG="$(sbatch --parsable --export=ALL --dependency="afterok:$JOB" \
        "${AGG_OPTS[@]+"${AGG_OPTS[@]}"}" "$HERE/sbatch/aggregate.sbatch")"
echo "submitted aggregation job $AGG (after $JOB)"
echo
echo "watch:   squeue -j $JOB,$AGG"
echo "logs:    task_runners/logs/"
echo "table:   python task_runners/run_task.py $TASK --aggregate"
