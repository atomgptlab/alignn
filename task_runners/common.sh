# Shared bootstrap, sourced by every script in task_runners/sbatch.
#
# Reads cluster.env, loads modules, activates the conda environment and
# exports the variables the underlying scripts read (ALIGNN_RUNS,
# ATOMBENCH_REPO, SCORE_ENV).  Sourcing this from an interactive shell is a
# perfectly good way to get the same environment by hand.

set -euo pipefail

CSP_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSP_REPO="$(cd "$CSP_HERE/.." && pwd)"

# shellcheck disable=SC1091
[ -f "$CSP_HERE/cluster.env" ] && source "$CSP_HERE/cluster.env"

if [ -n "${CSP_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load ${CSP_MODULES}
fi

if [ -n "${CSP_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CSP_ENV"
fi

# score.sh switches to this environment itself, if it is set.
export SCORE_ENV="${CSP_SCORE_ENV:-}"
[ -n "${CSP_ATOMBENCH_REPO:-}" ] && export ATOMBENCH_REPO="$CSP_ATOMBENCH_REPO"
export ALIGNN_RUNS="${CSP_RUNS:-$CSP_REPO/runs}"
export PYTHONUNBUFFERED=1

if [ -n "${CSP_PRE_RUN_HOOK:-}" ]; then
    eval "$CSP_PRE_RUN_HOOK"
fi

cd "$CSP_REPO"

echo "repo:      $CSP_REPO"
echo "runs:      $ALIGNN_RUNS"
echo "python:    $(command -v python)"
echo "host:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none} array element ${SLURM_ARRAY_TASK_ID:-none}"
