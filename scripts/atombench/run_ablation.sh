#!/usr/bin/env bash
# Generate and score the ALIGNN-CSP pipeline variants on AtomBench.
#
# Usage: bash scripts/atombench/run_ablation.sh <checkpoint> <data-dir> <out-dir>
#
# The four variants isolate what each stage of the pipeline contributes:
#
#   raw        one sample per target, straight from the diffusion model
#   rank       N samples, pick the lowest ALIGNN-FF energy (no relaxation)
#   relax      one sample per target, relaxed with ALIGNN-FF
#   full       N samples, each relaxed, lowest energy wins
#
# Comparing `raw` to `full` says how much the force field is worth; comparing
# `rank` to `relax` says whether selection or refinement is doing the work.
set -euo pipefail

CKPT="${1:?usage: run_ablation.sh <checkpoint> <data-dir> <out-dir>}"
DATA="${2:?}"
OUT="${3:?}"
NCAND="${NCAND:-8}"
GUIDANCE="${GUIDANCE:-2.0}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-20}"
RELAX_STEPS="${RELAX_STEPS:-100}"
# Optional conda environments; leave unset to use the current one. SCORE_ENV is
# passed through to score.sh, which needs pymatgen and average-minimum-distance.
GEN_ENV="${GEN_ENV:-}"
export SCORE_ENV="${SCORE_ENV:-}"

if [ -n "$GEN_ENV" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$GEN_ENV"
fi

gen () {
    local name="$1"; shift
    mkdir -p "$OUT/$name"
    echo "=== generating: $name"
    CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 python -u \
        scripts/atombench/generate_benchmark.py \
        --checkpoint "$CKPT" --data-dir "$DATA" \
        --output-csv "$OUT/$name/alignn_csp_$name.csv" \
        --guidance "$GUIDANCE" \
        --relax-steps "$RELAX_STEPS" --relax-workers "$WORKERS" \
        --save-candidates "$OUT/$name/candidates.json" \
        "$@"
}

gen raw   --num-candidates 1      --relax none --rank none
gen rank  --num-candidates "$NCAND" --relax none --rank energy
gen relax --num-candidates 1      --relax cell --rank energy
gen full  --num-candidates "$NCAND" --relax cell --rank energy

echo
echo "=== scoring all variants"
bash scripts/atombench/score.sh "$OUT"/*/alignn_csp_*.csv
