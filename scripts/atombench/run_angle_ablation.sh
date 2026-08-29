#!/usr/bin/env bash
# Train, generate and score the angular-diffusion ablation suite.
#
#   bash scripts/atombench/run_angle_ablation.sh <data-dir> <out-dir> [seeds...]
#
# Every arm gets the same data split, the same optimiser settings, the same
# epoch budget and the same seed list; the arms differ only in the switches
# under test. That is the whole point — a single run of the proposed model
# against a single run of the baseline cannot separate the effect from the
# spread across seeds, which on splits this size has been large enough to
# invert comparisons before (see alignn/inverse/README.md).
#
#   A0  baseline, angles as features only
#   A1  + explicit angular denoising
#   A2  + smooth radius topology, no angular objective
#   A3  both: the proposed model
#   A4  control: angular objective with the angle->bond coupling cut
#   A6  A3 with the Fourier angular basis
#
# A5 in the design brief is the A1-vs-A3 (and A0-vs-A2) contrast, which these
# runs already provide.
set -euo pipefail

DATA="${1:?usage: run_angle_ablation.sh <data-dir> <out-dir> [seeds...]}"
OUT="${2:?}"
shift 2
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)

ARMS="${ARMS:-A0 A1 A2 A3 A4}"
EPOCHS="${EPOCHS:-3000}"
NCAND="${NCAND:-8}"
GUIDANCE="${GUIDANCE:-2.0}"
ANGLE_WEIGHT="${ANGLE_WEIGHT:-1.0}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-20}"
export SCORE_ENV="${SCORE_ENV:-}"

for arm in $ARMS; do
    for seed in "${SEEDS[@]}"; do
        run="$OUT/${arm}_s${seed}"
        echo "=== training $arm seed $seed -> $run"
        CUDA_VISIBLE_DEVICES="$GPU" python -u -m alignn.inverse.train_csp \
            --data-dir "$DATA" --output "$run" \
            --ablation "$arm" --angle-weight "$ANGLE_WEIGHT" \
            --epochs "$EPOCHS" --seed "$seed"

        echo "=== generating $arm seed $seed"
        CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 python -u \
            scripts/atombench/generate_benchmark.py \
            --checkpoint "$run/best_model.pt" --data-dir "$DATA" \
            --output-csv "$run/bench.csv" \
            --num-candidates "$NCAND" --guidance "$GUIDANCE" \
            --relax cell --rank energy --relax-workers "$WORKERS"

        echo "=== mechanism metrics $arm seed $seed"
        python scripts/atombench/angle_eval.py "$run/bench.csv" --relax
    done
done

echo
echo "=== benchmark scores"
bash scripts/atombench/score.sh "$OUT"/*/bench.csv
