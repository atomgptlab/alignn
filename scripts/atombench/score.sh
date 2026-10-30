#!/usr/bin/env bash
# Score one or more ALIGNN-CSP benchmark CSVs with AtomBench's own metric code,
# writing metrics.json next to each CSV and printing the headline numbers.
#
#   bash scripts/atombench/score.sh <csv> [<csv> ...]
#
# Needs an environment with pymatgen and average-minimum-distance. Configure
# via environment variables:
#
#   ATOMBENCH_REPO  clone of github.com/atomgptlab/atombench (searched for in a
#                   few common places if unset)
#   SCORE_ENV       conda environment to activate first; leave unset to score
#                   in the current environment
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <benchmark.csv> [<benchmark.csv> ...]" >&2
    exit 2
fi

# Optional environment switch, for setups that keep the scoring dependencies
# separate from the training ones.
if [ -n "${SCORE_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$SCORE_ENV"
fi

find_compute_metrics() {
    local candidates=()
    [ -n "${ATOMBENCH_REPO:-}" ] && candidates+=("$ATOMBENCH_REPO")
    candidates+=("$HOME/atombench" "$(dirname "$0")/../../../atombench")
    for repo in "${candidates[@]}"; do
        local p="$repo/scripts/scripts_consolidated/compute_metrics.py"
        [ -f "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}

if ! COMPUTE="$(find_compute_metrics)"; then
    echo "error: could not find AtomBench's compute_metrics.py." >&2
    echo "  Set ATOMBENCH_REPO to a clone of github.com/atomgptlab/atombench" >&2
    exit 1
fi

for csv in "$@"; do
    echo "=== $csv"
    python "$COMPUTE" --csv "$csv"
    python - "$csv" <<'PY'
import json
import pathlib
import sys

m = json.loads(
    pathlib.Path(sys.argv[1]).with_name("metrics.json").read_text()
    .replace("NaN", "null").replace("Infinity", "null")
)
rmse = m["RMSE"]["AtomGen"]
mae = m["MAE"]["average_mae"]
kld = [v for v in m["KLD"].values() if v is not None]
abc = [mae[k] for k in "abc"]
ang = [mae[k] for k in ("alpha", "beta", "gamma")]
print(f"  match_rate  {rmse['match_rate']}   "
      f"({rmse['n_matched']}/{rmse['n_total']})")
print(f"  RMSD        {rmse['mean_cartesian_rms_angstrom']}")
print(f"  ccRMSD      {m.get('ccRMSD', m.get('ccRMSE', {})).get('value')}")
print(f"  MAE abc     {sum(abc) / len(abc):.4f}")
print(f"  MAE angles  {sum(ang) / len(ang):.4f}")
print(f"  KLD mean    {sum(kld) / len(kld):.4f}")
PY
done
