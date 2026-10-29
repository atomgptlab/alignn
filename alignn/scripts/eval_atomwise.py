"""Per-atom MAE/RMSE from an atomwise training run.

Flattens ``target_atomwise_pred`` / ``pred_atomwise_pred`` over all atoms of
all test structures in the run's ``Test_results.json`` (written by
``alignn.train``) and reports per-atom MAE and RMSE, e.g. for the
manuscript's per-atom charges/magmoms table.

Usage:
    python eval_atomwise.py --run_dir out_magmom
    python eval_atomwise.py --run_dir out_charge --output charges.json
"""

import argparse
import json
import math
import os


def flatten(values):
    """Nested per-batch lists -> flat list of floats."""
    out = []
    stack = [values]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(float(v))
    return out


def main():
    parser = argparse.ArgumentParser(description="Per-atom test MAE/RMSE.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Summary JSON path (default: <run_dir>/atomwise_summary.json).",
    )
    args = parser.parse_args()

    results_path = os.path.join(args.run_dir, "Test_results.json")
    if not os.path.exists(results_path):
        raise SystemExit("Missing " + results_path)
    results = json.load(open(results_path))

    targets, preds = [], []
    n_struct = 0
    for info in results:
        t = info.get("target_atomwise_pred")
        p = info.get("pred_atomwise_pred")
        if t is None or p is None:
            continue
        n_struct += 1
        targets.extend(flatten(t))
        preds.extend(flatten(p))
    if not targets or len(targets) != len(preds):
        raise SystemExit(
            "No (or mismatched) atomwise predictions in "
            + results_path
            + " - was the model trained with atomwise_output_features > 0?"
        )

    errors = [t - p for t, p in zip(targets, preds)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "n_test_batches": n_struct,
        "n_atoms": len(targets),
        "mae": mae,
        "rmse": rmse,
    }
    out_path = args.output or os.path.join(
        args.run_dir, "atomwise_summary.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
