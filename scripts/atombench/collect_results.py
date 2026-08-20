#!/usr/bin/env python3
"""Collect ALIGNN-CSP metrics.json files into one comparison table.

Prints our runs alongside the published AtomBench baselines (arXiv:2510.16165,
Table 2) so it is immediately clear which metrics we win and which we do not.

Usage:
    python scripts/atombench/collect_results.py \
        --runs name=path/to/metrics.json [name=path ...] --dataset jarvis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Published AtomBench baselines, arXiv:2510.16165 Table 2.
# match = match rate (higher better); everything else lower better.
BASELINES = {
    "jarvis": {
        "AtomGPT": {
            "kld": 0.0197,
            "abc": 0.518,
            "ang": 7.125,
            "rmsd": 0.0820,
            "ccrmsd": 0.8262,
            "match": 0.4902,
        },
        "CDVAE": {
            "kld": 0.0110,
            "abc": 0.348,
            "ang": 7.128,
            "rmsd": 0.4083,
            "ccrmsd": 1.1555,
            "match": 0.3592,
        },
        "FlowMM": {
            "kld": 0.0356,
            "abc": 0.936,
            "ang": 17.427,
            "rmsd": 0.4077,
            "ccrmsd": 1.5392,
            "match": 0.0291,
        },
        "MatterGen": {
            "kld": 0.0287,
            "abc": 0.505,
            "ang": 13.456,
            "rmsd": 0.0392,
            "ccrmsd": 0.4854,
            "match": 0.4660,
        },
    },
    "alex": {
        "AtomGPT": {
            "kld": 0.0215,
            "abc": 0.519,
            "ang": 8.937,
            "rmsd": 0.0378,
            "ccrmsd": 0.5224,
            "match": 0.5024,
        },
        "CDVAE": {
            "kld": 0.0118,
            "abc": 0.177,
            "ang": 8.351,
            "rmsd": 0.4181,
            "ccrmsd": 1.2218,
            "match": 0.3564,
        },
        "FlowMM": {
            "kld": 0.0380,
            "abc": 1.066,
            "ang": 17.883,
            "rmsd": 0.3810,
            "ccrmsd": 1.3998,
            "match": 0.0897,
        },
        "MatterGen": {
            "kld": 0.0242,
            "abc": 0.392,
            "ang": 12.114,
            "rmsd": 0.0138,
            "ccrmsd": 0.2380,
            "match": 0.6279,
        },
    },
}

PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")
LOWER_IS_BETTER = {
    "kld": True,
    "abc": True,
    "ang": True,
    "rmsd": True,
    "ccrmsd": True,
    "match": False,
}
COLUMNS = [
    ("match", "match↑"),
    ("rmsd", "RMSD↓"),
    ("ccrmsd", "ccRMSD↓"),
    ("abc", "MAEabc↓"),
    ("ang", "MAEang↓"),
    ("kld", "KLD↓"),
]


def extract(path: Path) -> dict:
    raw = json.loads(
        path.read_text().replace("NaN", "null").replace("Infinity", "null")
    )
    mae = raw.get("MAE", {}).get("average_mae", {})
    kld = raw.get("KLD", {})
    rmse = raw.get("RMSE", {}).get("AtomGen", {})
    cc = raw.get("ccRMSD", raw.get("ccRMSE", {}))

    def mean(vals):
        vals = [v for v in vals if v is not None and not _nan(v)]
        return sum(vals) / len(vals) if vals else None

    return {
        "kld": mean([kld.get(k) for k in PARAMS]),
        "abc": mean([mae.get(k) for k in ("a", "b", "c")]),
        "ang": mean([mae.get(k) for k in ("alpha", "beta", "gamma")]),
        "rmsd": rmse.get("mean_cartesian_rms_angstrom"),
        "ccrmsd": cc.get("value"),
        "match": rmse.get("match_rate"),
        "n_matched": rmse.get("n_matched"),
        "n_total": rmse.get("n_total"),
    }


def _nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def fmt(v, key) -> str:
    if v is None or _nan(v):
        return "     -"
    return (
        f"{v:6.4f}"
        if key in ("kld", "rmsd", "ccrmsd", "match")
        else f"{v:6.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="name=path/to/metrics.json entries",
    )
    ap.add_argument("--dataset", choices=["jarvis", "alex"], default="jarvis")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"!! missing {p} (skipping {name})")
            continue
        rows[name] = extract(p)

    base = BASELINES[args.dataset]
    all_rows = {**{f"[baseline] {k}": v for k, v in base.items()}, **rows}

    # Best value per column across everything shown.
    best = {}
    for key, _ in COLUMNS:
        vals = [
            r[key]
            for r in all_rows.values()
            if r.get(key) is not None and not _nan(r.get(key))
        ]
        if vals:
            best[key] = min(vals) if LOWER_IS_BETTER[key] else max(vals)

    label_w = max(len(k) for k in all_rows) + 2
    header = "model".ljust(label_w) + "".join(f"{h:>10}" for _, h in COLUMNS)
    print(f"\nAtomBench — {args.dataset}\n" + header)
    print("-" * len(header))
    for name, r in all_rows.items():
        line = name.ljust(label_w)
        for key, _ in COLUMNS:
            v = r.get(key)
            mark = (
                "*"
                if (
                    key in best
                    and v is not None
                    and not _nan(v)
                    and abs(v - best[key]) < 1e-12
                )
                else " "
            )
            line += f"{fmt(v, key)}{mark}".rjust(10)
        print(line)
    print("\n* = best in column.  Baselines: arXiv:2510.16165 Table 2.")

    for name, r in rows.items():
        if r.get("n_total"):
            print(f"  {name}: matched {r['n_matched']}/{r['n_total']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_rows, indent=2))


if __name__ == "__main__":
    main()
