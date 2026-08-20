#!/usr/bin/env python3
"""Drop pretraining-leaked targets from a benchmark CSV before scoring.

A model pretrained on a corpus drawn from the same database as the benchmark
can reach some test targets by recall rather than by generation.  Scoring the
complement of that overlap gives the number that is comparable to a model
trained only on the benchmark split.

Takes the leakage report written by ``check_pretrain_leakage.py`` and emits a
CSV with those ids removed, to be scored the usual way.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(10**8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--leakage-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = json.loads(Path(args.leakage_json).read_text())
    leaked = {str(h["test_id"]).strip() for h in report.get("hits", [])}

    rows_in, rows_out = 0, []
    with Path(args.csv).open() as fh:
        for row in csv.DictReader(fh):
            rows_in += 1
            if str(row["id"]).strip() in leaked:
                continue
            rows_out.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "target", "prediction"])
        w.writeheader()
        w.writerows(rows_out)

    print(
        f"{rows_in} rows in, dropped {rows_in - len(rows_out)} leaked, "
        f"{len(rows_out)} written -> {out}"
    )


if __name__ == "__main__":
    main()
