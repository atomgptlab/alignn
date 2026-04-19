"""Parity sweep: DGL ALIGNN vs pure-torch ALIGNN across the doc examples.

Runs each scenario from ``docs/training/*.md`` twice -- once with the
DGL-backed ``alignn_atomwise`` model and once with the DGL-free
``alignn_atomwise_pure`` model -- and compares final train/val/test
losses side-by-side. Graph construction and LMDB storage flip with the
model: the DGL run uses ``alignn/lmdb_dataset.py``, the pure run uses
``alignn/pure_lmdb_dataset.py`` (both selected automatically by
``train_alignn.py`` based on ``config.model.name``).

Scenarios covered (match the docs):
  1. Single-output regression      (alignn/examples/sample_data)
  2. Classification                (sample_data + classification_threshold)
  3. Multi-output regression       (alignn/examples/sample_data_multi_prop)
  4. Force field                   (alignn/examples/sample_data_ff)

Usage
-----
    python alignn/scripts/parity_dgl_vs_pure.py                # all scenarios
    python alignn/scripts/parity_dgl_vs_pure.py --only ff      # one scenario
    python alignn/scripts/parity_dgl_vs_pure.py --epochs 1     # quick smoke

Exits 0 on success; prints a table of metrics and |Δ| between backends.

Notes
-----
* Graph caches (<filename>{train,val,test}_data) are deleted before each
  run so the LMDB store always matches the current model backend.
* Residual drift of ~1e-2 on train/val is expected from scatter_add
  nondeterminism on CUDA (accumulation order differs from DGL's kernels
  and run-to-run). Set ``torch.use_deterministic_algorithms(True)`` to
  eliminate at the cost of speed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

# Repo root = two parents up from this file (alignn/scripts/ -> repo root).
REPO = Path(__file__).resolve().parents[2]
FLOAT = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"


SCENARIOS: List[Dict] = [
    {
        "key": "single",
        "name": "single-output regression (sample_data)",
        "base_config": "alignn/examples/sample_data/config_example.json",
        "root_dir": "alignn/examples/sample_data",
        "extra": None,
    },
    {
        "key": "classification",
        "name": "classification (sample_data, threshold 0.01)",
        "base_config": "alignn/examples/sample_data/config_example.json",
        "root_dir": "alignn/examples/sample_data",
        "extra": {"classification_threshold": 0.01},
    },
    {
        "key": "multi",
        "name": "multi-output regression (sample_data_multi_prop)",
        "base_config": "alignn/examples/sample_data/config_example.json",
        "root_dir": "alignn/examples/sample_data_multi_prop",
        "extra": None,
    },
    {
        "key": "ff",
        "name": "force field (sample_data_ff)",
        "base_config": "alignn/examples/sample_data_ff/config_example_atomwise.json",
        "root_dir": "alignn/examples/sample_data_ff",
        "extra": None,
    },
]


def _patch_cfg(
    base_path: Path,
    out_path: Path,
    *,
    pure: bool,
    extra: Optional[dict] = None,
    epochs: Optional[int] = None,
) -> None:
    cfg = json.load(open(base_path))
    cfg["use_lmdb"] = True
    cfg["read_existing"] = False
    cfg["num_workers"] = 0
    if pure:
        cfg["model"]["name"] = "alignn_atomwise_pure"
    if extra:
        cfg.update(extra)
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    json.dump(cfg, open(out_path, "w"))


def _run(cmd: List[str], log_path: Path) -> "tuple[int, float]":
    t0 = time.time()
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def _extract(log_path: Path) -> Dict:
    txt = log_path.read_text()
    train = re.findall(rf"Train Loss:.*\n.*?(\d+)\s+({FLOAT})", txt)
    val = re.findall(rf"Val Loss:.*\n.*?(\d+)\s+({FLOAT})", txt)
    test = re.search(rf"TestLoss\s+\d+\s+({FLOAT})", txt)
    return {
        "train_last": float(train[-1][1]) if train else None,
        "val_last": float(val[-1][1]) if val else None,
        "test": float(test.group(1)) if test else None,
        "final_epoch": int(train[-1][0]) if train else None,
    }


def _wipe_graph_caches(filename_stem: str) -> None:
    """Remove any stale LMDB cache dirs tied to the cfg's ``filename`` stem."""
    for suffix in ("train_data", "val_data", "test_data"):
        p = REPO / f"{filename_stem}{suffix}"
        if p.exists():
            shutil.rmtree(p)


def run_scenario(s: Dict, epochs: Optional[int] = None) -> Dict[str, Dict]:
    base_cfg = REPO / s["base_config"]
    stem = json.load(open(base_cfg)).get("filename", "")
    results: Dict[str, Dict] = {}
    for mode in ("dgl", "pure"):
        _wipe_graph_caches(stem)
        patched_cfg = Path(f"/tmp/parity_{s['key']}_{mode}.json")
        _patch_cfg(
            base_cfg,
            patched_cfg,
            pure=(mode == "pure"),
            extra=s["extra"],
            epochs=epochs,
        )
        out_dir = Path(f"/tmp/parity_out_{s['key']}_{mode}")
        shutil.rmtree(out_dir, ignore_errors=True)
        log = Path(f"/tmp/parity_{s['key']}_{mode}.log")
        rc, dt = _run(
            [
                "python",
                "alignn/train_alignn.py",
                "--root_dir",
                s["root_dir"],
                "--config",
                str(patched_cfg),
                "--output_dir",
                str(out_dir),
            ],
            log,
        )
        res = _extract(log) if rc == 0 else {
            "train_last": None,
            "val_last": None,
            "test": None,
            "final_epoch": None,
        }
        res.update(rc=rc, time_s=dt, log=str(log))
        results[mode] = res
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(
            f"  [{mode:4s}] {status:10s}  "
            f"train={res['train_last']}  val={res['val_last']}  "
            f"test={res['test']}  ({dt:.1f}s)"
        )
    return results


def _fmt(x: Optional[float], width: int = 14) -> str:
    if x is None:
        return f"{'n/a':>{width}s}"
    return f"{x:>{width}.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=[s["key"] for s in SCENARIOS],
        help="Run just one scenario.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config.epochs (default: use JSON value).",
    )
    args = ap.parse_args()

    os.chdir(REPO)
    scenarios = (
        [s for s in SCENARIOS if s["key"] == args.only]
        if args.only
        else SCENARIOS
    )

    all_results: Dict[str, Dict[str, Dict]] = {}
    for s in scenarios:
        print(f"\n=== {s['name']} ===")
        all_results[s["name"]] = run_scenario(s, epochs=args.epochs)

    print("\n=== Parity summary ===")
    hdr = f"{'scenario':52s} {'metric':8s} {'DGL':>14s} {'pure':>14s} {'|Δ|':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for name, d in all_results.items():
        for metric in ("train_last", "val_last", "test"):
            a = d.get("dgl", {}).get(metric)
            b = d.get("pure", {}).get(metric)
            delta = abs(a - b) if (a is not None and b is not None) else None
            print(
                f"{name:52s} {metric:8s} "
                f"{_fmt(a)} {_fmt(b)} {_fmt(delta, width=12)}"
            )


if __name__ == "__main__":
    main()
