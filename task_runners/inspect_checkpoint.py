#!/usr/bin/env python3
"""Print the training flags baked into an ALIGNN-CSP checkpoint.

``train_csp.py`` stores the full argument namespace in every checkpoint and in
``config.json`` next to it, so a released model can say exactly how it was
trained.  Two epoch counts in ``task_runners/tasks.py`` (Alexandria and the
dft_3d pretraining run) are *not* pinned by the manuscript; this is how to
replace them with the published values instead of guessing.

    python task_runners/inspect_checkpoint.py csp_supercon_alex
    python task_runners/inspect_checkpoint.py runs/train/jarvis_A0/seed0
    python task_runners/inspect_checkpoint.py path/to/best_model.pt --command
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Flags that describe the run rather than the machine it ran on.
INTERESTING = [
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_features",
    "alignn_layers",
    "gcn_layers",
    "knn",
    "num_steps",
    "sigma_min",
    "sigma_max",
    "prop_dropout",
    "composition_dropout",
    "lattice_weight",
    "frac_weight",
    "angle_weight",
    "ablation",
    "angle_diffusion",
    "angle_feedback",
    "topology",
    "radius_cutoff",
    "envelope_exponent",
    "gate_pair_messages",
    "angle_basis",
    "ema_decay",
    "grad_clip",
    "augment",
    "init_from",
    "seed",
    "n_parameters",
]

SKIP_IN_COMMAND = {"n_parameters", "device", "log_every", "output", "data_dir"}


def load_config(target: str) -> dict:
    """Config for a local checkpoint, a run directory, or a released name."""
    path = Path(target)
    if path.is_dir():
        for name in ("config.json",):
            if (path / name).exists():
                return json.loads((path / name).read_text())
        path = path / "best_model.pt"
    if path.suffix == ".pt" and path.exists():
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("config", {})
    if path.exists():
        return json.loads(path.read_text())

    # Not a path: treat it as a name in the ALIGNN 2.0 registry.
    from alignn.pretrained import get_alignn2_model

    paths = get_alignn2_model(target)
    cfg_path = _find(paths, "config.json")
    if cfg_path:
        return json.loads(Path(cfg_path).read_text())
    ckpt_path = _find(paths, "best_model.pt")
    if not ckpt_path:
        raise FileNotFoundError(f"no config or checkpoint for {target!r}")
    import torch

    return torch.load(ckpt_path, map_location="cpu", weights_only=False).get(
        "config", {}
    )


def _find(paths, name: str):
    """Pull one artifact out of whatever get_alignn2_model returned."""
    if isinstance(paths, dict):
        for value in paths.values():
            if str(value).endswith(name):
                return value
        return None
    if isinstance(paths, (list, tuple)):
        for value in paths:
            if str(value).endswith(name):
                return value
        return None
    candidate = Path(paths)
    if candidate.is_dir() and (candidate / name).exists():
        return candidate / name
    return candidate if str(candidate).endswith(name) else None


def as_command(cfg: dict) -> str:
    parts = [
        "python -m alignn.inverse.train_csp",
        "    --data-dir DATA",
        "    --output OUT",
    ]
    for key, value in sorted(cfg.items()):
        if key in SKIP_IN_COMMAND or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"    {flag} {value}")
    return " \\\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "target",
        help="a run directory, a .pt/config.json, or a registered model name "
        "(csp_supercon_jarvis, csp_supercon_alex, csp_pretrain_dft3d, ...)",
    )
    ap.add_argument(
        "--command",
        action="store_true",
        help="print a train_csp command line instead of a table",
    )
    ap.add_argument("--all", action="store_true", help="every stored key")
    args = ap.parse_args()

    try:
        cfg = load_config(args.target)
    except Exception as exc:  # noqa: BLE001 - a CLI, not a library
        print(f"could not read {args.target!r}: {exc}", file=sys.stderr)
        return 1
    if not cfg:
        print(f"{args.target}: no config recorded", file=sys.stderr)
        return 1

    if args.command:
        print(as_command(cfg))
        return 0

    keys = sorted(cfg) if args.all else [k for k in INTERESTING if k in cfg]
    width = max(len(k) for k in keys)
    print(f"\n{args.target}")
    for key in keys:
        print(f"  {key.ljust(width)}  {cfg[key]}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
