#!/usr/bin/env python
"""Export a trained ALIGNN-FF model as a TorchScript `.pt` file for the
native LAMMPS pair style (pair_alignn).

Installed as the `export_torchscript.py` console script after
`pip install -e .` (see setup.py).

Usage:
    export_torchscript.py --model-dir OutputDir --out alignn_ff.pt

Or during development:
    python alignn/scripts/torch/export_torchscript.py \\
        --model-dir OutputDir --out alignn_ff.pt

The saved module exposes:
    forward_tensors_z(positions, lattice, atomic_numbers,
                       src, dst, shift, compute_stress) -> {"energy","forces","stress"}
The C++ pair style calls this entry point each MD step.
"""
from __future__ import annotations
import argparse, json
import torch

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="alignn_ff.pt")
    ap.add_argument("--atom-features", default="atomic_number",
                    help="must match what was used at training time")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float64", "float16", "bfloat16"],
                    help="float32 is the safe default; bfloat16 halves memory "
                         "with minimal accuracy loss; float16 is risky for "
                         "MD forces (may NaN on small gradient magnitudes).")
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float64": torch.float64,
                 "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.dtype == "float16":
        print("[warn] fp16 forces can NaN during long MD runs. "
              "Consider --dtype bfloat16 instead.")

    cfg_raw = json.load(open(f"{args.model_dir}/config.json"))
    mcfg = dict(cfg_raw["model"])
    mcfg["name"] = "alignn_atomwise_pure"
    mcfg["calculate_gradient"] = True
    if mcfg.get("stresswise_weight", 0) == 0:
        mcfg["stresswise_weight"] = 0.01
    cfg = ALIGNNAtomWisePureConfig(**mcfg)

    model = ALIGNNAtomWisePure(cfg)
    sd = torch.load(f"{args.model_dir}/best_model.pt", map_location="cpu",
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)}  unexpected={len(unexpected)}")

    # Baked-in CGCNN species table so C++ only ships atomic numbers (Z).
    model.register_species_table(atom_features=args.atom_features)
    model = model.to(dtype).eval()

    # TorchScript compile — this is the critical step. If it fails here,
    # you'll need to fix the offending annotation in the model source.
    scripted = torch.jit.script(model)

    # Smoke test the scripted model on a minimal 2-atom example.
    with torch.no_grad():
        pos = torch.tensor([[0., 0., 0.], [1.35, 1.35, 1.35]], dtype=dtype)
    pos.requires_grad_(True)
    lat = torch.eye(3, dtype=dtype) * 5.43
    Z   = torch.tensor([14, 14], dtype=torch.long)
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 0], dtype=torch.long)
    shift = torch.zeros(2, 3, dtype=dtype)
    # torch.det() isn't implemented for bf16/fp16 on CPU, so skip stress
    # in the smoke test for reduced-precision exports. It works fine on CUDA.
    needs_fp32_det = dtype in (torch.bfloat16, torch.float16)
    out = scripted.forward_tensors_z(pos, lat, Z, src, dst, shift,
                                      not needs_fp32_det)
    print("smoke test OK:",
          {k: (v.shape if hasattr(v, 'shape') else v) for k, v in out.items()})
    if needs_fp32_det:
        print("  (stress path skipped on CPU — bf16/fp16 det only works on CUDA)")

    torch.jit.save(scripted, args.out)
    print(f"wrote {args.out}  ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    main()
