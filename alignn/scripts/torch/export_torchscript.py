"""Export a trained ALIGNN-FF model as a TorchScript `.pt` file for the
native LAMMPS pair style (pair_alignn).

Usage:
    python scripts/torch/export_torchscript.py \\
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
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

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
    out = scripted.forward_tensors_z(pos, lat, Z, src, dst, shift, True)
    print("smoke test OK:",
          {k: (v.shape if hasattr(v, 'shape') else v) for k, v in out.items()})

    torch.jit.save(scripted, args.out)
    print(f"wrote {args.out}  ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    main()
