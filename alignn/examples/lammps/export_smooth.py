"""Export a trained ALIGNN-atomwise-pure-SMOOTH checkpoint to TorchScript.

The stock `export_torchscript` script hardcodes the non-smooth class;
this is the smooth-variant analogue.

Usage:
    python export_smooth.py --model-dir <dir-with-best_model.pt+config.json> \
                            --out alignn_ff.pt
"""
import argparse
import json
import torch

from alignn.models.alignn_atomwise_pure_smooth import (
    ALIGNNAtomWisePureSmooth, ALIGNNAtomWisePureSmoothConfig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="alignn_ff.pt")
    ap.add_argument("--atom-features", default="atomic_number")
    args = ap.parse_args()

    cfg_raw = json.load(open(f"{args.model_dir}/config.json"))
    mcfg = dict(cfg_raw["model"])
    mcfg["name"] = "alignn_atomwise_pure_smooth"
    mcfg["calculate_gradient"] = True
    if mcfg.get("stresswise_weight", 0) == 0:
        mcfg["stresswise_weight"] = 0.01
    cfg = ALIGNNAtomWisePureSmoothConfig(**mcfg)

    model = ALIGNNAtomWisePureSmooth(cfg)
    sd = torch.load(f"{args.model_dir}/best_model.pt", map_location="cpu",
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing: {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")

    model.register_species_table(atom_features=args.atom_features)
    model = model.to(torch.float32).eval()

    scripted = torch.jit.script(model)

    with torch.no_grad():
        pos = torch.tensor([[0., 0., 0.], [1.35, 1.35, 1.35]],
                           dtype=torch.float32)
    pos.requires_grad_(True)
    lat = torch.eye(3, dtype=torch.float32) * 5.43
    Z = torch.tensor([14, 14], dtype=torch.long)
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 0], dtype=torch.long)
    shift = torch.zeros(2, 3, dtype=torch.float32)
    out = scripted.forward_tensors_z(pos, lat, Z, src, dst, shift, True)
    print("smoke test OK:",
          {k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()})

    torch.jit.save(scripted, args.out)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out}  ({n:,} params)")


if __name__ == "__main__":
    main()
