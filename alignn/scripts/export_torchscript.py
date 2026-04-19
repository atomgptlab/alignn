"""Export a trained pure-torch ALIGNN model to TorchScript.

The saved ``.pt`` file is self-contained: it holds the model weights,
architecture code (scripted), and an entry-point ``forward(...)``
that takes plain tensors (positions, lattice, atom_features, src, dst,
shift) and returns ``{"energy", "forces", ["stress"]}``.

This is what a LAMMPS ``pair_alignn`` plugin (or any libtorch C++ host)
would call each MD step — the only runtime dependency is libtorch.

Usage
-----
    python alignn/scripts/export_torchscript.py \
        --checkpoint /path/to/best_model.pt \
        --config     /path/to/config.json \
        --output     alignn_scripted.pt
    # test the scripted file on a random structure
    python alignn/scripts/export_torchscript.py \
        --checkpoint ... --config ... --output out.pt --test

Design notes
------------
Only the ``forward_tensors`` path is scripted (a thin Wrapper module
exposes it as ``forward``). The dataclass-based training forward is
left alone. Classification and extra-features branches are not
exported — they're training-only features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)


class _ScriptWrapper(nn.Module):
    """LAMMPS-style entry point for libtorch.

    Takes atomic numbers (not pre-computed features) — the feature
    table is baked into the inner model via ``register_species_table``
    so the scripted ``.pt`` is fully self-contained.
    """

    def __init__(self, inner: ALIGNNAtomWisePure):
        super().__init__()
        self.inner = inner

    def forward(
        self,
        positions: torch.Tensor,  # (N, 3), requires_grad=True for forces
        lattice: torch.Tensor,  # (3, 3)
        atomic_numbers: torch.Tensor,  # (N,) long
        src: torch.Tensor,  # (E,) long
        dst: torch.Tensor,  # (E,) long
        shift: torch.Tensor,  # (E, 3) integer cell offsets
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.inner.forward_tensors_z(
            positions, lattice, atomic_numbers, src, dst, shift, compute_stress
        )


def build_model(
    checkpoint_path: Path,
    config_path: Path,
    atom_features: str = "cgcnn",
) -> ALIGNNAtomWisePure:
    config = json.load(open(config_path))
    mcfg = config["model"] if "model" in config else config
    if mcfg.get("name") != "alignn_atomwise_pure":
        # Accept a DGL-trained checkpoint's config by overriding the name.
        # Architecture is identical between alignn_atomwise and
        # alignn_atomwise_pure, so weights are state-dict compatible.
        mcfg = dict(mcfg, name="alignn_atomwise_pure")
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**mcfg))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Drop any previously-saved species table — it'll be rebuilt fresh.
    state = {k: v for k, v in state.items() if k != "_species_table"}
    model.load_state_dict(state, strict=False)
    model.register_species_table(atom_features=atom_features)
    model.eval()
    return model


def _smoke_test(scripted: torch.jit.ScriptModule) -> None:
    """Run a tiny Cu-FCC box through the scripted model (atomic-number input)."""
    from jarvis.io.vasp.inputs import Poscar
    from alignn.graphs import Graph
    from alignn.torch_graph_builder import torchgraph_from_dgl

    atoms = Poscar.from_string(
        """Cu
1.0
3.6 0.0 0.0
0.0 3.6 0.0
0.0 0.0 3.6
Cu
4
direct
0.0 0.0 0.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
"""
    ).atoms
    g, _ = Graph.atom_dgl_multigraph(
        atoms,
        neighbor_strategy="fast_graph",
        cutoff=4.0,
        max_neighbors=12,
        atom_features="cgcnn",
    )
    tg = torchgraph_from_dgl(g)

    pos = torch.as_tensor(np.asarray(atoms.cart_coords), dtype=torch.float32)
    pos.requires_grad_(True)
    lat = torch.as_tensor(np.asarray(atoms.lattice_mat), dtype=torch.float32)
    # LAMMPS-style input: atomic numbers, not feature vectors.
    z_tensor = torch.as_tensor(atoms.atomic_numbers, dtype=torch.long)
    shift = tg.edata["images"].float()

    out = scripted(pos, lat, z_tensor, tg.src, tg.dst, shift, True)
    print("Scripted smoke test (atomic-number input):")
    print(f"  energy: {out['energy'].item():.6f}")
    print(
        f"  forces: shape={tuple(out['forces'].shape)}  "
        f"max|F|={out['forces'].abs().max().item():.4e}  "
        f"sum(F)={out['forces'].detach().sum(0).tolist()}"
    )
    print(
        f"  stress: shape={tuple(out['stress'].shape)}  "
        f"max|σ|={out['stress'].abs().max().item():.4e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--atom-features",
        default=None,
        help="Feature scheme for the atomic-number lookup table "
        "(cgcnn/atomic_number/basic/cfid). Defaults to the value "
        "found in the training config, or 'cgcnn'.",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Run the scripted model on a tiny Cu-FCC box after saving.",
    )
    args = ap.parse_args()

    # Resolve atom_features: CLI > training config > 'cgcnn'.
    cfg = json.load(open(args.config))
    atom_features = args.atom_features or cfg.get("atom_features", "cgcnn")
    print(f"Loading model from {args.checkpoint} + {args.config}")
    print(f"  atom_features (species lookup) = {atom_features!r}")
    model = build_model(
        args.checkpoint, args.config, atom_features=atom_features
    )

    print("Wrapping + scripting...")
    wrapper = _ScriptWrapper(model).eval()
    scripted = torch.jit.script(wrapper)

    scripted.save(str(args.output))
    print(f"Saved scripted model -> {args.output}")
    print(
        "  Input signature: (positions[N,3], lattice[3,3], "
        "atomic_numbers[N], src[E], dst[E], shift[E,3], compute_stress: bool)"
    )
    print('  Output: {"energy": scalar, "forces": [N,3], ["stress": [3,3]]}')

    if args.test:
        scripted = torch.jit.load(str(args.output), map_location="cpu")
        _smoke_test(scripted)


if __name__ == "__main__":
    main()
