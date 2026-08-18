"""Export a trained pure-torch ALIGNN-FF to an OpenMM ``TorchForce`` module.

OpenMM's ``openmm-torch`` plugin (``TorchForce``) calls a scripted module
``forward(positions[, boxvectors]) -> energy`` and obtains forces by autograd.
This wrapper bridges the unit/interface gap:

  * OpenMM units: positions/box in **nm**, energy in **kJ/mol**.
    ALIGNN units: **Angstrom / eV**.  We convert on the way in/out; forces
    come out correct (kJ/mol/nm) automatically through autograd.
  * ``TorchForce`` only supplies positions (+ box), so the atomic numbers of
    the system are **baked into the module** at export time -- export one
    module per system (same atom count/order as your OpenMM ``System``).
  * The wrapper builds its own periodic neighbor list (no NNPOps dependency),
    scriptable and differentiable through the atom positions.

Usage
-----
    # bundled default ALIGNN-FF (matpes_r2scan), atoms taken from a POSCAR
    python alignn/scripts/export_openmm.py \
        --structure POSCAR --output alignn_openmm.pt
    # then in OpenMM:
    #   from openmmtorch import TorchForce
    #   f = TorchForce('alignn_openmm.pt')
    #   f.setUsesPeriodicBoundaryConditions(True)
    #   system.addForce(f)

Scaling note: the neighbor search is O(N^2 * n_images) -- fine for materials
cells / modest systems.  For large solvated biomolecular systems use NNPOps'
``getNeighborPairs`` in place of ``_build_neighbors``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)

EV_TO_KJMOL = 96.48533212331  # 1 eV in kJ/mol
NM_TO_ANG = 10.0


class OpenMMALIGNNForce(nn.Module):
    """Scriptable positions(+box) -> energy(kJ/mol) module for TorchForce."""

    def __init__(
        self,
        inner: ALIGNNAtomWisePure,
        atomic_numbers: Tensor,  # (N,) long -- the system's atoms
        cutoff: float,
        n_images: int = 1,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("atomic_numbers", atomic_numbers.long())
        self.cutoff = float(cutoff)
        self.n_images = int(n_images)

    def _image_shifts(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        rng = torch.arange(
            -self.n_images, self.n_images + 1, device=device, dtype=dtype
        )
        out: List[Tensor] = []
        for a in rng:
            for b in rng:
                for c in rng:
                    out.append(torch.stack([a, b, c]))
        return torch.stack(out)  # (n_shifts, 3)

    def _build_neighbors(self, pos: Tensor, cell: Tensor):
        # pos (N,3) in Angstrom; cell (3,3) rows are lattice vectors.
        n = pos.shape[0]
        idx = torch.arange(n, device=pos.device)
        ii = idx.repeat_interleave(n)  # (N*N,)
        jj = idx.repeat(n)
        shifts = self._image_shifts(pos.device, pos.dtype)
        src_l: List[Tensor] = []
        dst_l: List[Tensor] = []
        sh_l: List[Tensor] = []
        for s in range(shifts.shape[0]):
            sh = shifts[s]
            disp = (
                pos.index_select(0, jj)
                + torch.matmul(sh, cell)
                - pos.index_select(0, ii)
            )
            d = torch.linalg.vector_norm(disp, dim=1)
            zero_shift = bool(torch.sum(torch.abs(sh)).item() == 0.0)
            if zero_shift:
                mask = (d < self.cutoff) & (ii != jj)
            else:
                mask = (d < self.cutoff) & (d > 1e-8)
            sel = torch.nonzero(mask).squeeze(-1)
            if sel.shape[0] > 0:
                src_l.append(ii.index_select(0, sel))
                dst_l.append(jj.index_select(0, sel))
                sh_l.append(sh.unsqueeze(0).expand(sel.shape[0], 3))
        src = torch.cat(src_l, dim=0)
        dst = torch.cat(dst_l, dim=0)
        shift = torch.cat(sh_l, dim=0)
        return src, dst, shift

    def forward(self, positions: Tensor, boxvectors: Tensor) -> Tensor:
        # nm -> Angstrom (10.0); eV -> kJ/mol (96.48533212331). Literals
        # because TorchScript cannot close over module-level globals.
        pos = positions.to(torch.float32) * 10.0
        cell = boxvectors.to(torch.float32) * 10.0
        src, dst, shift = self._build_neighbors(pos, cell)
        out = self.inner.forward_tensors_z(
            pos, cell, self.atomic_numbers, src, dst, shift, False
        )
        energy_ev = out["energy"]
        return energy_ev * 96.48533212331


def build_model(checkpoint: Path, config: Path, atom_features: str):
    cfg = json.load(open(config))
    mcfg = cfg["model"] if "model" in cfg else cfg
    mcfg = dict(mcfg, name="alignn_atomwise_pure")
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**mcfg))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k: v for k, v in state.items() if k != "_species_table"}
    model.load_state_dict(state, strict=False)
    model.register_species_table(atom_features=atom_features)
    model.eval()
    return model, mcfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, type=Path)
    ap.add_argument("--config", default=None, type=Path)
    ap.add_argument(
        "--structure",
        default=None,
        help="POSCAR whose atomic numbers/order match your OpenMM System. "
        "Defaults to an 8-atom Si cell (demo).",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--atom-features", default=None)
    ap.add_argument("--n-images", type=int, default=1)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.checkpoint is None or args.config is None:
        from alignn.ff.calculators import default_path

        d = Path(default_path())
        args.checkpoint = args.checkpoint or d / "best_model.pt"
        args.config = args.config or d / "config.json"

    from jarvis.io.vasp.inputs import Poscar

    if args.structure:
        atoms = Poscar.from_file(args.structure).atoms
    else:
        atoms = Poscar.from_string(
            "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n8\ndirect\n"
            "0 0 0\n0.5 0.5 0\n0.5 0 0.5\n0 0.5 0.5\n0.25 0.25 0.25\n"
            "0.75 0.75 0.25\n0.75 0.25 0.75\n0.25 0.75 0.75\n"
        ).atoms

    cfg = json.load(open(args.config))
    mcfg = cfg["model"] if "model" in cfg else cfg
    atom_features = args.atom_features or cfg.get("atom_features", "cgcnn")
    cutoff = float(mcfg.get("cutoff", 5.0))

    model, _ = build_model(args.checkpoint, args.config, atom_features)
    z = torch.as_tensor(atoms.atomic_numbers, dtype=torch.long)
    wrapper = OpenMMALIGNNForce(model, z, cutoff, args.n_images).eval()
    print(
        f"Scripting OpenMM TorchForce module (N={len(z)}, cutoff={cutoff})..."
    )
    scripted = torch.jit.script(wrapper)
    scripted.save(str(args.output))
    print(f"Saved -> {args.output}")
    print(
        "OpenMM usage:\n"
        "  from openmmtorch import TorchForce\n"
        f"  f = TorchForce('{args.output}')\n"
        "  f.setUsesPeriodicBoundaryConditions(True)\n"
        "  system.addForce(f)   # atom order must match the export structure"
    )

    if args.test:
        _smoke(scripted, atoms, model)


def _smoke(scripted, atoms, model):
    """Compare scripted OpenMM-unit energy/force to ASE calc (eV/A)."""
    from alignn.ff.calculators import AlignnAtomwiseCalculator, default_path

    pos_nm = torch.as_tensor(
        np.asarray(atoms.cart_coords) / NM_TO_ANG, dtype=torch.float32
    )
    pos_nm.requires_grad_(True)
    box_nm = torch.as_tensor(
        np.asarray(atoms.lattice_mat) / NM_TO_ANG, dtype=torch.float32
    )
    e_kj = scripted(pos_nm, box_nm)
    (force_kj_nm,) = torch.autograd.grad(-e_kj, pos_nm)
    e_ev = float(e_kj.item()) / EV_TO_KJMOL
    fmax_ev_a = float(force_kj_nm.abs().max().item()) / EV_TO_KJMOL / NM_TO_ANG
    print(
        f"\nScripted: E = {e_ev:.5f} eV  ({e_kj.item():.3f} kJ/mol), "
        f"max|F| = {fmax_ev_a:.4f} eV/A"
    )
    # reference from the ASE calculator
    a = atoms.ase_converter()
    a.calc = AlignnAtomwiseCalculator(path=default_path())
    print(
        f"ASE calc: E = {a.get_potential_energy():.5f} eV, "
        f"max|F| = {abs(a.get_forces()).max():.4f} eV/A  (should match)"
    )


if __name__ == "__main__":
    main()
