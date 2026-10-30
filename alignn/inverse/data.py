"""Dataset and batching for ALIGNN-CSP.

Reads the split JSONs written by ``scripts/atombench/prepare_data.py`` and
produces flattened (all crystals concatenated along the atom axis) batches,
which is the layout the pure-torch ALIGNN scatter primitives expect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from alignn.inverse.diffusion import lattice_to_vec6, wrap_frac


@dataclass
class Normalizer:
    """Standardisation statistics for the lattice 6-vector and property."""

    lattice_mean: torch.Tensor
    lattice_std: torch.Tensor
    prop_mean: float
    prop_std: float

    def to_dict(self) -> Dict:
        return {
            "lattice_mean": self.lattice_mean.tolist(),
            "lattice_std": self.lattice_std.tolist(),
            "prop_mean": self.prop_mean,
            "prop_std": self.prop_std,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        return cls(
            lattice_mean=torch.tensor(d["lattice_mean"], dtype=torch.float32),
            lattice_std=torch.tensor(d["lattice_std"], dtype=torch.float32),
            prop_mean=float(d["prop_mean"]),
            prop_std=float(d["prop_std"]),
        )

    def to(self, device):
        self.lattice_mean = self.lattice_mean.to(device)
        self.lattice_std = self.lattice_std.to(device)
        return self

    def norm_lattice(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.lattice_mean) / self.lattice_std

    def denorm_lattice(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.lattice_std + self.lattice_mean

    def norm_prop(self, p: torch.Tensor) -> torch.Tensor:
        return (p - self.prop_mean) / self.prop_std


def _signed_permutations() -> torch.Tensor:
    """The 48 signed 3x3 permutation matrices."""
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = torch.zeros(3, 3)
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            mats.append(m)
    return torch.stack(mats)


_SIGNED_PERMS = _signed_permutations()


class CrystalDataset(Dataset):
    """Crystals from one AtomBench split JSON.

    Parameters
    ----------
    augment : bool
        Relabel each crystal with a random signed permutation of its lattice
        basis.  The physical crystal is untouched — Cartesian positions,
        distances and angles are all identical — but the fractional
        coordinates and the lattice 6-vector the model has to predict change.
        With only a few hundred training crystals this is the one augmentation
        that adds real signal: it teaches the model the basis-relabelling
        equivariance it does not have built in, instead of letting it memorise
        one arbitrary cell setting per compound.
    """

    def __init__(self, path: str | Path, augment: bool = False):
        self.records: List[Dict] = json.loads(Path(path).read_text())
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _apply_augmentation(self, lattice, frac):
        # L' = P L keeps cart = frac' L' equal to frac L when frac' = frac P^T.
        p = _SIGNED_PERMS[int(torch.randint(len(_SIGNED_PERMS), (1,)))]
        return p @ lattice, wrap_frac(frac @ p.t())

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        lattice = torch.tensor(r["lattice_mat"], dtype=torch.float32)
        frac = wrap_frac(torch.tensor(r["frac_coords"], dtype=torch.float32))
        if self.augment:
            lattice, frac = self._apply_augmentation(lattice, frac)
        return {
            "material_id": r["material_id"],
            "formula": r["formula"],
            "lattice": lattice,
            "frac": frac,
            "atomic_numbers": torch.tensor(
                r["atomic_numbers"], dtype=torch.long
            ),
            "prop": torch.tensor(float(r["target"]), dtype=torch.float32),
        }


def collate(samples: List[Dict]) -> Dict:
    """Flatten a list of crystals into one batch."""
    natoms = torch.tensor([s["frac"].shape[0] for s in samples])
    return {
        "material_id": [s["material_id"] for s in samples],
        "formula": [s["formula"] for s in samples],
        "lattice": torch.stack([s["lattice"] for s in samples]),
        "frac": torch.cat([s["frac"] for s in samples]),
        "atomic_numbers": torch.cat([s["atomic_numbers"] for s in samples]),
        "prop": torch.stack([s["prop"] for s in samples]),
        "natoms": natoms,
        "node_graph_id": torch.repeat_interleave(
            torch.arange(len(samples)), natoms
        ),
    }


def batch_to(batch: Dict, device) -> Dict:
    out = dict(batch)
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def compute_normalizer(
    dataset: CrystalDataset,
    clamp_std: float = 1e-3,
    symmetrize: bool = True,
) -> Normalizer:
    """Fit standardisation stats on a (training) split.

    ``symmetrize`` imposes the statistics that basis-permutation augmentation
    implies exactly, instead of estimating them from random draws: under a
    signed permutation ``S -> P S P^T`` the three diagonal components of the
    log-lattice vector are exchangeable and the three off-diagonals are
    zero-mean.  Pooling them keeps the normaliser deterministic and matches
    the distribution the model actually trains on.
    """
    plain = CrystalDataset.__new__(CrystalDataset)
    plain.records = dataset.records
    plain.augment = False

    vecs, props = [], []
    for i in range(len(plain)):
        s = plain[i]
        n = torch.tensor([s["frac"].shape[0]])
        vecs.append(lattice_to_vec6(s["lattice"].unsqueeze(0), n)[0])
        props.append(float(s["prop"]))
    v = torch.stack(vecs)
    mean, std = v.mean(0), v.std(0)

    if symmetrize:
        diag_mean = mean[:3].mean()
        diag_std = v[:, :3].reshape(-1).std()
        # Off-diagonals are symmetric about zero once signs are randomised;
        # their spread is what matters, so pool it across the three.
        off_std = v[:, 3:].reshape(-1).std()
        mean = torch.cat([diag_mean.repeat(3), torch.zeros(3)])
        std = torch.cat([diag_std.repeat(3), off_std.repeat(3)])

    return Normalizer(
        lattice_mean=mean,
        lattice_std=std.clamp_min(clamp_std),
        prop_mean=float(np.mean(props)),
        prop_std=float(max(np.std(props), 1e-6)),
    )


def compositions_from_split(
    path: str | Path,
) -> List[Dict]:
    """Conditioning inputs for generation: composition + property + id.

    The benchmark's stoichiometry-conditioned track gives the model the target
    composition, so generation is crystal *structure* prediction: place the
    known atoms in the unknown cell.
    """
    records = json.loads(Path(path).read_text())
    return [
        {
            "material_id": r["material_id"],
            "formula": r["formula"],
            "atomic_numbers": [int(z) for z in r["atomic_numbers"]],
            "prop": float(r["target"]),
            "target_poscar": r["target_poscar"],
        }
        for r in records
    ]


def make_generation_batch(
    items: List[Dict], device, prop_override: Optional[float] = None
) -> Dict:
    """Build a conditioning-only batch (no ground-truth geometry)."""
    natoms = torch.tensor([len(it["atomic_numbers"]) for it in items])
    props = torch.tensor(
        [
            it["prop"] if prop_override is None else prop_override
            for it in items
        ],
        dtype=torch.float32,
    )
    return {
        "material_id": [it["material_id"] for it in items],
        "atomic_numbers": torch.cat(
            [
                torch.tensor(it["atomic_numbers"], dtype=torch.long)
                for it in items
            ]
        ).to(device),
        "natoms": natoms.to(device),
        "node_graph_id": torch.repeat_interleave(
            torch.arange(len(items)), natoms
        ).to(device),
        "prop": props.to(device),
    }
