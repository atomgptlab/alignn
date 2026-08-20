"""The ALIGNN-CSP generator: conditioning encoders + ALIGNN denoiser."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from alignn.inverse.conditioners import (
    CompositionConditioner,
    MultiModalConditioner,
    build_conditioner,
)
from alignn.inverse.denoiser import ALIGNNCSPDenoiser

# Modality name reserved for the element-count vector, which is derived from
# the batch rather than read from the dataset record.
COMPOSITION_KEY = "composition"


class ALIGNNCSP(nn.Module):
    """Conditional diffusion generator for crystal structures.

    Parameters
    ----------
    denoiser_config : dict
        Keyword arguments for :class:`ALIGNNCSPDenoiser`.
    conditioner_spec : dict
        Modality name -> config, see
        :func:`alignn.inverse.conditioners.build_conditioner`.  May be empty
        for an unconditional model.
    """

    def __init__(
        self,
        denoiser_config: Optional[Dict] = None,
        conditioner_spec: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        denoiser_config = dict(denoiser_config or {})
        self.denoiser = ALIGNNCSPDenoiser(**denoiser_config)
        self.conditioner_spec = dict(conditioner_spec or {})
        self.conditioner = build_conditioner(
            self.conditioner_spec, self.denoiser.hidden_features
        )
        self.denoiser_config = denoiser_config

    @property
    def modalities(self):
        return self.conditioner.names

    def encode_conditioning(
        self,
        cond_values: Dict[str, Optional[torch.Tensor]],
        cond_masks: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.conditioner.names:
            return None
        return self.conditioner(cond_values, cond_masks)

    def forward(
        self,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        lattice_vec6: torch.Tensor,
        atomic_numbers: torch.Tensor,
        natoms: torch.Tensor,
        node_graph_id: torch.Tensor,
        t: torch.Tensor,
        cond_values: Optional[Dict] = None,
        cond_masks: Optional[Dict] = None,
        pair_index=None,
    ) -> Dict[str, torch.Tensor]:
        cond_emb = None
        if cond_masks is not None:
            cond_emb = self.encode_conditioning(cond_values or {}, cond_masks)
        return self.denoiser(
            frac=frac,
            lattice=lattice,
            lattice_vec6=lattice_vec6,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            node_graph_id=node_graph_id,
            t=t,
            cond_embedding=cond_emb,
            pair_index=pair_index,
        )


def build_cond_values(
    batch: Dict,
    conditioner: MultiModalConditioner,
    num_elements: int = 118,
) -> Dict[str, Optional[torch.Tensor]]:
    """Assemble the conditioning inputs the model expects from a batch.

    Every modality is looked up by name in ``batch``, except ``composition``,
    which is derived from the atom types already present.  A modality missing
    from the batch yields ``None``, which the conditioner turns into its null
    embedding — so a partially-labelled dataset trains without special casing.
    """
    n_graphs = int(batch["natoms"].shape[0])
    values: Dict[str, Optional[torch.Tensor]] = {}
    for name in conditioner.names:
        if name == COMPOSITION_KEY:
            values[name] = CompositionConditioner.from_atomic_numbers(
                batch["atomic_numbers"],
                batch["node_graph_id"],
                n_graphs,
                num_elements=num_elements,
            )
        else:
            values[name] = batch.get(name)
    return values


def available_modalities(
    batch: Dict, conditioner: MultiModalConditioner
) -> list:
    """Names of modalities actually present for this batch."""
    return [
        n
        for n in conditioner.names
        if n == COMPOSITION_KEY or batch.get(n) is not None
    ]
