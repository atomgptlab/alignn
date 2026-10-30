"""Conditioning encoders for ALIGNN-CSP.

Inverse design is only as useful as the range of things you can condition on,
so the generator does not hard-code any particular target.  Every modality is
a :class:`Conditioner` that maps its raw input to one ``(B, hidden)`` vector;
:class:`MultiModalConditioner` sums the active ones and hands the result to
the denoiser, which never learns what the conditioning actually was.

Built-in modalities:

``scalar``       a single number — Tc, band gap, formation energy, ...
``composition``  a 118-long element-fraction (or count) vector
``vector``       a fixed-length 1-D signal — an XRD/diffraction pattern, a
                 DOS curve, an absorption spectrum
``image``        a 2-D map — a STEM/HAADF micrograph, a simulated diffraction
                 image

Each conditioner owns a learned *null* embedding, and each is dropped
independently during training.  That is what lets you guide on any subset at
sampling time: train once with composition+Tc+XRD, then generate from XRD
alone, or from Tc alone, without retraining.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


def sinusoidal_embedding(
    x: torch.Tensor, dim: int, max_period: float = 1e4
) -> torch.Tensor:
    """Sinusoidal embedding of a scalar, as used for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=x.device)
        / half
    )
    args = x.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Conditioner(nn.Module):
    """Base class: encode one modality to ``(B, hidden_features)``.

    Subclasses implement :meth:`encode`.  The base class handles the learned
    null embedding and per-sample masking used for classifier-free guidance.
    """

    def __init__(self, hidden_features: int):
        super().__init__()
        self.hidden_features = hidden_features
        self.null_embedding = nn.Parameter(torch.zeros(hidden_features))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, value: Optional[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``value``, substituting the null embedding where masked out.

        ``mask`` is ``(B,)`` with 1 = use the conditioning, 0 = drop it.
        ``value`` may be ``None``, meaning the modality is absent entirely.
        """
        null = self.null_embedding.view(1, -1).expand(mask.shape[0], -1)
        if value is None:
            return null
        emb = self.encode(value)
        return torch.where(mask.view(-1, 1).bool(), emb, null)


class ScalarConditioner(Conditioner):
    """A single continuous property (Tc, band gap, formation energy, ...).

    Values are standardised with statistics supplied at construction so the
    sinusoidal embedding sees a well-scaled input.
    """

    def __init__(
        self,
        hidden_features: int,
        embedding_features: int = 128,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__(hidden_features)
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(max(float(std), 1e-6)))
        self.embedding_features = embedding_features
        self.mlp = nn.Sequential(
            nn.Linear(embedding_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        z = (value.float().view(-1) - self.mean) / self.std
        return self.mlp(sinusoidal_embedding(z, self.embedding_features))


class CompositionConditioner(Conditioner):
    """A 118-long element vector (fractions or counts).

    Note this is *soft* conditioning, separate from the hard constraint of
    fixing the atom types on the graph nodes.  It is what you use when the
    composition should steer generation without being nailed down, or when a
    downstream mode (XRD-only inference) has no explicit atom list.
    """

    def __init__(
        self,
        hidden_features: int,
        num_elements: int = 118,
        normalize: bool = True,
    ):
        super().__init__(hidden_features)
        self.num_elements = num_elements
        self.normalize = normalize
        self.mlp = nn.Sequential(
            nn.Linear(num_elements, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        v = value.float()
        if self.normalize:
            v = v / v.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.mlp(v)

    @staticmethod
    def from_atomic_numbers(
        atomic_numbers: torch.Tensor,
        node_graph_id: torch.Tensor,
        n_graphs: int,
        num_elements: int = 118,
    ) -> torch.Tensor:
        """Build the element-count vector for a flattened batch."""
        out = torch.zeros(n_graphs, num_elements, device=atomic_numbers.device)
        idx = atomic_numbers.clamp(1, num_elements) - 1
        out.index_put_(
            (node_graph_id, idx),
            torch.ones_like(idx, dtype=out.dtype),
            accumulate=True,
        )
        return out


class VectorConditioner(Conditioner):
    """A fixed-length 1-D signal: XRD pattern, DOS, spectrum.

    A small dilated 1-D CNN rather than a flat MLP: diffraction information
    lives in *peak positions and spacings*, which convolutions with growing
    receptive field pick up far more readily than a dense layer over raw bins.
    """

    def __init__(
        self,
        hidden_features: int,
        input_length: int,
        channels: int = 64,
        num_layers: int = 4,
        log1p: bool = True,
    ):
        super().__init__(hidden_features)
        self.input_length = input_length
        self.log1p = log1p
        layers: List[nn.Module] = []
        in_ch = 1
        for i in range(num_layers):
            layers += [
                nn.Conv1d(
                    in_ch,
                    channels,
                    kernel_size=5,
                    padding=2 * (2**i),
                    dilation=2**i,
                ),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        # Mean+max pooling: mean carries the overall profile, max carries the
        # sharp Bragg peaks that mean-pooling would wash out.
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        v = value.float()
        if v.dim() == 2:
            v = v.unsqueeze(1)  # (B, 1, L)
        if self.log1p:
            v = torch.log1p(v.clamp_min(0.0))
        # Per-pattern scaling: absolute intensities are not meaningful.
        v = v / v.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        h = self.conv(v)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.head(pooled)


class ImageConditioner(Conditioner):
    """A 2-D map: STEM/HAADF micrograph, simulated diffraction image."""

    def __init__(
        self,
        hidden_features: int,
        in_channels: int = 1,
        channels: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__(hidden_features)
        layers: List[nn.Module] = []
        in_ch = in_channels
        ch = channels
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, ch),
                nn.SiLU(),
            ]
            in_ch = ch
            ch = min(ch * 2, 256)
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * in_ch, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        v = value.float()
        if v.dim() == 3:
            v = v.unsqueeze(1)  # (B, 1, H, W)
        # Per-image standardisation: detector gain and exposure vary.
        mu = v.mean(dim=(-1, -2), keepdim=True)
        sd = v.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        v = (v - mu) / sd
        h = self.conv(v)
        pooled = torch.cat(
            [h.mean(dim=(-1, -2)), h.amax(dim=-1).amax(dim=-1)], dim=-1
        )
        return self.head(pooled)


class MultiModalConditioner(nn.Module):
    """Combine any subset of modalities into one conditioning vector.

    Modalities are summed rather than concatenated so that a model trained
    with several of them still works when only some are supplied — the
    missing ones contribute their null embedding instead of shifting every
    downstream feature index.
    """

    def __init__(self, conditioners: Dict[str, Conditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        hidden = {c.hidden_features for c in conditioners.values()}
        assert (
            len(hidden) <= 1
        ), f"all conditioners must share hidden_features, got {hidden}"
        self.hidden_features = hidden.pop() if hidden else 0

    @property
    def names(self) -> List[str]:
        return list(self.conditioners.keys())

    def forward(
        self,
        values: Dict[str, Optional[torch.Tensor]],
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = None
        for name, cond in self.conditioners.items():
            emb = cond(values.get(name), masks[name])
            total = emb if total is None else total + emb
        return total

    def sample_masks(
        self,
        batch_size: int,
        device,
        dropout: Dict[str, float],
        available: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Draw independent per-modality keep-masks for CFG training."""
        avail = set(self.names if available is None else available)
        masks = {}
        for name in self.names:
            if name not in avail:
                masks[name] = torch.zeros(batch_size, device=device)
                continue
            p = float(dropout.get(name, 0.0))
            masks[name] = (torch.rand(batch_size, device=device) >= p).float()
        return masks

    def full_masks(
        self, batch_size: int, device, active: Optional[Iterable[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """All-on masks (or on only for ``active``), for inference."""
        act = set(self.names if active is None else active)
        return {
            name: (
                torch.ones(batch_size, device=device)
                if name in act
                else torch.zeros(batch_size, device=device)
            )
            for name in self.names
        }

    def zero_masks(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        """All-off masks: the unconditional branch of guidance."""
        return {
            name: torch.zeros(batch_size, device=device) for name in self.names
        }


def build_conditioner(
    spec: Dict[str, Dict], hidden_features: int
) -> MultiModalConditioner:
    """Build a :class:`MultiModalConditioner` from a plain config dict.

    ``spec`` maps a modality name to its kwargs plus a ``type`` key, e.g.::

        {
          "Tc":   {"type": "scalar", "mean": 3.68, "std": 4.75},
          "comp": {"type": "composition"},
          "xrd":  {"type": "vector", "input_length": 1000},
          "stem": {"type": "image"},
        }
    """
    registry = {
        "scalar": ScalarConditioner,
        "composition": CompositionConditioner,
        "vector": VectorConditioner,
        "image": ImageConditioner,
    }
    built: Dict[str, Conditioner] = {}
    for name, cfg in spec.items():
        cfg = dict(cfg)
        kind = cfg.pop("type")
        assert kind in registry, (
            f"unknown conditioner type {kind!r} for {name!r}; "
            f"expected one of {sorted(registry)}"
        )
        built[name] = registry[kind](hidden_features=hidden_features, **cfg)
    return MultiModalConditioner(built)
