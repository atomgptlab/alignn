"""Bundle multiple pretrained models behind one nn.Module.

Each entry is a (model, output_spec) pair. ``output_spec`` says how to map
the underlying model's raw output dict to a single tensor for that
property — e.g. EFS returns three tensors, formation energy returns one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)


@dataclass
class PropertySpec:
    """How to invoke one member model and what to extract from it."""

    name: str
    model: nn.Module
    # Map raw model output (whatever ``model(graph)`` returns) → dict of
    # named tensors for this property. Default pulls "out" from an
    # ALIGNN-Pure dict.
    extract: Callable[[object], Dict[str, torch.Tensor]] = field(
        default=lambda d: {"value": d["out"]}
    )


def _efs_extract(d):
    return {"energy": d["out"], "forces": d["grad"], "stress": d["stresses"]}


def _scalar_extract(d):
    return {"value": d["out"]}


class ModelZoo(nn.Module):
    """Registry of pretrained models exposed as one nn.Module.

    All members live in a single ``nn.ModuleDict`` so ``state_dict``,
    ``.to(device)``, ``torch.save`` etc. work on the whole collection.
    """

    def __init__(self, specs: Optional[List[PropertySpec]] = None):
        super().__init__()
        self._models = nn.ModuleDict()
        self._extractors: Dict[str, Callable] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: PropertySpec) -> None:
        if spec.name in self._models:
            raise KeyError(f"property {spec.name!r} already registered")
        self._models[spec.name] = spec.model
        self._extractors[spec.name] = spec.extract

    def properties(self) -> List[str]:
        return list(self._models.keys())

    def predict(self, graph, prop: str) -> Dict[str, torch.Tensor]:
        out = self._models[prop](graph)
        return self._extractors[prop](out)

    def predict_all(self, graph) -> Dict[str, Dict[str, torch.Tensor]]:
        return {p: self.predict(graph, p) for p in self._models}

    # default forward = predict_all so torch.compile sees one entry point
    def forward(self, graph) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.predict_all(graph)

    # ----- convenience constructors -----

    # ----- manifest-based lazy loading -----

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        map_location: str = "cpu",
        eager: bool = False,
    ) -> "LazyModelZoo":
        """Build a zoo whose members live on disk and load on first use.

        Manifest schema (JSON):
            {
              "base_dir": "optional; paths are resolved relative to this",
              "properties": {
                "<name>": {
                  "checkpoint": "<path>",
                  "config": { ... ALIGNNAtomWisePureConfig kwargs ... },
                  "is_efs": false
                },
                ...
              }
            }
        """
        with open(manifest_path) as f:
            spec = json.load(f)
        base = Path(spec.get("base_dir") or Path(manifest_path).parent)
        zoo = LazyModelZoo(map_location=map_location)
        for name, entry in spec["properties"].items():
            ckpt = base / entry["checkpoint"]
            cfg_kwargs = dict(entry.get("config") or {})
            cfg = ALIGNNAtomWisePureConfig(**cfg_kwargs)
            zoo.add_lazy(
                name=name,
                checkpoint=str(ckpt),
                config=cfg,
                is_efs=bool(entry.get("is_efs", False)),
            )
        if eager:
            for name in zoo.properties():
                zoo._materialize(name)
        return zoo

    @classmethod
    def from_model_dirs(
        cls,
        model_dirs: Dict[str, str],
        map_location: str = "cpu",
    ) -> "LazyModelZoo":
        """Build a lazy zoo from directories of config.json + best_model.pt.

        This is the layout produced by
        ``alignn.ff.ff.get_figshare_model_ff`` and by training runs, so
        figshare-downloaded property models plug in directly:

            zoo = ModelZoo.from_model_dirs(
                {"formation_energy_peratom": get_figshare_model_ff(
                    model_name="formation_energy_peratom")}
            )
        """
        try:  # pydantic v2 / v1
            fields = set(ALIGNNAtomWisePureConfig.model_fields)
        except AttributeError:
            fields = set(ALIGNNAtomWisePureConfig.__fields__)
        zoo = LazyModelZoo(map_location=map_location)
        for name, model_dir in model_dirs.items():
            model_dir = Path(model_dir)
            with open(model_dir / "config.json") as f:
                cfg_dict = json.load(f)
            mcfg = dict(cfg_dict.get("model", cfg_dict))
            is_efs = bool(mcfg.get("calculate_gradient")) and bool(
                mcfg.get("gradwise_weight", 0)
            )
            mcfg = {k: v for k, v in mcfg.items() if k in fields}
            mcfg["name"] = "alignn_atomwise_pure"
            zoo.add_lazy(
                name=name,
                checkpoint=str(model_dir / "best_model.pt"),
                config=ALIGNNAtomWisePureConfig(**mcfg),
                is_efs=is_efs,
            )
        return zoo

    @classmethod
    def from_alignn_checkpoints(
        cls,
        ckpts: Dict[str, str],
        configs: Optional[Dict[str, ALIGNNAtomWisePureConfig]] = None,
        efs_keys: Optional[List[str]] = None,
        map_location: str = "cpu",
    ) -> "ModelZoo":
        """Build a zoo from {property_name: checkpoint_path}.

        ``efs_keys`` lists names whose model returns energy/forces/stress
        (everything else is treated as a single scalar head).
        ``configs`` lets you pass a per-model config; missing entries fall
        back to the default ``ALIGNNAtomWisePureConfig()``.
        """
        configs = configs or {}
        efs = set(efs_keys or [])
        specs: List[PropertySpec] = []
        for name, path in ckpts.items():
            cfg = configs.get(name, ALIGNNAtomWisePureConfig(name=name))
            model = ALIGNNAtomWisePure(cfg)
            state = torch.load(path, map_location=map_location)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            model.eval()
            specs.append(
                PropertySpec(
                    name=name,
                    model=model,
                    extract=_efs_extract if name in efs else _scalar_extract,
                )
            )
        return cls(specs)


class LazyModelZoo(ModelZoo):
    """Zoo whose members are loaded from disk on first use.

    Only one ALIGNN trunk's weights occupy memory until you actually
    request a property. Materialized members are cached, so repeated
    calls hit RAM, not disk.
    """

    def __init__(self, map_location: str = "cpu"):
        super().__init__()
        self._map_location = map_location
        # name -> {checkpoint, config, is_efs}
        self._lazy: Dict[str, dict] = {}

    def add_lazy(
        self,
        name: str,
        checkpoint: str,
        config: ALIGNNAtomWisePureConfig,
        is_efs: bool = False,
    ) -> None:
        if name in self._lazy or name in self._models:
            raise KeyError(f"property {name!r} already registered")
        self._lazy[name] = {
            "checkpoint": checkpoint,
            "config": config,
            "is_efs": is_efs,
        }
        self._extractors[name] = _efs_extract if is_efs else _scalar_extract

    def properties(self) -> List[str]:
        return list(self._lazy.keys()) + [
            n for n in self._models.keys() if n not in self._lazy
        ]

    def _materialize(self, name: str) -> nn.Module:
        if name in self._models:
            return self._models[name]
        if name not in self._lazy:
            raise KeyError(name)
        entry = self._lazy[name]
        model = ALIGNNAtomWisePure(entry["config"])
        state = torch.load(
            entry["checkpoint"],
            map_location=self._map_location,
            weights_only=False,
        )
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[model_zoo:{name}] load_state_dict: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        model.eval()
        self._models[name] = model
        return model

    def predict(self, graph, prop: str) -> Dict[str, torch.Tensor]:
        self._materialize(prop)
        return super().predict(graph, prop)

    def evict(self, name: str) -> None:
        """Drop a materialized member back to disk-only state."""
        if name in self._models and name in self._lazy:
            del self._models[name]


# ----- example: registering a non-ALIGNN model (e.g. SlakoNet) ---------------
#
# from slakonet import SlakoNetModel
#
# zoo = ModelZoo.from_alignn_checkpoints(
#     {
#         "efs":               "ff/alignnff.pt",
#         "formation_energy":  "jv_formation_energy_peratom_alignn.pt",
#         "mbj_bandgap":       "jv_mbj_bandgap_alignn.pt",
#         "bulk_modulus":      "jv_bulk_modulus_kv_alignn.pt",
#     },
#     efs_keys=["efs"],
# )
#
# slakonet = SlakoNetModel.load("slakonet.pt").eval()
# zoo.register(PropertySpec(
#     name="bandstructure",
#     model=slakonet,
#     extract=lambda d: {"bandgap": d["bandgap"], "bands": d["bands"]},
# ))
#
# torch.save(zoo.state_dict(), "alignn_zoo.pt")
# preds = zoo.predict_all(graph)            # {prop: {field: tensor}}
# fe    = zoo.predict(graph, "formation_energy")["value"]
