"""Unified ALIGNN calculator driven by a pydantic config.

One calculator, one config object. The config selects which outputs are
produced:

  * force-field properties  : energy / forces / stress  (one ALIGNN-FF
    model)
  * scalar property predictors : formation energy, band gap, bulk
    modulus, ... (one pretrained ALIGNN regression model per property)

All models are loaded once and reused for every structure / every call
(no per-call reload).

Example
-------
    from alignn.ff.unified_calculator import (
        AlignnUnifiedCalculator, AlignnUnifiedConfig)
    from ase.build import bulk

    cfg = AlignnUnifiedConfig(
        energy=True, forces=True, stress=True,
        properties=["formation_energy_peratom",
                    "optb88vdw_bandgap", "bulk_modulus_kv"],
    )
    calc = AlignnUnifiedCalculator(cfg)

    si = bulk("Si", "diamond", a=5.43); si.calc = calc
    si.get_potential_energy(); si.get_forces(); si.get_stress()
    print(calc.results["formation_energy_peratom"],
          calc.results["optb88vdw_bandgap"],
          calc.results["bulk_modulus_kv"])
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from ase.calculators.calculator import Calculator, all_changes
from pydantic import BaseModel, Field, field_validator

from alignn.ff.calculators import (
    AlignnAtomwiseCalculator,
    ase_to_atoms,
)
from alignn.graphs import Graph
from alignn.pretrained import (
    get_alignn2_model,
    resolve_by_target,
    ALIGNN2_MODELS,
)
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)
import json as _json


def _prop2_name(friendly, graph):
    """Resolve a friendly property name to a pretrained (ALIGNN 2.0) model key, preferring the
    requested graph. Handles all three registry conventions: a direct model name
    (``elastic_tensor``), ``{name}_{graph}`` (``ir_radius``), and target lookup
    (``formation_energy_peratom`` -> ``..._radius``). Falls back to the other graph
    when only one variant exists (e.g. ``raman`` -> ``raman_knn``)."""
    if friendly in ALIGNN2_MODELS:  # direct model name
        return friendly
    other = "knn" if graph == "radius" else "radius"
    for g in (graph, other):  # {name}_{graph}
        if "{}_{}".format(friendly, g) in ALIGNN2_MODELS:
            return "{}_{}".format(friendly, g)
    cands = resolve_by_target(friendly)  # by training target
    if cands:
        pref = [m for m in cands if m.endswith("_" + graph)]
        return (pref or cands)[0]
    raise KeyError("No ALIGNN 2.0 property model for '{}'".format(friendly))


def _load_prop2_model(friendly, graph, device):
    """Load a pure-PyTorch ALIGNN 2.0 property model for `friendly`
    on the requested `graph` ("radius"/"knn"). Scalar, spectra (D>1) and tensor
    outputs are all supported. Returns a dict with the model and its own
    graph-construction settings (cutoff/max_neighbors/atom_features)."""
    name = _prop2_name(friendly, graph)
    paths = get_alignn2_model(name)
    cfg = _json.load(open(paths["config.json"]))
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**cfg["model"]))
    model.load_state_dict(
        torch.load(
            paths["best_model.pt"], map_location=device, weights_only=False
        )
    )
    model.to(device).eval()
    return {
        "model": model,
        "name": name,
        "cutoff": float(cfg.get("cutoff", 5.0)),
        "max_neighbors": int(cfg.get("max_neighbors", 12)),
        "atom_features": cfg.get("atom_features", "cgcnn"),
        "use_canonize": bool(cfg.get("use_canonize", False)),
    }


class AlignnUnifiedConfig(BaseModel):
    """Declarative spec of what the calculator should output."""

    # force-field model (energy/forces/stress source)
    ff_model: str = "matpes_r2scan"
    energy: bool = True
    forces: bool = True
    stress: bool = True
    charges: bool = False  # only if the FF model emits them

    # scalar property predictors to also evaluate (friendly names)
    properties: List[str] = Field(default_factory=list)

    # shared knobs
    device: Optional[str] = None
    # property-predictor graph: "radius" (default, FF-compatible) or "knn".
    # Uses the pure-PyTorch ALIGNN 2.0 models from the ALIGNN 2.0 registry; each carries its
    # own cutoff/max_neighbors from its training config.
    prop_graph: str = "radius"

    model_config = {"extra": "forbid"}

    @field_validator("properties")
    @classmethod
    def _check_props(cls, v: List[str]) -> List[str]:
        for p in v:
            try:  # scalar, spectra or tensor ALIGNN 2.0 model must exist
                _prop2_name(p, "radius")
            except KeyError as exc:
                raise ValueError(str(exc))
        return v

    @classmethod
    def from_file(cls, path: str) -> "AlignnUnifiedConfig":
        """Load from a JSON file."""
        import json

        with open(path) as fh:
            return cls(**json.load(fh))


class AlignnUnifiedCalculator(Calculator):
    """ASE calculator: ALIGNN-FF + selected ALIGNN property predictors.

    Args:
        config: an ``AlignnUnifiedConfig``, a dict, a path to a JSON
            config, or None (defaults: energy/forces/stress only).
        ff_path: optional local path/dir for the FF model (overrides
            ``config.ff_model``).
    """

    def __init__(self, config=None, ff_path=None, **kw):
        Calculator.__init__(self, **kw)

        if config is None:
            config = AlignnUnifiedConfig()
        elif isinstance(config, str):
            config = AlignnUnifiedConfig.from_file(config)
        elif isinstance(config, dict):
            config = AlignnUnifiedConfig(**config)
        self.cfg: AlignnUnifiedConfig = config

        self.device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        props = []
        if config.energy:
            props.append("energy")
        if config.forces:
            props.append("forces")
        if config.stress:
            props.append("stress")
        self.implemented_properties = props

        # --- force-field calculator (loaded ONCE) ---
        self._ff = None
        if config.energy or config.forces or config.stress:
            ff_kwargs = dict(include_stress=config.stress, device=self.device)
            if ff_path is not None:
                ff_kwargs["path"] = ff_path
            elif config.ff_model:
                ff_kwargs["path"] = _ff_model_path(config.ff_model)
            self._ff = AlignnAtomwiseCalculator(**ff_kwargs)

        # --- property predictor models (pure-torch ALIGNN 2.0, loaded ONCE) ---
        self._prop_models: Dict[str, dict] = {}
        for friendly in config.properties:
            self._prop_models[friendly] = _load_prop2_model(
                friendly, config.prop_graph, self.device
            )

    # -- scalar property forward (reuses cached model, no reload) -------
    def _predict_scalar(self, info, j_atoms) -> float:
        # pure-torch graph at the property model's own cutoff/neighbors
        g, lg = Graph.atom_dgl_multigraph(
            j_atoms,
            neighbor_strategy="pure_torch",
            cutoff=info["cutoff"],
            max_neighbors=info["max_neighbors"],
            atom_features=info["atom_features"],
            use_canonize=info["use_canonize"],
        )
        lat = torch.tensor(j_atoms.lattice_mat).type(torch.get_default_dtype())
        with torch.no_grad():
            out = info["model"](
                (
                    g.to(self.device),
                    lg.to(self.device),
                    lat.to(self.device),
                )
            )
        if isinstance(out, dict):
            out = out.get("out", out.get("energy", next(iter(out.values()))))
        arr = out.detach().cpu().numpy().flatten()
        return float(arr[0]) if arr.size == 1 else arr.tolist()

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {}

        # force-field block
        if self._ff is not None:
            self._ff.calculate(
                self.atoms,
                properties=properties,
                system_changes=system_changes,
            )
            if self.cfg.energy:
                self.results["energy"] = self._ff.results["energy"]
                self.results["free_energy"] = self._ff.results["energy"]
            if self.cfg.forces:
                self.results["forces"] = self._ff.results["forces"]
            if self.cfg.stress:
                self.results["stress"] = self._ff.results["stress"]
            if self.cfg.charges:
                ch = self._ff.results.get("charges")
                if ch is not None:
                    self.results["charges"] = ch
                else:
                    self.results.setdefault("warnings", []).append(
                        "charges requested but the FF model does not "
                        "emit them"
                    )

        # scalar property predictors
        if self._prop_models:
            j_atoms = ase_to_atoms(self.atoms)
            for friendly, info in self._prop_models.items():
                self.results[friendly] = self._predict_scalar(info, j_atoms)

    # convenience
    def predictions(self) -> Dict[str, object]:
        """All scalar property predictions from the last calculate()."""
        return {
            k: self.results[k]
            for k in self.cfg.properties
            if k in self.results
        }


def _ff_model_path(name_or_path: str) -> str:
    """Accept a local dir or a known FF figshare model name."""
    import os

    if os.path.isdir(name_or_path):
        return name_or_path
    from alignn.ff.calculators import get_figshare_model_ff

    return get_figshare_model_ff(model_name=name_or_path)
