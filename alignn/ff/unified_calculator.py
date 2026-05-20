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

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from pydantic import BaseModel, Field, field_validator

from alignn.ff.calculators import (
    AlignnAtomwiseCalculator,
    ase_to_atoms,
)
from alignn.graphs import Graph
from alignn.pretrained import get_figshare_model

# friendly name -> figshare model name (extend freely; raw figshare
# names ending in "_alignn" are also accepted as-is)
PROP_ALIASES: Dict[str, str] = {
    "formation_energy_peratom": "jv_formation_energy_peratom_alignn",
    "total_energy": "jv_optb88vdw_total_energy_alignn",
    "optb88vdw_bandgap": "jv_optb88vdw_bandgap_alignn",
    "mbj_bandgap": "jv_mbj_bandgap_alignn",
    "bulk_modulus_kv": "jv_bulk_modulus_kv_alignn",
    "shear_modulus_gv": "jv_shear_modulus_gv_alignn",
    "ehull": "jv_ehull_alignn",
    "spillage": "jv_spillage_alignn",
    "slme": "jv_slme_alignn",
    "magmom_oszicar": "jv_magmom_oszicar_alignn",
    "exfoliation_energy": "jv_exfoliation_energy_alignn",
    "supercon_tc": "jv_supercon_tc_alignn",
    "epsx": "jv_epsx_alignn",
    "n_seebeck": "jv_n-Seebeck_alignn",
    "n_powerfact": "jv_n-powerfact_alignn",
}


def _resolve_prop(name: str) -> str:
    """friendly or raw -> figshare model name."""
    if name in PROP_ALIASES:
        return PROP_ALIASES[name]
    if name.endswith("_alignn"):  # raw figshare name passed through
        return name
    raise ValueError(
        f"Unknown property '{name}'. Known: "
        f"{sorted(PROP_ALIASES)} (or a raw *_alignn figshare name)."
    )


class AlignnUnifiedConfig(BaseModel):
    """Declarative spec of what the calculator should output."""

    # force-field model (energy/forces/stress source)
    ff_model: str = "v12.2.2024_dft_3d_307k"
    energy: bool = True
    forces: bool = True
    stress: bool = True
    charges: bool = False  # only if the FF model emits them

    # scalar property predictors to also evaluate (friendly names)
    properties: List[str] = Field(default_factory=list)

    # shared knobs
    device: Optional[str] = None
    prop_cutoff: float = 8.0
    prop_max_neighbors: int = 12

    model_config = {"extra": "forbid"}

    @field_validator("properties")
    @classmethod
    def _check_props(cls, v: List[str]) -> List[str]:
        for p in v:
            _resolve_prop(p)  # raises on unknown
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
            ff_kwargs = dict(
                include_stress=config.stress, device=self.device
            )
            if ff_path is not None:
                ff_kwargs["path"] = ff_path
            elif config.ff_model:
                ff_kwargs["path"] = _ff_model_path(config.ff_model)
            self._ff = AlignnAtomwiseCalculator(**ff_kwargs)

        # --- property predictor models (each loaded ONCE) ---
        self._prop_models: Dict[str, object] = {}
        for friendly in config.properties:
            self._prop_models[friendly] = get_figshare_model(
                _resolve_prop(friendly)
            )

    # -- scalar property forward (reuses cached model, no reload) -------
    def _predict_scalar(self, model, j_atoms) -> float:
        g, lg = Graph.atom_dgl_multigraph(
            j_atoms,
            cutoff=float(self.cfg.prop_cutoff),
            max_neighbors=self.cfg.prop_max_neighbors,
        )
        lat = torch.tensor(j_atoms.lattice_mat)
        with torch.no_grad():
            out = model(
                [
                    g.to(self.device),
                    lg.to(self.device),
                    lat.to(self.device),
                ]
            )
        if isinstance(out, dict):
            out = out["out"]
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
                self.atoms, properties=properties,
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
            for friendly, model in self._prop_models.items():
                self.results[friendly] = self._predict_scalar(
                    model, j_atoms
                )

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
