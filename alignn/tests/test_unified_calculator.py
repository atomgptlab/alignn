"""Tests for the unified ALIGNN calculator (FF + pure-torch property predictors).

Covers the force field (energy/forces/stress via matpes_r2scan default), a scalar
property (formation_energy_peratom), a spectrum (edos, D=300), a tensor
(elastic_tensor, D=36), the radius/knn graph switch, and config validation.
Models are downloaded+cached on first use (like the other pretrained tests).
"""

import math

import numpy as np
from ase.build import bulk

from alignn.ff.unified_calculator import (
    AlignnUnifiedCalculator,
    AlignnUnifiedConfig,
    _prop2_name,
)


def _finite(x):
    return all(math.isfinite(v) for v in np.asarray(x).ravel().tolist())


def test_unified_ff_scalar_spectrum_tensor():
    """One calculator returning FF + scalar + spectrum + tensor for Si."""
    cfg = AlignnUnifiedConfig(
        energy=True,
        forces=True,
        stress=True,
        properties=[
            "formation_energy_peratom",  # scalar
            "edos",  # spectrum, D=300
            "elastic_tensor",  # tensor, D=36
        ],
    )
    calc = AlignnUnifiedCalculator(cfg)

    si = bulk("Si", "diamond", a=5.43)
    si.calc = calc

    energy = si.get_potential_energy()
    forces = si.get_forces()
    stress = si.get_stress()
    assert math.isfinite(energy)
    assert forces.shape == (len(si), 3) and _finite(forces)
    assert stress.shape == (6,) and _finite(stress)
    # relaxed diamond Si: forces are ~zero by symmetry
    assert np.abs(forces).max() < 1e-3

    preds = calc.predictions()
    # scalar: elemental Si formation energy is ~0
    fe = preds["formation_energy_peratom"]
    assert isinstance(fe, float) and abs(fe) < 0.5
    # spectrum
    assert isinstance(preds["edos"], list) and len(preds["edos"]) == 300
    assert _finite(preds["edos"])
    # tensor (6x6 elastic flattened); C11 in a physical range for Si
    et = preds["elastic_tensor"]
    assert isinstance(et, list) and len(et) == 36 and _finite(et)
    assert 80.0 < et[0] < 260.0  # C11 ~ 160 GPa


def test_unified_ff_only_default_model():
    """FF-only path uses the pure-torch matpes_r2scan default (no dgl)."""
    cfg = (
        AlignnUnifiedConfig()
    )  # defaults: energy/forces/stress, no properties
    assert cfg.ff_model == "matpes_r2scan"
    calc = AlignnUnifiedCalculator(cfg)
    si = bulk("Si", "diamond", a=5.43)
    si.calc = calc
    assert math.isfinite(si.get_potential_energy())
    assert calc.predictions() == {}


def test_prop_name_resolution_radius_and_knn():
    """Name resolver handles direct names, {name}_{graph}, and graph fallback."""
    assert (
        _prop2_name("formation_energy_peratom", "radius")
        == "formation_energy_peratom_radius"
    )
    assert (
        _prop2_name("formation_energy_peratom", "knn")
        == "formation_energy_peratom_knn"
    )
    assert (
        _prop2_name("elastic_tensor", "radius") == "elastic_tensor"
    )  # direct
    assert _prop2_name("ir", "knn") == "ir_knn"  # {name}_{graph}
    assert _prop2_name("raman", "radius") == "raman_knn"  # knn-only fallback


def test_unified_knn_switch():
    """prop_graph='knn' loads the knn property variant."""
    cfg = AlignnUnifiedConfig(
        energy=True,
        forces=True,
        stress=True,
        prop_graph="knn",
        properties=["formation_energy_peratom"],
    )
    calc = AlignnUnifiedCalculator(cfg)
    assert (
        calc._prop_models["formation_energy_peratom"]["name"]
        == "formation_energy_peratom_knn"
    )
    si = bulk("Si", "diamond", a=5.43)
    si.calc = calc
    si.get_potential_energy()
    assert math.isfinite(calc.predictions()["formation_energy_peratom"])


def test_unified_unknown_property_raises():
    """An unknown property is rejected at config time."""
    try:
        AlignnUnifiedConfig(properties=["not_a_real_property"])
    except Exception as exc:  # pydantic ValidationError wraps the ValueError
        assert "not_a_real_property" in str(exc)
    else:
        raise AssertionError("expected validation error for unknown property")


if __name__ == "__main__":
    test_prop_name_resolution_radius_and_knn()
    test_unified_unknown_property_raises()
    test_unified_ff_only_default_model()
    test_unified_ff_scalar_spectrum_tensor()
    test_unified_knn_switch()
    print("all unified calculator tests passed")
