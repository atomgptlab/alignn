"""Pretrained-model registry, stored as YAML.

- ``alignn_ff.yml``: force fields
- ``alignn_property.yml``: property predictors
- ``alignn_inverse.yml``: generative inverse-design models

Each file may hold the sections ``figshare_ff`` (name -> zip url, for
``get_figshare_model_ff``), ``figshare_prop`` (name -> zip url, for
``get_figshare_model_prop``) and ``alignn2`` (ALIGNN 2.0 entries grouped by
category, for ``alignn.pretrained.get_alignn2_model``). Sections with the same
name are merged across files.
"""

import os
from functools import lru_cache

import yaml

REGISTRY_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILES = ("alignn_ff.yml", "alignn_property.yml", "alignn_inverse.yml")


@lru_cache(maxsize=None)
def _load(fname):
    with open(os.path.join(REGISTRY_DIR, fname)) as f:
        return yaml.safe_load(f) or {}


def _section(section):
    out = {}
    for fname in REGISTRY_FILES:
        out.update(_load(fname).get(section) or {})
    return out


def figshare_ff_models():
    """Name -> url for models loaded with ``get_figshare_model_ff``."""
    return _section("figshare_ff")


def figshare_prop_models():
    """Name -> url for models loaded with ``get_figshare_model_prop``."""
    return _section("figshare_prop")


def alignn2_models():
    """Name -> metadata for ALIGNN 2.0 models, with ``category`` filled in."""
    out = {}
    for fname in REGISTRY_FILES:
        for category, models in (_load(fname).get("alignn2") or {}).items():
            for name, meta in models.items():
                out[name] = {"category": category, **meta}
    return out
