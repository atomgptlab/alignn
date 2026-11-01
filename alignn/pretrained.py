"""ALIGNN 2.0 pretrained-model registry (glossary).

Analogous to :mod:`jarvis.db.figshare`: a single dict maps a model name to its
Figshare artifacts and metadata, and :func:`get_alignn2_model` downloads
(and caches) the ``config.json``, ``best_model.pt``, and
``ids_train_val_test.json`` for that model.

All models live in the Figshare **ALIGNN2** project
(https://figshare.com/projects/ALIGNN2, id ``279395``). Each variant (e.g.
``formation_energy_peratom_radius`` and ``..._knn``) is its OWN article holding
ONE flat zip (best_model.pt + config.json + ids_train_val_test.json) with its
own ``url`` (``https://ndownloader.figshare.com/files/<id>``) -- mirroring the
per-model download URLs in the original ``alignn/pretrained.py``.

Note: Figshare *draft* files are not publicly downloadable; the ``url`` loader
activates once the article is published (token needed meanwhile).
"""

import os
import zipfile
import requests
from alignn.cache import check_model_cache
from alignn.registry import alignn2_models

FIGSHARE_PROJECT_ID = 279395
FIGSHARE_PROJECT_URL = "https://figshare.com/projects/ALIGNN2/{}".format(
    FIGSHARE_PROJECT_ID
)

# --- registry -----------------------------------------------------------------
# name -> metadata, loaded from alignn/registry/*.yml
ALIGNN2_MODELS = alignn2_models()

_API = "https://api.figshare.com/v2"


def list_alignn2_models(category=None):
    """Return the registry, optionally filtered by category."""
    return {
        k: v
        for k, v in ALIGNN2_MODELS.items()
        if category is None or v["category"] == category
    }


def resolve_by_target(target, category=None):
    """Return registry keys whose ``target`` matches (e.g. a property name)."""
    return [
        k
        for k, v in ALIGNN2_MODELS.items()
        if v.get("target") == target
        and (category is None or v["category"] == category)
    ]


def _cache_dir():
    d = os.path.join(os.path.expanduser("~"), ".alignn2_models")
    os.makedirs(d, exist_ok=True)
    return d


def get_alignn2_model(name, download=True, cache_dir=None):
    """Return local paths to a model's artifacts, downloading+caching on first use.

    Downloads the variant's single flat Figshare zip (``config.json``,
    ``best_model.pt``, ``ids_train_val_test.json``) via its ``url`` and extracts
    it to ``~/.alignn2_models/<name>/``.
    """
    if name not in ALIGNN2_MODELS:
        raise KeyError(
            "Unknown model '{}'. Available: {}".format(
                name, sorted(ALIGNN2_MODELS)
            )
        )
    check_model_cache()
    meta = ALIGNN2_MODELS[name]
    dest = os.path.join(cache_dir or _cache_dir(), name)
    triplet = ["config.json", "best_model.pt", "ids_train_val_test.json"]
    have = os.path.exists(
        os.path.join(dest, "best_model.pt")
    ) and os.path.exists(os.path.join(dest, "config.json"))
    if download and not have:
        os.makedirs(dest, exist_ok=True)
        url = meta.get("url")
        if not url:  # unmigrated fallback: fetch the article's zip file url
            r = requests.get(
                "{}/articles/{}".format(_API, meta["figshare_article_id"])
            )
            r.raise_for_status()
            files = {
                f["name"]: f["download_url"] for f in r.json().get("files", [])
            }
            url = files.get(meta.get("zip")) or next(iter(files.values()))
        print("Obtaining model", name, "from", url)
        r = requests.get(url, stream=True)
        r.raise_for_status()
        zp = os.path.join(dest, "_model.zip")
        with open(zp, "wb") as o:
            for c in r.iter_content(1 << 20):
                o.write(c)
        with zipfile.ZipFile(zp) as z:
            members = z.namelist()
            sub = meta.get("subdir", "")
            for base in triplet + ["multi_out_predictions.json"]:
                hit = [
                    n for n in members if n == base or n.endswith("/" + base)
                ]
                if sub:
                    hit = [
                        n for n in hit if ("%s/%s" % (sub, base)) in n
                    ] or hit
                if hit:
                    with z.open(hit[0]) as fsrc, open(
                        os.path.join(dest, base), "wb"
                    ) as fdst:
                        fdst.write(fsrc.read())
        os.remove(zp)
    return {
        f: os.path.join(dest, f)
        for f in triplet
        if os.path.exists(os.path.join(dest, f))
    }


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------
# The legacy DGL-based helpers (``get_figshare_model``,
# ``get_multiple_predictions``, ``get_prediction`` ...) moved to
# ``alignn.deprecated.pretrained`` when this module was renamed from
# ``pretrained2``. They are re-exported here so existing imports of the form
# ``from alignn.pretrained import get_figshare_model`` keep working. Guarded
# because the legacy path requires DGL, which the pure-PyTorch ALIGNN 2.0
# models above do not.
try:  # pragma: no cover - optional DGL dependency
    from alignn.deprecated.pretrained import (  # noqa: F401
        get_all_models,
        get_figshare_model,
        get_figshare_model_pure,
        load_pure_torch_model,
        get_prediction,
        get_prediction_pure,
        get_multiple_predictions,
    )
except Exception:  # DGL not installed / legacy import unavailable
    pass
