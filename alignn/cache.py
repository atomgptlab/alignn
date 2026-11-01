"""Invalidate cached pretrained models downloaded by older alignn versions."""

import glob
import os
import shutil
import warnings

from jarvis.core.utils import get_cache_dir

# Caches written before this version may hold models that were since
# replaced under the same name (e.g. matpes_pbe, matpes_r2scan, mptrj_ff).
# Raise it whenever a model name is repointed to a new file.
MIN_CACHE_VERSION = "2026.10.15"
_MARKER = "alignn_cache_version"
# ALIGNN 2.0 registry entries (alignn.pretrained) repointed to new models
_ALIGNN2_STALE = ["matpes_pbe_ff", "matpes_r2scan_ff", "mptrj_ff"]
_checked = False


def _vtuple(v):
    try:
        return tuple(int(x) for x in str(v).strip().split("."))
    except ValueError:
        return (0,)


def check_model_cache():
    """Delete stale alignn model caches once, warning the user."""
    global _checked
    if _checked:
        return
    _checked = True
    base = get_cache_dir()
    marker = os.path.join(base, _MARKER)
    cached = None
    if os.path.exists(marker):
        with open(marker) as f:
            cached = f.read()
    if cached is not None and _vtuple(cached) >= _vtuple(MIN_CACHE_VERSION):
        return
    stale = sorted(glob.glob(os.path.join(base, "alignn*")))
    a2 = os.path.join(os.path.expanduser("~"), ".alignn2_models")
    stale += [
        os.path.join(a2, n)
        for n in _ALIGNN2_STALE
        if os.path.isdir(os.path.join(a2, n))
    ]
    stale = [p for p in stale if os.path.isdir(p)]
    if stale:
        warnings.warn(
            "alignn: removing model caches from version {} (< {}), "
            "since some model names now point to new models. They will be "
            "re-downloaded on use: {}".format(
                cached.strip() if cached else "unknown",
                MIN_CACHE_VERSION,
                ", ".join(stale),
            )
        )
        for p in stale:
            shutil.rmtree(p, ignore_errors=True)
    with open(marker, "w") as f:
        f.write(MIN_CACHE_VERSION)
