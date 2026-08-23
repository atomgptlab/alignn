#!/usr/bin/env python3
"""Upload a trained model to the ALIGNN2 figshare project.

Packages a model directory into one flat zip, creates an article inside the
project, uploads the zip through figshare's chunked upload protocol, and
prints the registry fields needed by ``alignn/pretrained.py``.

The article is left **unpublished** unless ``--publish`` is given: publishing
mints a permanent DOI and cannot be undone, so it is deliberately a separate,
explicit decision.

Requires a personal token in ``FIGSHARE_TOKEN``.

    python scripts/figshare/upload_model.py \\
        --model-dir runs/csp_supercon_jarvis --name csp_supercon_jarvis \\
        --description "..." --tags alignn2 inverse csp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests

API = "https://api.figshare.com/v2"
ALIGNN2_PROJECT = 279395
# Matches the category the existing ALIGNN2 articles use.
DEFAULT_CATEGORY = 30166  # Condensed matter modelling and DFT
CHUNK = 8 * 1024 * 1024


def _headers() -> dict:
    token = os.environ.get("FIGSHARE_TOKEN")
    if not token:
        raise SystemExit("FIGSHARE_TOKEN is not set")
    return {"Authorization": f"token {token}"}


def _check(r: requests.Response, what: str) -> dict:
    """Raise on failure; return the body as a dict when it is JSON.

    Not every step answers in JSON: a part upload replies with the bare text
    ``OK`` and the completion step replies 202 with an HTML page, so parsing
    unconditionally would fail on a successful call.
    """
    if not r.ok:
        raise SystemExit(f"{what} failed [{r.status_code}]: {r.text[:400]}")
    if "json" not in r.headers.get("Content-Type", ""):
        return {}
    try:
        return r.json()
    except ValueError:
        return {}


def make_zip(model_dir: Path, name: str, out_dir: Path) -> Path:
    """One flat zip of the model directory, matching the ALIGNN2 layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"
    files = sorted(p for p in model_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no files in {model_dir}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)  # flat: no directory prefix
    return zip_path


def md5_and_size(path: Path):
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest(), path.stat().st_size


def create_article(meta: dict, project: int) -> int:
    r = requests.post(
        f"{API}/account/projects/{project}/articles",
        headers=_headers(),
        json=meta,
        timeout=60,
    )
    loc = _check(r, "create article")["location"]
    return int(loc.rstrip("/").split("/")[-1])


def upload_file(article_id: int, path: Path) -> int:
    """Figshare's initiate / parts / PUT / complete upload dance."""
    md5, size = md5_and_size(path)
    r = requests.post(
        f"{API}/account/articles/{article_id}/files",
        headers=_headers(),
        json={"name": path.name, "md5": md5, "size": size},
        timeout=60,
    )
    file_url = _check(r, "initiate upload")["location"]
    file_id = int(file_url.rstrip("/").split("/")[-1])

    info = _check(
        requests.get(file_url, headers=_headers(), timeout=60), "file info"
    )
    parts = _check(
        requests.get(info["upload_url"], headers=_headers(), timeout=60),
        "part list",
    )["parts"]

    with path.open("rb") as fh:
        for part in parts:
            fh.seek(part["startOffset"])
            data = fh.read(part["endOffset"] - part["startOffset"] + 1)
            _check(
                requests.put(
                    f"{info['upload_url']}/{part['partNo']}",
                    headers=_headers(),
                    data=data,
                    timeout=600,
                ),
                f"upload part {part['partNo']}",
            )
            print(f"    part {part['partNo']}/{len(parts)}", flush=True)

    _check(
        requests.post(file_url, headers=_headers(), timeout=120),
        "complete upload",
    )
    return file_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--tags", nargs="*", default=["alignn2"])
    ap.add_argument("--project", type=int, default=ALIGNN2_PROJECT)
    ap.add_argument("--category", type=int, default=DEFAULT_CATEGORY)
    ap.add_argument("--title", default=None)
    ap.add_argument("--zip-dir", default=None)
    ap.add_argument(
        "--publish",
        action="store_true",
        help="mint a DOI and make the article public; permanent",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    zip_dir = (
        Path(args.zip_dir) if args.zip_dir else model_dir.parent / "_zips"
    )
    zip_path = make_zip(model_dir, args.name, zip_dir)
    print(f"  zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    article_id = create_article(
        {
            "title": args.title or f"ALIGNN2 {args.name}",
            "description": args.description,
            "defined_type": "dataset",
            "categories": [args.category],
            "tags": list(args.tags),
            "license": 1,  # CC BY 4.0
        },
        args.project,
    )
    print(f"  article: {article_id}")

    file_id = upload_file(article_id, zip_path)
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    print(f"  file: {file_id}")

    if args.publish:
        _check(
            requests.post(
                f"{API}/account/articles/{article_id}/publish",
                headers=_headers(),
                timeout=120,
            ),
            "publish",
        )
        print("  published (DOI minted)")
    else:
        print(
            "  left unpublished; re-run with --publish, or publish in the UI"
        )

    print(
        json.dumps(
            {
                "name": args.name,
                "figshare_article_id": article_id,
                "file_id": file_id,
                "url": url,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
