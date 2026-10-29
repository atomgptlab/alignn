"""Tests for the bulk multi-model screening CLI (alignn/screen.py).

Uses a randomly-initialized pure-torch model saved to a temp model dir, so
the test runs offline (no figshare download).
"""

import gzip
import json
import os
import sys

import torch

from alignn import screen

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "sample_data_atomwise",
    "id_prop.json",
)


def make_toy_model_dir(path):
    """Write config.json + best_model.pt for a tiny random pure model."""
    from alignn.models.alignn_atomwise_pure import (
        ALIGNNAtomWisePure,
        ALIGNNAtomWisePureConfig,
    )

    mcfg = {
        "name": "alignn_atomwise_pure",
        "alignn_layers": 1,
        "gcn_layers": 1,
        "atom_input_features": 92,
        "hidden_features": 32,
        "embedding_features": 32,
        "output_features": 1,
        "calculate_gradient": False,
    }
    model = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**mcfg))
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "best_model.pt"))
    config = {
        "cutoff": 5.0,
        "max_neighbors": 12,
        "three_body_cutoff": 3.5,
        "atom_features": "cgcnn",
        "model": mcfg,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def write_inputs(tmp_path):
    entries = json.load(open(EXAMPLE))[:5]
    records = [
        {"id": e["jid"], "atoms": e["atoms"]} for e in entries
    ]
    input_path = os.path.join(str(tmp_path), "structures.json")
    with open(input_path, "w") as f:
        json.dump(records, f)
    manifest = {
        "properties": {
            "toy_prop": {
                "figshare": None,
                "dir": os.path.join(str(tmp_path), "toy_model"),
            }
        }
    }
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    make_toy_model_dir(os.path.join(str(tmp_path), "toy_model"))
    return input_path, manifest_path


def run_screen(input_path, manifest_path, out_dir, extra=None):
    argv = [
        "screen.py",
        "--input", input_path,
        "--manifest", manifest_path,
        "--models", "all",
        "--device", "cpu",
        "--format", "jsonl",
        "--batch_size", "2",
        "--shard_size", "3",
        "--output_dir", out_dir,
    ] + (extra or [])
    old_argv = sys.argv
    sys.argv = argv
    try:
        screen.main()
    finally:
        sys.argv = old_argv


def read_shard(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_screen_writes_shards_and_resumes(tmp_path):
    input_path, manifest_path = write_inputs(tmp_path)
    out_dir = os.path.join(str(tmp_path), "out")

    run_screen(input_path, manifest_path, out_dir)
    shard0 = os.path.join(out_dir, "shard_00000.jsonl.gz")
    shard1 = os.path.join(out_dir, "shard_00001.jsonl.gz")
    assert os.path.exists(shard0) and os.path.exists(shard1)
    rows = read_shard(shard0) + read_shard(shard1)
    assert len(rows) == 5
    for row in rows:
        assert set(row) == {"id", "formula", "natoms", "toy_prop"}
        assert isinstance(row["toy_prop"], float)
    run_info = json.load(open(os.path.join(out_dir, "manifest_run.json")))
    assert run_info["n_structures"] == 5
    assert run_info["n_failed"] == 0

    # resume: drop one shard, rerun; the surviving shard must be skipped
    # (unchanged mtime) and the missing one regenerated
    os.remove(shard1)
    mtime0 = os.path.getmtime(shard0)
    run_screen(input_path, manifest_path, out_dir)
    assert os.path.exists(shard1)
    assert os.path.getmtime(shard0) == mtime0


def test_screen_limit(tmp_path):
    input_path, manifest_path = write_inputs(tmp_path)
    out_dir = os.path.join(str(tmp_path), "out_limited")
    run_screen(input_path, manifest_path, out_dir, extra=["--limit", "2"])
    rows = read_shard(os.path.join(out_dir, "shard_00000.jsonl.gz"))
    assert len(rows) == 2
