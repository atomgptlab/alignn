"""Download the default ALIGNN-FF `mps` model from figshare and export
it to TorchScript (`alignn_ff.pt`) for the native LAMMPS `pair_alignn`.

Run once before the LAMMPS examples:
    python get_model.py
"""
import os
import subprocess
import sys

from alignn.ff.ff import get_figshare_model_ff


def default_path():
    """Default ALIGNN-FF model path (downloads on first call)."""
    return get_figshare_model_ff(model_name="mps")


def main():
    model_dir = default_path()
    print(f"model dir: {model_dir}")
    assert os.path.exists(os.path.join(model_dir, "best_model.pt")), (
        f"best_model.pt not found in {model_dir}"
    )
    assert os.path.exists(os.path.join(model_dir, "config.json")), (
        f"config.json not found in {model_dir}"
    )

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "alignn_ff.pt")
    cmd = [sys.executable,
           "-m", "alignn.scripts.torch.export_torchscript",
           "--model-dir", model_dir,
           "--out", out]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
