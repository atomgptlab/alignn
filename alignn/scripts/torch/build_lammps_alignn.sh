#!/bin/bash
# ---------------------------------------------------------------------------
# build_lammps_alignn.sh
#
# One-shot build of LAMMPS stable_29Aug2024_update1 with the native
# pair_alignn style linked against the conda-env libtorch.
#
# Does:
#   1. Discovers PyTorch/CUDA/ABI in the active conda env
#   2. (Optional) installs matching cuda-toolkit + MKL if missing
#   3. Clones LAMMPS to $LAMMPS_DIR (default ~/lammps-alignn)
#   4. Copies pair_alignn.{cpp,h} from this repo into LAMMPS src/
#   5. Appends find_package(Torch) + target_link_libraries to main CMakeLists.txt
#   6. Configures + builds with Ninja
#   7. Installs the lammps Python module into the active env
#   8. Overwrites $CONDA_PREFIX/lib/liblammps.so.0 with the pair_alignn build
#      (fixes the standalone `lmp` binary's RPATH picking up a stale .so)
#   9. Verifies "alignn" appears in both Python module and the standalone binary
#
# Env vars:
#   LAMMPS_DIR       (default: $HOME/lammps-alignn)
#   LAMMPS_TAG       (default: stable_29Aug2024_update1)
#   SKIP_CUDA_MKL    set to 1 to skip the mamba install step
#   JOBS             (default: $(nproc))
# ---------------------------------------------------------------------------
set -euo pipefail

LAMMPS_TAG="${LAMMPS_TAG:-stable_29Aug2024_update1}"
LAMMPS_DIR="${LAMMPS_DIR:-$HOME/lammps-alignn}"
JOBS="${JOBS:-$(nproc)}"

# Locate the alignn repo root from *this* script's path (so it's portable).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_SRC="$SCRIPT_DIR/pair_alignn"
ALIGNN_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "── paths ────────────────────────────────"
echo "script dir   : $SCRIPT_DIR"
echo "alignn repo  : $ALIGNN_REPO"
echo "pair sources : $PAIR_SRC"
echo "LAMMPS target: $LAMMPS_DIR (tag $LAMMPS_TAG)"
echo

# ── sanity checks ────────────────────────────────────────────────────────
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: CONDA_PREFIX not set. Activate a conda env first." >&2
    exit 1
fi
if ! command -v cmake >/dev/null; then
    echo "ERROR: cmake not found. `mamba install cmake` or install it." >&2
    exit 1
fi

# ── discover torch + CUDA ABI ────────────────────────────────────────────
TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_CMAKE="$TORCH_DIR/share/cmake/Torch"
TORCH_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'cpu')")

echo "── environment ──────────────────────────"
echo "python           = $(which python)"
echo "torch            = $(python -c 'import torch; print(torch.__version__)')"
echo "torch CUDA       = $TORCH_CUDA"
echo "torch CXX11 ABI  = $TORCH_CXX11_ABI"
echo "cmake            = $(cmake --version | head -1)"
echo "CONDA_PREFIX     = $CONDA_PREFIX"
echo

# ── ensure CUDA toolkit + MKL match libtorch ─────────────────────────────
need_install=0
if [ "${SKIP_CUDA_MKL:-0}" != "1" ] && [ "$TORCH_CUDA" != "cpu" ]; then
    # CUDA toolkit check: does nvcc match torch's CUDA?
    if command -v nvcc >/dev/null; then
        NVCC_VER=$(nvcc --version | awk '/release/{gsub(",","",$5); print $5}')
    else
        NVCC_VER="none"
    fi
    if [ "$NVCC_VER" != "$TORCH_CUDA" ]; then
        echo "[!] nvcc=$NVCC_VER does not match torch CUDA=$TORCH_CUDA"
        need_install=1
    fi
    # MKL header check
    if [ ! -f "$CONDA_PREFIX/include/mkl.h" ]; then
        echo "[!] MKL headers not found (mkl.h missing)"
        need_install=1
    fi
    if [ $need_install -eq 1 ]; then
        echo "── installing cuda-toolkit=$TORCH_CUDA + mkl-devel via mamba ──"
        mamba install -y -c nvidia -c conda-forge \
            "cuda-toolkit=$TORCH_CUDA" mkl-devel mkl-include
    else
        echo "CUDA toolkit + MKL already satisfy torch requirements."
    fi
fi

# ── clone LAMMPS ─────────────────────────────────────────────────────────
if [ ! -d "$LAMMPS_DIR" ]; then
    echo "[1/5] cloning LAMMPS $LAMMPS_TAG..."
    git clone --depth=1 --branch "$LAMMPS_TAG" \
        https://github.com/lammps/lammps.git "$LAMMPS_DIR"
else
    echo "[1/5] LAMMPS dir exists at $LAMMPS_DIR, reusing"
fi

# ── drop pair_alignn into src/ (no USER- package machinery) ──────────────
echo "[2/5] installing pair_alignn.{cpp,h} into LAMMPS src/..."
cp "$PAIR_SRC/pair_alignn.cpp" "$LAMMPS_DIR/src/pair_alignn.cpp"
cp "$PAIR_SRC/pair_alignn.h"   "$LAMMPS_DIR/src/pair_alignn.h"

# ── append unconditional libtorch link to the main CMakeLists.txt ────────
MAIN_CMAKE="$LAMMPS_DIR/cmake/CMakeLists.txt"
HOOK_MARKER="# ALIGNN-FF libtorch hook"
if ! grep -q "$HOOK_MARKER" "$MAIN_CMAKE"; then
    cat >> "$MAIN_CMAKE" <<EOF

# ── ALIGNN-FF (libtorch) ──────────────────────────────────────────────────
$HOOK_MARKER
# pair_alignn.cpp lives directly in src/ and is picked up by the main glob.
# We only need to supply libtorch includes + link.
find_package(Torch REQUIRED)
target_link_libraries(lammps PRIVATE \${TORCH_LIBRARIES})
target_compile_features(lammps PRIVATE cxx_std_17)
EOF
    echo "  appended libtorch hook to cmake/CMakeLists.txt"
else
    echo "  libtorch hook already present"
fi

# ── configure + build ────────────────────────────────────────────────────
BUILD_DIR="$LAMMPS_DIR/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CUDA_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
CUDA_FLAG=""
[ -d "$CUDA_INC" ] && CUDA_FLAG="-I${CUDA_INC}"

echo "[3/5] cmake configure..."
cmake ../cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLAMMPS_EXCEPTIONS=ON \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;$CONDA_PREFIX" \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CUDA_FLAG}" \
    -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DPython_EXECUTABLE="$(which python)" \
    -DCMAKE_CUDA_COMPILER=$(command -v nvcc 2>/dev/null || echo "") \
    2>&1 | tail -8

echo
echo "[4/5] compiling (this takes 5–15 min)..."
cmake --build . -j"$JOBS" 2>&1 | tail -8

# ── install the lammps Python module ─────────────────────────────────────
echo "[5/5] installing lammps Python module into $CONDA_PREFIX..."
cmake --build . --target install-python 2>&1 | tail -5

# ── fix standalone binary's RPATH issue by overwriting conda liblammps ───
if [ -f "$CONDA_PREFIX/lib/liblammps.so.0" ]; then
    echo "  syncing liblammps.so.0 → $CONDA_PREFIX/lib/ (for standalone lmp binary)"
    cp "$BUILD_DIR/liblammps.so.0" "$CONDA_PREFIX/lib/liblammps.so.0"
fi

# ── verify ───────────────────────────────────────────────────────────────
echo
echo "── verification ─────────────────────────"

PY_OK=$(python - <<'PYEOF'
from lammps import lammps
l = lammps()
ok = "alignn" in l.available_styles("pair")
print("YES" if ok else "NO")
l.close()
PYEOF
)
if [ "$PY_OK" = "YES" ]; then
    echo "✓ Python module has pair_alignn"
else
    echo "✗ Python module does NOT have pair_alignn"
fi

if "$BUILD_DIR/lmp" -help 2>&1 | grep -q alignn; then
    echo "✓ Standalone binary has pair_alignn:  $BUILD_DIR/lmp"
else
    echo "✗ Standalone binary does NOT have pair_alignn"
fi

echo
echo "──────────────────────────────────────────"
echo "Done. Try:"
echo "  export_torchscript.py --model-dir <OutputDir> --out alignn_ff.pt"
echo "  $BUILD_DIR/lmp -in <your_input.in>"
echo "──────────────────────────────────────────"
